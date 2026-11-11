# -*- coding: utf-8 -*-
"""AsyncTLS 语义复现的 Qwen3 适配层（C 轨语义复现，非官方源码）。

背景与边界（2026-10-17，deepseek 轨）：
  AsyncTLS（arXiv:2604.07815，人大+美团）截至本文件尚无官方开源仓库，
  按用户明确授权「无源码则读论文给出近似实现」，本文件为语义复现：
  **与官方源码无关，非逐字节一致，任何算法细节以论文为主、本文注释为准**。

复现的三级流水（论文 §4）：
  Stage-1 块粗筛（无需校准，动态）：key 按 B=64 切块，每块 min/max 代表，
    符号感知 GEMM 化打分 s = Σ_h(qmax_h·kmax + qmin_h·kmin)，取 top-kb=128 块
    （≈8192 token 候选）。
  Stage-2 token 细筛（需校准 channel 集 C）：q/k 投影到每层 top-dc=32 的
    校准 channel；k̃ 做 INT4 量化-反量化（本实现 per-token 16 级，数学等价
    于 INT4 索引精度，但未压缩足迹，已在 identity 标注）；组内平均
    ᾱ_j = (1/G)Σ_h softmax(q̃_h·k̃_j / √d)；在 Stage-1 选中块内 top-kt。
  Stage-3：原始全维 Q/K/V 在 S_t 上计算注意力（无 sink/swa 保送，kt 即
    最终 token 数）。

复现口径（对齐论文 §5）：
  block_size=64、top_kb=128 块、channel dc=32（GQA）、INT4 索引量化、
  kt：RULER 512 / LongBench 1024（由 token_budget 传入）、Qwen3-8B。
  论文未声明前层 dense，本复现全层稀疏（layer 0 也走稀疏）。

已知近似（语义复现边界，均写入 _baseline_identity）：
  - GQA 组内平均：Qwen3 每 KV 组 4 个 query head，块打分与细筛均先组内
    逐 query 算再平均（论文 MQA/MLA 场景无此细节）。
  - INT4 量化-反量化不压缩足迹（只复现精度行为）。
  - channel 校准数据集采用与 FASA 同口径的 held-out 样本（见
    identify_channel.py），非论文原始 Dcal。
"""
import json
import math
import os
import types
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb, repeat_kv,
)


def _load_channels(channel_file: str) -> list:
    """读取 ``[layer][kv_group][channel]`` 校准索引，拒绝旧全局格式。"""
    with open(channel_file, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if isinstance(raw.get("channels"), list):
            layers = raw["channels"]
        else:
            # Keep legacy numeric-suffix keys, but require complete native IDs.
            try:
                entries = [(int(key.split("_")[-1]), value)
                           for key, value in raw.items()]
            except (AttributeError, ValueError) as error:
                raise ValueError("channel keys must end in integer layer IDs") from error
            if {idx for idx, _ in entries} != set(range(len(entries))):
                raise ValueError("channel layer IDs must uniquely cover 0..N-1")
            layers = [value for _, value in sorted(entries)]
    else:
        layers = list(raw)
    for layer in layers:
        if not isinstance(layer, (list, tuple)) or not layer:
            raise ValueError("channel file must hold per-layer per-KV-group lists")
        for idx_list in layer:
            if (not isinstance(idx_list, (list, tuple)) or not idx_list
                    or not all(type(i) is int for i in idx_list)):
                raise ValueError(
                    "AsyncTLS requires [layer][kv_group][channel] per-KV-group indices; "
                    "legacy per-layer global channels are not paper-aligned")
    return [[list(group) for group in layer] for layer in layers]


def _validate_channels_for_modules(modules, channels, expected_layers,
                                   dimension_attr="head_dim"):
    """Preflight nested per-KV-group calibration without mutating the model."""
    layer_ids = [int(module.layer_idx) for module in modules]
    if (len(modules) != expected_layers
            or set(layer_ids) != set(range(expected_layers))):
        raise ValueError("AsyncTLS attention layers must uniquely cover config layers")
    if len(channels) != expected_layers:
        raise ValueError(
            f"channel layers {len(channels)} != model layers {expected_layers}")
    for module in modules:
        if hasattr(module, "asynctls_channels"):
            raise ValueError("AsyncTLS adapter already installed; use an unpatched model")
        layer_id = int(module.layer_idx)
        groups = channels[layer_id]
        if len(groups) != int(module.config.num_key_value_heads):
            raise ValueError(f"AsyncTLS layer {layer_id} KV group count is invalid")
        widths = {len(indices) for indices in groups}
        if len(widths) != 1:
            raise ValueError(f"AsyncTLS layer {layer_id} channel widths must match across groups")
        for indices in groups:
            if (not indices or len(set(indices)) != len(indices)
                    or min(indices) < 0
                    or max(indices) >= getattr(module, dimension_attr)):
                raise ValueError(f"AsyncTLS layer {layer_id} channel indices are invalid")


def _int4_quant_dequant(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """per-token 16 级量化-反量化（语义等价 INT4 索引精度，不压缩足迹）。

    x: [..., L, C]。对最后一行（token）做 min/max 归一化到 [-8, 7]。
    """
    levels = 2 ** bits
    lo = -levels // 2
    hi = lo + levels - 1
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    scale = (mx - mn) / (levels - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round((x - mn) / scale + lo).clamp(lo, hi)
    return (q - lo) * scale + mn


def _qwen3_block_score(query, key, block_size: int):
    """Stage-1：每个 KV group 独立打分，返回 ``[kv_heads, n_blocks]``。"""
    kv_heads, length, head_dim = key.shape[1], key.shape[2], key.shape[-1]
    groups = query.shape[1] // kv_heads
    n_blocks = length // block_size
    key_blocks = key[..., : n_blocks * block_size, :].reshape(
        1, kv_heads, n_blocks, block_size, head_dim)
    kmax = key_blocks.amax(dim=3)
    kmin = key_blocks.amin(dim=3)
    query_grouped = query[0, :, 0, :].reshape(kv_heads, groups, head_dim)
    qmax = query_grouped.clamp(min=0)
    qmin = query_grouped.clamp(max=0)
    return torch.einsum("hgd,hmd->hm", qmax, kmax[0]) \
        + torch.einsum("hgd,hmd->hm", qmin, kmin[0])


def _padded_key(key_states, block_size):
    pad = (block_size - key_states.shape[2] % block_size) % block_size
    if not pad:
        return key_states
    key_pad = torch.zeros(
        1, key_states.shape[1], pad, key_states.shape[-1],
        device=key_states.device, dtype=key_states.dtype)
    return torch.cat([key_states, key_pad], dim=2)


def _select_blocks(query_states, padded_key, block_size, top_kb):
    scores = _qwen3_block_score(query_states, padded_key, block_size)
    n_blocks = min(top_kb, padded_key.shape[2] // block_size)
    return torch.topk(scores, k=n_blocks, dim=-1).indices


def _candidate_indices(top_blocks, length, block_size):
    """逐 KV group 取上一步块集，并并入当前 token 后去重排序。"""
    current = torch.tensor([length - 1], device=top_blocks.device)
    candidates = []
    for group_blocks in top_blocks:
        pieces = [
            torch.arange(
                int(block * block_size),
                min(int((block + 1) * block_size), length),
                device=top_blocks.device)
            for block in group_blocks.tolist()
        ]
        pieces.append(current)
        candidates.append(torch.unique(torch.cat(pieces)))
    return candidates


def _pad_candidates(candidates):
    n_max = max(candidate.numel() for candidate in candidates)
    indices = torch.zeros(
        len(candidates), n_max, device=candidates[0].device, dtype=torch.long)
    valid = torch.zeros(len(candidates), n_max, device=indices.device, dtype=torch.bool)
    for group, candidate in enumerate(candidates):
        indices[group, :candidate.numel()] = candidate
        valid[group, :candidate.numel()] = True
    return indices, valid


def _select_tokens(candidate_idx, candidate_valid, candidate_score, token_budget):
    """逐组 top-k；当前 token 强制入选，padding 不占有效预算。"""
    current = candidate_idx.max().item()
    is_current = candidate_valid & candidate_idx.eq(current)
    scores = candidate_score.masked_fill(~candidate_valid, -torch.inf)
    scores = scores.masked_fill(is_current, torch.inf)
    counts = candidate_valid.sum(dim=1)
    selected_counts = counts.clamp(max=token_budget)
    k_max = int(selected_counts.max().item())
    if k_max <= 0:
        raise RuntimeError("AsyncTLS 候选为空，无法执行稀疏 decode")
    selected_local = torch.topk(scores, k=k_max, dim=-1).indices
    selected_idx = torch.gather(candidate_idx, 1, selected_local)
    selected_valid = torch.gather(candidate_valid, 1, selected_local)
    rank = torch.arange(k_max, device=candidate_idx.device).unsqueeze(0)
    selected_valid &= rank < selected_counts.unsqueeze(1)
    selected_idx = torch.where(selected_valid, selected_idx, current)
    selected_idx, order = torch.sort(selected_idx, dim=-1)
    selected_valid = torch.gather(selected_valid, 1, order)
    return selected_idx, selected_valid


def _asynctls_sparse_attention(module, query_states, key_states, value_states,
                               previous_blocks):
    """Stage-2/3：每个 KV group 使用自己的 channel、候选和 token 选集。"""
    length = key_states.shape[2]
    kv_heads = key_states.shape[1]
    query_heads = query_states.shape[1]
    groups = query_heads // kv_heads
    head_dim = key_states.shape[-1]
    candidates = _candidate_indices(
        previous_blocks, length, module.asynctls_block_size)
    candidate_idx, candidate_valid = _pad_candidates(candidates)

    channels = torch.tensor(
        module.asynctls_channels, device=query_states.device, dtype=torch.long)
    channel_count = channels.shape[1]
    query_grouped = query_states[0, :, 0, :].reshape(kv_heads, groups, head_dim)
    query_projected = torch.gather(
        query_grouped, 2,
        channels[:, None, :].expand(kv_heads, groups, channel_count))
    candidate_key = torch.gather(
        key_states[0], 1,
        candidate_idx[..., None].expand(kv_heads, candidate_idx.shape[1], head_dim))
    candidate_key = torch.gather(
        candidate_key, 2,
        channels[:, None, :].expand(
            kv_heads, candidate_idx.shape[1], channel_count))
    candidate_key = _int4_quant_dequant(candidate_key)
    logits = torch.einsum("hgc,hjc->hgj", query_projected, candidate_key)
    logits = logits * module.scaling
    logits = logits.masked_fill(~candidate_valid[:, None, :], -torch.inf)
    candidate_score = F.softmax(logits.float(), dim=-1).mean(dim=1)

    selected_idx, selected_valid = _select_tokens(
        candidate_idx, candidate_valid, candidate_score,
        module.asynctls_token_budget)
    selected_count = selected_idx.shape[1]
    selected_key = torch.gather(
        key_states[0], 1,
        selected_idx[..., None].expand(kv_heads, selected_count, head_dim))
    selected_value = torch.gather(
        value_states[0], 1,
        selected_idx[..., None].expand(kv_heads, selected_count, head_dim))
    attn_weights = torch.matmul(
        query_grouped, selected_key.transpose(-2, -1)) * module.scaling
    attn_weights = attn_weights.masked_fill(
        ~selected_valid[:, None, :], -torch.inf)
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, selected_value)
    attn_output = attn_output.reshape(1, query_heads, 1, head_dim)
    return attn_output, attn_weights.reshape(1, query_heads, 1, selected_count)


def _cache_keys(cache, layer_idx):
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys
    return cache[layer_idx][0]


def _qwen3_initialize_blocks(module, hidden_states, position_embeddings, cache):
    """dense prefill 后，用 prompt 最后位置建立第一步 decode 的 M_{t-1}。"""
    if cache is None:
        module._asynctls_prev_blocks = None
        return
    hidden = hidden_states[:, -1:, :]
    hidden_shape = (hidden.shape[0], 1, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    cos, sin = position_embeddings
    query, _ = apply_rotary_pos_emb(query, key, cos[:, -1:, :], sin[:, -1:, :])
    cached_key = _cache_keys(cache, module.layer_idx)
    padded_key = _padded_key(cached_key, module.asynctls_block_size)
    module._asynctls_prev_blocks = _select_blocks(
        query, padded_key, module.asynctls_block_size, module.asynctls_top_kb).detach()


def _qwen3_asynctls_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """dense prefill 初始化块集；decode 使用 M_{t-1} 并保存当前 M_t。"""
    q_len = hidden_states.shape[1]
    if q_len > 1:
        output = self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs,
        )
        _qwen3_initialize_blocks(self, hidden_states, position_embeddings, past_key_values)
        return output

    previous_length = (
        int(past_key_values.get_seq_length(self.layer_idx))
        if past_key_values is not None else 0)
    if cache_position is None and past_key_values is not None:
        cache_position = torch.arange(
            previous_length, previous_length + q_len,
            device=hidden_states.device)

    from sparse_attn.baseline_runtime import require_unmasked_decode
    require_unmasked_decode(hidden_states, attention_mask, past_key_values, cache_position)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    padded_key = _padded_key(key_states, self.asynctls_block_size)
    current_blocks = _select_blocks(
        query_states, padded_key, self.asynctls_block_size, self.asynctls_top_kb)
    if previous_length == 0:
        previous_blocks = current_blocks
    else:
        previous_blocks = getattr(self, "_asynctls_prev_blocks", None)
        if previous_blocks is None:
            raise RuntimeError("AsyncTLS decode 缺少 dense prefill 初始化的 M_{t-1}")
    self._asynctls_prev_blocks = current_blocks.detach()
    attn_output, attn_weights = _asynctls_sparse_attention(
        self, query_states, key_states, value_states, previous_blocks)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def enable_asynctls(
    model,
    channel_file: str,
    token_budget: int = 1024,
    block_size: int = 64,
    top_kb: int = 128,
) -> int:
    """把 model 内全部 Qwen3Attention 绑定 AsyncTLS 三级稀疏 forward。

    返回 patch 层数。channel_file 指向 identify_channel.py 生成的校准索引。
    """
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("AsyncTLS token_budget must be a positive integer")
    if type(block_size) is not int or block_size <= 0 or block_size % 8:
        raise ValueError("AsyncTLS block_size must be positive multiple of 8")
    if type(top_kb) is not int or top_kb <= 0:
        raise ValueError("AsyncTLS top_kb must be a positive integer")
    channels = _load_channels(channel_file)
    modules = [module for module in model.modules()
               if isinstance(module, Qwen3Attention)]
    _validate_channels_for_modules(
        modules, channels, int(model.config.num_hidden_layers))
    for module in modules:
        module.layer_id = int(module.layer_idx)
        module.asynctls_channels = channels[module.layer_id]
        module.asynctls_token_budget = token_budget
        module.asynctls_block_size = block_size
        module.asynctls_top_kb = top_kb
        module._asynctls_prev_blocks = None
        module.flash_forward = module.forward
        module.forward = types.MethodType(_qwen3_asynctls_forward, module)
    model._asynctls_full_prompt_prefill = True
    return len(modules)
