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
  - GQA：论文按 KV 组选择；本适配在组内 query 聚合之外进一步跨 KV
    组聚合块分数和细筛概率，所有头共享选集，是额外的适配约定。
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
    """读取校准 channel 索引。支持三种格式：
    [ [idx...] x layers ]、{"channels": [[...]xL]}（identify_channel.py
    输出）、{"layer_0": [...], ...}。"""
    with open(channel_file, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if isinstance(raw.get("channels"), list):
            layers = raw["channels"]
        else:
            layers = [raw[k] for k in sorted(raw, key=lambda k: int(k.split("_")[-1]))]
    else:
        layers = list(raw)
    for idx_list in layers:
        if not isinstance(idx_list, (list, tuple)) or not all(
                isinstance(i, int) for i in idx_list):
            raise ValueError("channel file must hold per-layer integer index lists")
    return layers


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
    """Stage-1 块粗筛：GEMM 化符号感知打分，返回每块分数 [m]。

    query: [1, n_qh, 1, d]（n_qh=32，Qwen3 GQA，8 组 × 4）；key: [1, n_kvh, L, d]。
    按 kv head 分组：第 h 个 kv head 服务 query head h*G..h*G+G-1。
    """
    kv_heads, L, d = key.shape[1], key.shape[2], key.shape[-1]
    groups = query.shape[1] // kv_heads
    m = L // block_size
    # 块 min/max：[1, kv_heads, m, d]
    key_blocks = key[..., : m * block_size, :].reshape(1, kv_heads, m, block_size, d)
    kmax = key_blocks.amax(dim=3)
    kmin = key_blocks.amin(dim=3)
    # query 分组成 [1, kv_heads, G, 1, d]
    q = query[0, :, 0, :].reshape(kv_heads, groups, 1, d)
    qmax = q.clamp(min=0)
    qmin = q.clamp(max=0)
    # s[h] = Σ_g (qmax[h,g]·kmax[h] + qmin[h,g]·kmin[h])，跨组求和后再跨 kv head 求和
    s_h = torch.einsum("hgd,hmd->hm", qmax[..., 0, :], kmax[0]) \
        + torch.einsum("hgd,hmd->hm", qmin[..., 0, :], kmin[0])
    return s_h.sum(dim=0)  # [m]


def _qwen3_asynctls_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Qwen3Attention forward：prefill 走原逻辑；decode 走 AsyncTLS 三级流水。"""
    q_len = hidden_states.shape[1]
    if q_len > 1:
        # Transformers 5.x 删除了 cache_position 位置参数；原始 prefill
        # 前向也不需要它（cache_position 仅用于下方 decode cache 更新）。
        return self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs,
        )

    if cache_position is None and past_key_values is not None:
        # Transformers 5.x 不向 attention 透传 cache_position；从现有 cache
        # 长度恢复同一位置，仅供契约校验和 cache 更新，RoPE 仍用原生输入。
        previous_length = int(past_key_values.get_seq_length(self.layer_idx))
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

    L = key_states.shape[2]
    B = self.asynctls_block_size
    # Repeat a real tail key for score-only padding: zeros would change the
    # partial block extrema (and can promote it over a better real block).
    # Candidate indices below still exclude every padded position.
    pad = (B - L % B) % B
    if pad:
        key_pad = key_states[..., -1:, :].expand(-1, -1, pad, -1)
        key_states = torch.cat([key_states, key_pad], dim=2)
    m = key_states.shape[2] // B

    # ---- Stage-1 块粗筛 ----
    block_score = _qwen3_block_score(query_states, key_states, B)
    top_kb = self.asynctls_top_kb
    # 末块（可能含 pad）不整块丢弃：候选构建时 cand_idx 已用
    # min((b+1)*B, L) 排除 pad 区 token，末块真实 token 仍参与 Stage-2/3。
    # 2026-10-17 修复：旧逻辑把含 pad 的末块整块置 -inf 并排除，导致每次
    # decode 丢弃最近一整块真实上下文（短序列破坏问句、长序列每步丢 64
    # token），RULER 32K 全崩（重复输出）。n_blocks 下限 1 保证超短序列
    # （m=1，唯一块含 pad）仍以全序列候选交给 Stage-2/3 细筛。
    n_blocks = min(top_kb, m)
    top_blocks = torch.topk(block_score, k=n_blocks).indices  # [n_blocks]
    # 候选 token 索引（选中块内真实 token，排除 pad）
    cand_idx = torch.cat(
        [torch.arange(int(b * B), min(int((b + 1) * B), L),
                      device=key_states.device) for b in top_blocks.tolist()])
    cand_idx = torch.unique(cand_idx)
    k_cand = key_states[..., cand_idx, :]      # [1, kv_heads, n_cand, d]
    v_cand = value_states[..., cand_idx, :]

    # ---- Stage-2 组内 channel 细筛 ----
    # enable_asynctls 已按层把整份 channel list 存进 asynctls_channels，
    # 这里直接用（勿再按 layer_idx 索引——list[0] 会取到该层首个 channel）
    C = torch.tensor(self.asynctls_channels,
                     device=query_states.device, dtype=torch.long)
    q_tilde = query_states[..., C]            # [1, n_qh, 1, c]
    k_tilde = _int4_quant_dequant(k_cand[..., C])  # [1, kv_heads, n_cand, c]
    # 组内平均 softmax：每组 4 个 query head 对同一 kv head 打分，取平均
    n_qh, c = q_tilde.shape[1], q_tilde.shape[-1]
    kv_heads = k_tilde.shape[1]
    groups = n_qh // kv_heads
    # [kv_heads, G, c] @ [kv_heads, n_cand, c] -> [kv_heads, G, n_cand]（channel 求和）
    q_tilde_g = q_tilde[0].reshape(kv_heads, groups, c)
    logits_g = torch.einsum("hgc,hjc->hgj", q_tilde_g, k_tilde[0]) * self.scaling
    prob_g = F.softmax(logits_g.float(), dim=-1).to(q_tilde.dtype)  # [kv_heads, G, n_cand]
    tilde_alpha = prob_g.mean(dim=1)          # [kv_heads, n_cand]，GQA 组内平均
    # P0 排查（2026-10-24→revert）：max 聚合实验结果=0.40（灾难），mean=19.90。
    # GQA 跨 head 平均是必要的去噪机制，max 过度放大噪声 → revert 回 mean。
    # 精度差距根因不在此处，待排查其他候选（scorer口径 / INT4 / Stage-1 参数等）。
    cand_alpha = tilde_alpha.mean(dim=0)      # [n_cand]，跨 kv head 平均

    # ---- Stage-3 全维 KV 上 top-kt ----
    kt = min(self.asynctls_token_budget, cand_alpha.numel())
    if kt <= 0:
        raise RuntimeError("AsyncTLS 候选为空，无法执行稀疏 decode")
    sel_local = torch.topk(cand_alpha, k=kt).indices
    # 保持原始 token 顺序：topk 返回降序索引，若直接用 cand_idx[sel_local] 会
    # 得到乱序的 k_sel/v_sel。softmax 数学上对 token 排列不变，但 bfloat16
    # matmul 对排列顺序敏感，乱序在长上下文（数千 key）下累积 36 层可放大到
    # logits 0.5+ 量级的舍入误差（2026-10-19 锚点验证定位：v3 预算拉满 diff
    # 0.55 = 重排精度，order 变体 diff 0.000000）。排序仅稳定数值路径，不改变
    # 选中 token 集合（数学等价）。
    sel_idx = torch.sort(cand_idx[sel_local])[0]

    k_sel = repeat_kv(key_states[..., sel_idx, :], groups)   # [1, n_qh, kt, d]
    v_sel = repeat_kv(value_states[..., sel_idx, :], groups)
    attn_weights = torch.matmul(query_states, k_sel.transpose(-2, -1)) * self.scaling
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, v_sel)          # [1, n_qh, 1, d]
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


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
    n_patched = 0
    layer_id = 0

    def _patch_recursive(container):
        nonlocal n_patched, layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                if layer_id >= len(channels):
                    raise ValueError(
                        "channel file layer count < model layers "
                        f"({len(channels)} < {layer_id})")
                module.layer_id = layer_id
                layer_id += 1
                module.asynctls_channels = channels[module.layer_id]
                module.asynctls_token_budget = token_budget
                module.asynctls_block_size = block_size
                module.asynctls_top_kb = top_kb
                module.flash_forward = module.forward
                module.forward = types.MethodType(_qwen3_asynctls_forward, module)
                n_patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    _patch_recursive(model)
    if n_patched != len(channels):
        raise ValueError(
            f"patched layers {n_patched} != channel layers {len(channels)}")
    return n_patched
