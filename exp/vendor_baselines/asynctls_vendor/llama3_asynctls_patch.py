"""AsyncTLS 三级选择语义的 Llama 适配；非官方源码实现（C 轨语义复现）。

与 qwen3_asynctls_patch.py 的语义一致：
  Stage-1 块粗筛（B=64 min/max 符号感知打分，top-kb=128 块）
  Stage-2 token 细筛（校准 channel dc=32，INT4 量化-反量化，GQA 组内平均
             softmax，块内 top-kt）
  Stage-3 原始全维 Q/K/V 在 S_t 上精确注意力

Llama 与 Qwen3 的差异：Llama 无 q_norm/k_norm，投影后不额外归一化。
"""
import types
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb, repeat_kv,
)

from qwen3_asynctls_patch import (
    _int4_quant_dequant, _load_channels, _qwen3_block_score,
)


def _llama_asynctls_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    q_len = hidden_states.shape[1]
    if q_len > 1:
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        return self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs)

    if cache_position is None and past_key_values is not None:
        previous_length = int(past_key_values.get_seq_length(self.layer_idx))
        cache_position = torch.arange(
            previous_length, previous_length + q_len,
            device=hidden_states.device)

    from sparse_attn.baseline_runtime import require_unmasked_decode
    require_unmasked_decode(
        hidden_states, attention_mask, past_key_values, cache_position)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    length = key_states.shape[2]
    block_size = self.asynctls_block_size
    pad = (block_size - length % block_size) % block_size
    if pad:
        key_pad = torch.zeros(
            1, key_states.shape[1], pad, key_states.shape[-1],
            device=key_states.device, dtype=key_states.dtype)
        padded_key = torch.cat([key_states, key_pad], dim=2)
    else:
        padded_key = key_states

    block_score = _qwen3_block_score(query_states, padded_key, block_size)
    n_blocks = min(self.asynctls_top_kb, padded_key.shape[2] // block_size)
    top_blocks = torch.topk(block_score, k=n_blocks).indices
    candidate_idx = torch.cat([
        torch.arange(
            int(block * block_size),
            min(int((block + 1) * block_size), length),
            device=key_states.device)
        for block in top_blocks.tolist()
    ])
    candidate_idx = torch.unique(candidate_idx)
    candidate_key = key_states[..., candidate_idx, :]

    channels = torch.tensor(
        self.asynctls_channels, device=query_states.device, dtype=torch.long)
    projected_query = query_states[..., channels]
    projected_key = _int4_quant_dequant(candidate_key[..., channels])
    query_heads, channel_count = projected_query.shape[1], projected_query.shape[-1]
    kv_heads = projected_key.shape[1]
    groups = query_heads // kv_heads
    grouped_query = projected_query[0].reshape(kv_heads, groups, channel_count)
    logits = torch.einsum(
        "hgc,hjc->hgj", grouped_query, projected_key[0]) * self.scaling
    probabilities = F.softmax(logits.float(), dim=-1).to(projected_query.dtype)
    candidate_score = probabilities.mean(dim=1).mean(dim=0)

    n_selected = min(self.asynctls_token_budget, candidate_score.numel())
    if n_selected <= 0:
        raise RuntimeError("AsyncTLS 候选为空，无法执行稀疏 decode")
    selected_local = torch.topk(candidate_score, k=n_selected).indices
    selected_idx = torch.sort(candidate_idx[selected_local])[0]

    selected_key = repeat_kv(key_states[..., selected_idx, :], groups)
    selected_value = repeat_kv(value_states[..., selected_idx, :], groups)
    attn_weights = torch.matmul(
        query_states, selected_key.transpose(-2, -1)) * self.scaling
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, selected_value)
    attn_output = attn_output.transpose(1, 2).reshape(
        *input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def enable_asynctls(
    model,
    channel_file: str,
    token_budget: int = 1024,
    block_size: int = 64,
    top_kb: int = 128,
) -> int:
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("AsyncTLS token_budget must be a positive integer")
    if type(block_size) is not int or block_size <= 0 or block_size % 8:
        raise ValueError("AsyncTLS block_size must be positive multiple of 8")
    if type(top_kb) is not int or top_kb <= 0:
        raise ValueError("AsyncTLS top_kb must be a positive integer")

    channels = _load_channels(channel_file)
    expected_layers = int(model.config.num_hidden_layers)
    if len(channels) != expected_layers:
        raise ValueError(
            f"channel layers {len(channels)} != model layers {expected_layers}")

    n_patched = 0
    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue
        layer_idx = int(module.layer_idx)
        layer_channels = channels[layer_idx]
        if (not layer_channels or len(set(layer_channels)) != len(layer_channels)
                or min(layer_channels) < 0
                or max(layer_channels) >= module.head_dim):
            raise ValueError(f"AsyncTLS layer {layer_idx} channel 索引非法")
        module.asynctls_channels = layer_channels
        module.asynctls_token_budget = token_budget
        module.asynctls_block_size = block_size
        module.asynctls_top_kb = top_kb
        module.flash_forward = module.forward
        module.forward = types.MethodType(_llama_asynctls_forward, module)
        n_patched += 1

    if n_patched != expected_layers:
        raise RuntimeError(
            f"AsyncTLS Llama patch 层数 {n_patched} != config 层数 {expected_layers}")
    return n_patched
