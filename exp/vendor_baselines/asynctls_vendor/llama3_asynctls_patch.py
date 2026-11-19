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
from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb,
)

from qwen3_asynctls_patch import (
    _begin_asynctls_step, _commit_asynctls_step,
    _asynctls_sparse_attention, _cache_keys, _load_channels,
    _padded_key, _select_blocks,
)


def _llama_initialize_blocks(module, hidden_states, position_embeddings, cache):
    if cache is None:
        module._asynctls_prev_blocks = None
        return
    hidden = hidden_states[:, -1:, :]
    hidden_shape = (hidden.shape[0], 1, -1, module.head_dim)
    query = module.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = module.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, _ = apply_rotary_pos_emb(query, key, cos[:, -1:, :], sin[:, -1:, :])
    cached_key = _cache_keys(cache, module.layer_idx)
    padded_key = _padded_key(cached_key, module.asynctls_block_size)
    blocks = _select_blocks(
        query, padded_key, module.asynctls_block_size, module.asynctls_top_kb).detach()
    _commit_asynctls_step(module, cache, blocks)


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
    previous_length = (
        int(past_key_values.get_seq_length(self.layer_idx))
        if past_key_values is not None else 0)
    previous_blocks, cache_owner = _begin_asynctls_step(
        self, past_key_values, previous_length, decode=(q_len == 1))
    if q_len > 1:
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        output = self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs)
        _llama_initialize_blocks(self, hidden_states, position_embeddings, past_key_values)
        return output

    if cache_position is None and past_key_values is not None:
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

    padded_key = _padded_key(key_states, self.asynctls_block_size)
    current_blocks = _select_blocks(
        query_states, padded_key, self.asynctls_block_size, self.asynctls_top_kb)
    if previous_length == 0:
        previous_blocks = current_blocks
    attn_output, attn_weights = _asynctls_sparse_attention(
        self, query_states, key_states, value_states, previous_blocks)
    attn_output = attn_output.transpose(1, 2).reshape(
        *input_shape, -1).contiguous()
    output = self.o_proj(attn_output)
    _commit_asynctls_step(self, past_key_values, current_blocks, cache_owner)
    return output, attn_weights


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
        expected_groups = int(module.config.num_key_value_heads)
        if len(layer_channels) != expected_groups:
            raise ValueError(
                f"AsyncTLS layer {layer_idx} KV groups "
                f"{len(layer_channels)} != {expected_groups}")
        for group_channels in layer_channels:
            if (not group_channels
                    or len(set(group_channels)) != len(group_channels)
                    or min(group_channels) < 0
                    or max(group_channels) >= module.head_dim):
                raise ValueError(f"AsyncTLS layer {layer_idx} channel 索引非法")
        module.asynctls_channels = layer_channels
        module.asynctls_token_budget = token_budget
        module.asynctls_block_size = block_size
        module.asynctls_top_kb = top_kb
        module._asynctls_prev_blocks = None
        module.flash_forward = module.forward
        module.forward = types.MethodType(_llama_asynctls_forward, module)
        n_patched += 1

    if n_patched != expected_layers:
        raise RuntimeError(
            f"AsyncTLS Llama patch 层数 {n_patched} != config 层数 {expected_layers}")
    model._asynctls_full_prompt_prefill = True
    return n_patched
