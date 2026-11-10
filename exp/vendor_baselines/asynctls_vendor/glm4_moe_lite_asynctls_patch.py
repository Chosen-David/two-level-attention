# -*- coding: utf-8 -*-
"""AsyncTLS 三级选择语义的 GLM-4.7-Flash（MLA/MQA）适配层。

与 qwen3_asynctls_patch.py / llama3_asynctls_patch.py 同一套语义复现：
  Stage-1 块粗筛（B=64 min/max 符号感知打分，top-kb=128 块）
  Stage-2 token 细筛（校准 channel dc=32，INT4 量化-反量化，KV group 内
             softmax 平均，块内 top-kt）
  Stage-3 原始全维 Q/K/V 在 S_t 上精确注意力

GLM-4.7-Flash 为 MLA：num_attention_heads == num_key_value_heads == 20
（MQA），qk_head_dim == v_head_dim == 256（[NoPE 192 | RoPE 64]），因此
每个 KV group 仅含 1 个 query head，Stage-2 无需 GQA 组内平均。
"""
import types
from typing import Tuple

import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)

from qwen3_asynctls_patch import (
    _asynctls_sparse_attention,
    _cache_keys,
    _load_channels,
    _padded_key,
    _select_blocks,
)


def _glm_project_qkv(self, hidden_states, position_embeddings):
    """复刻 GLM 原生 MLA 投影：返回旋转后的 Q/K 与 V。

    Q: [batch, num_heads, seq, qk_head_dim]
    K: [batch, num_kv_heads, seq, qk_head_dim]
    V: [batch, num_kv_heads, seq, v_head_dim]
    """
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(
        q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
        key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(
        k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    return query_states, key_states, value_states


def _glm_dense_attention(self, hidden_states, position_embeddings,
                         attention_mask, past_key_values, **kwargs):
    """GLM MLA dense prefill：原生投影 + cache 写入 + eager 注意力。"""
    batch_size, seq_length = hidden_states.shape[:-1]
    query_states, key_states, value_states = _glm_project_qkv(
        self, hidden_states, position_embeddings)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx)
    impl = self.config._attn_implementation or "sdpa"
    attention_interface = ALL_ATTENTION_FUNCTIONS.get(
        impl, eager_attention_forward)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _glm_initialize_blocks(self, hidden_states, position_embeddings, cache):
    """dense prefill 后，用 prompt 最后位置初始化第一步 decode 的 M_{t-1}。"""
    if cache is None:
        self._asynctls_prev_blocks = None
        return
    hidden = hidden_states[:, -1:, :]
    cos, sin = position_embeddings
    last_pos_emb = (cos[:, -1:, :], sin[:, -1:, :])
    query_states, _, _ = _glm_project_qkv(self, hidden, last_pos_emb)
    cached_key = _cache_keys(cache, self.layer_idx)
    padded_key = _padded_key(cached_key, self.asynctls_block_size)
    self._asynctls_prev_blocks = _select_blocks(
        query_states, padded_key, self.asynctls_block_size,
        self.asynctls_top_kb).detach()


def _glm_asynctls_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """dense prefill 初始化块集；decode 使用 M_{t-1} 并保存当前 M_t。"""
    q_len = hidden_states.shape[1]
    if q_len > 1:
        output = _glm_dense_attention(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs)
        _glm_initialize_blocks(
            self, hidden_states, position_embeddings, past_key_values)
        return output

    previous_length = (
        int(past_key_values.get_seq_length(self.layer_idx))
        if past_key_values is not None else 0)
    cache_position = kwargs.get("cache_position", None)
    if cache_position is None and past_key_values is not None:
        cache_position = torch.arange(
            previous_length, previous_length + q_len,
            device=hidden_states.device)

    from sparse_attn.baseline_runtime import require_unmasked_decode
    require_unmasked_decode(
        hidden_states, attention_mask, past_key_values, cache_position)

    query_states, key_states, value_states = _glm_project_qkv(
        self, hidden_states, position_embeddings)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx)

    padded_key = _padded_key(key_states, self.asynctls_block_size)
    current_blocks = _select_blocks(
        query_states, padded_key, self.asynctls_block_size,
        self.asynctls_top_kb)
    if previous_length == 0:
        previous_blocks = current_blocks
    else:
        previous_blocks = getattr(self, "_asynctls_prev_blocks", None)
        if previous_blocks is None:
            raise RuntimeError("AsyncTLS decode 缺少 dense prefill 初始化的 M_{t-1}")
    self._asynctls_prev_blocks = current_blocks.detach()

    attn_output, attn_weights = _asynctls_sparse_attention(
        self, query_states, key_states, value_states, previous_blocks)
    batch_size, seq_length = hidden_states.shape[:-1]
    attn_output = attn_output.transpose(1, 2).reshape(
        batch_size, seq_length, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def enable_asynctls(
    model,
    channel_file: str,
    token_budget: int = 1024,
    block_size: int = 64,
    top_kb: int = 128,
) -> int:
    """把 model 内全部 Glm4MoeLiteAttention 绑定 AsyncTLS 三级稀疏 forward。"""
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
        if not isinstance(module, Glm4MoeLiteAttention):
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
                    or max(group_channels) >= module.qk_head_dim):
                raise ValueError(
                    f"AsyncTLS layer {layer_idx} channel 索引非法")
        module.asynctls_channels = layer_channels
        module.asynctls_token_budget = token_budget
        module.asynctls_block_size = block_size
        module.asynctls_top_kb = top_kb
        module._asynctls_prev_blocks = None
        module.flash_forward = module.forward
        module.forward = types.MethodType(_glm_asynctls_forward, module)
        n_patched += 1

    if n_patched != expected_layers:
        raise RuntimeError(
            f"AsyncTLS GLM patch 层数 {n_patched} != config 层数 {expected_layers}")
    model._asynctls_full_prompt_prefill = True
    return n_patched
