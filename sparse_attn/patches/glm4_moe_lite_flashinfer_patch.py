"""GLM-4.7-Flash FullKV FlashInfer paged-decode attention path.

Prefill keeps the HF attention implementation and RoPE semantics. Decode appends
the same post-RoPE K/V to a fixed paged cache and calls FlashInfer directly.
"""
from types import MethodType

import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack

from .qwen3_flashinfer_patch import FlashInferQwen3Runtime


class FlashInferGLM4MoeLiteRuntime(FlashInferQwen3Runtime):
    """Reuse the FullKV paged runtime with GLM's expanded 256-dim K/V heads."""

    def __init__(self, model, batch_size, prefill_length, max_new_tokens,
                 device, page_size=64):
        config = model.config
        head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        if config.v_head_dim != head_dim:
            raise ValueError('GLM FlashInfer FullKV requires QK and V head_dim equality')
        super().__init__(model, batch_size, prefill_length, max_new_tokens,
                         device=device, page_size=page_size, head_dim=head_dim)


def glm4_moe_lite_flashinfer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    runtime = getattr(self, '_flashinfer_runtime', None)
    if runtime is None:
        raise RuntimeError('FlashInfer GLM runtime has not been bound')
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

    if seq_length != 1:
        runtime.append_prefill(self.layer_idx, key_states, value_states)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs)
        return self.o_proj(
            attn_output.reshape(batch_size, seq_length, -1).contiguous()
        ), attn_weights

    attn_output = runtime.decode_attention(
        self.layer_idx, query_states, key_states, value_states)
    return self.o_proj(
        attn_output.reshape(batch_size, seq_length, -1).contiguous()), None


def register_glm4_moe_lite_flashinfer_patch(model):
    patched = 0
    for module in model.modules():
        if isinstance(module, Glm4MoeLiteAttention):
            if not hasattr(module, '_flashinfer_original_forward'):
                module._flashinfer_original_forward = module.forward
            module.forward = MethodType(glm4_moe_lite_flashinfer_forward, module)
            patched += 1
    return patched


def bind_glm4_moe_lite_flashinfer_runtime(model, runtime):
    patched = 0
    for module in model.modules():
        if isinstance(module, Glm4MoeLiteAttention):
            module._flashinfer_runtime = runtime
            patched += 1
    if patched != runtime.num_layers:
        raise RuntimeError(
            f'FlashInfer bound {patched}/{runtime.num_layers} GLM layers')


def clear_glm4_moe_lite_flashinfer_runtime(model):
    for module in model.modules():
        if isinstance(module, Glm4MoeLiteAttention):
            module._flashinfer_runtime = None
