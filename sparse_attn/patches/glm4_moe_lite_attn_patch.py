"""GLM-4.7-Flash graph-ready frozen-region attention patch.

This file is imported lazily only when a real Glm4MoeLiteAttention module is
present.  It implements dense prefill plus the GraphReadyPagedRegionCache
sparse-gather decode path used by ``prefill_frozen_request``.
"""
import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)

from ..cache import GraphReadyPagedRegionCache
from ..ops.paged_region_kernels_nfa import decode_graph_ready


def glm4_moe_lite_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GLM MLA projection + dense prefill + graph-ready sparse decode."""
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

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
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
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

    if isinstance(past_key_values, GraphReadyPagedRegionCache):
        if seq_length != 1:
            request_id = past_key_values.active_request_id
            if request_id is None:
                raise ValueError('GLM GraphReady prefill requires an active request id')
            past_key_values.write_prefill(
                self.layer_idx, request_id, key_states, value_states)
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(
                self.config._attn_implementation, eager_attention_forward)
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
            return self.o_proj(
                attn_output.reshape(batch_size, seq_length, -1).contiguous()
            ), attn_weights

        slot_mapping = past_key_values.slot_mapping_for_layer(
            self.layer_idx)[:batch_size]
        past_key_values.write_decode(
            self.layer_idx, key_states, value_states, slot_mapping)
        from ..indexer.nfa_layer_batch_selector import select_topk_for_layer_batched
        select_topk_for_layer_batched(
            past_key_values, self.layer_idx,
            past_key_values.active_req_ids, query_states[:, :, 0], attention=self)
        result = decode_graph_ready(
            self,
            query_states,
            past_key_values,
            gather_indices=past_key_values.gather_indices_for_layer(self.layer_idx),
            visible_lens=past_key_values.visible_lens_for_layer(self.layer_idx),
            query_to_req=past_key_values.query_to_req_buf,
        )
        return self.o_proj(
            result.reshape(batch_size, seq_length, -1).contiguous()
        ), None

    raise RuntimeError(
        'GLM NFA requires GraphReadyPagedRegionCache; use prefill_frozen_request')
