import os

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

from ..indexer import Indexer
from ..ops.eager_decoding import eager_decoding_attn
from ..ops.quality_graph import owned_decode_mask
from ..baseline_runtime import batch_decode_ctx
from .qwen3_attn_patch import _tl_decode_batched, _can_drop_static_mask, _paper_route
from .utils import (
    prepare_seqlens,
    prepare_cu_seqlens,
    prepare_cu_seqlens_from_mask,
    unpad_tensor,
)


def glm4_moe_lite_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GLM-4.7-Flash dense prefill + TLI sparse decode（HF 5.12.1）。"""
    if os.environ.get("TLI_BATCH_INDEXER", "legacy") != "legacy":
        raise RuntimeError("GLM TLI 仅支持 TLI_BATCH_INDEXER=legacy")
    if os.environ.get("TLI_SPARSE_PREFILL", "0") == "1":
        raise RuntimeError("GLM TLI 不支持 sparse prefill，拒绝静默改走其他路由")
    if os.environ.get("TLI_DECODE_DENSE", "0") == "1":
        raise RuntimeError("GLM TLI 不支持 dense decode 消融，拒绝 dense 结果标 TLI")

    batch_size, seq_length = hidden_states.shape[:-1]
    if self.indexer.dyn_dgate:
        ctx = batch_decode_ctx()
        # Batch runners prefill each row with B=1 inside the outer B>1
        # context. Reject before clearing/calibrating/updating any request.
        if batch_size != 1 or (ctx is not None and ctx.batch != 1):
            raise RuntimeError(
                "GLM dynamic far-skip 仅支持单请求；批量请求需要逐行独立 gate 状态")
        if (attention_mask is not None and not (
                seq_length == 1 and owned_decode_mask(past_key_values, attention_mask, ctx))):
            raise RuntimeError(
                "GLM dynamic far-skip 尚不支持显式 attention_mask/padding；"
                "标定必须只覆盖本请求有效 token")
    region_cache = getattr(past_key_values, "tli_regions", False)
    if region_cache:
        from ..ops.cuda_graph_state import graph_state
        ctx = batch_decode_ctx()
        if ((ctx is not None and ctx.batch != 1) or graph_state().enabled
                or os.environ.get("TLI_OPERATOR_MODE", "off") != "off"):
            raise ValueError("GLM frozen regions require eager B1 with operator mode off")
        if batch_size != 1 or attention_mask is not None or self.indexer.dyn_dgate:
            raise ValueError("GLM frozen regions require unpadded B1 and independent region gates")
        layer = past_key_values.layers[self.layer_idx]
        if layer.logical_length and (seq_length != 1 or not layer.frozen):
            raise ValueError("GLM frozen regions require frozen single-token decode")
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

    # Dispatch after native MLA projections/RoPE, before the generic KV write.
    # Region decode owns the single write and absolute-to-physical mapping.
    if region_cache and layer.logical_length:
        from ..ops.frozen_region_tli import decode as region_decode
        result = region_decode(self, query_states, key_states, value_states,
                               past_key_values, cache_position)
        return self.o_proj(result.reshape(batch_size, seq_length, -1).contiguous()), None

    cur_len = 0
    is_static_cache = False
    if past_key_values is not None:
        if (batch_decode_ctx() is None and cache_position is not None
                and int(cache_position[0]) != 0
                and past_key_values.get_seq_length(self.layer_idx) == 0):
            raise RuntimeError(
                "GLM TLI 非零 cache_position 需要已有 KV 历史；"
                "空 cache 不得伪装成新请求 dense prefill")
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx)
        capacity = past_key_values.get_max_cache_shape()
        is_static_cache = capacity is not None and capacity > 0
        if is_static_cache:
            ctx = batch_decode_ctx()
            if ctx is not None and seq_length == 1:
                if not key_states.is_contiguous():
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()
                cur_len = key_states.shape[-2]
            else:
                if (self.layer_idx == 0
                        or getattr(past_key_values, "_tli_seq_len", None) is None):
                    if cache_position is not None:
                        past_key_values._tli_seq_len = int(cache_position[-1]) + 1
                    else:
                        past_key_values._tli_seq_len = past_key_values.get_seq_length(
                            self.layer_idx)
                cur_len = past_key_values._tli_seq_len
                key_states = key_states[:, :, :cur_len, :]
                value_states = value_states[:, :, :cur_len, :]
                if seq_length > 1 and not key_states.is_contiguous():
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()
        else:
            cur_len = past_key_values.get_seq_length(self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward)

    if (past_key_values is None and cache_position is not None
            and int(cache_position[0]) != 0):
        raise RuntimeError(
            "GLM TLI 后续 decode/chunked prefill 需要 KV cache（use_cache=True），"
            "拒绝无 cache dense 回退")

    if past_key_values is None or cur_len == seq_length:
        indexer: Indexer = self.indexer
        indexer.clear()
        if (attention_mask is not None and is_static_cache
                and _can_drop_static_mask(
                    past_key_values, attention_mask, cur_len, prefill=True)):
            attention_mask = None
        indexer.calibrate_dyn_dgate(query_states, key_states)
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
        _paper_route(self.layer_idx, "dense-prefill")
    else:
        self.indexer.freeze_dyn_dgate()
        ctx = batch_decode_ctx()
        if ctx is not None:
            attn_output = _tl_decode_batched(
                self, self.indexer, query_states, key_states, value_states, ctx)
            attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            _paper_route(self.layer_idx, "sparse-decode-batch")
            return attn_output, None
        if past_key_values is None:
            raise RuntimeError("GLM TLI 稀疏 decode 需要 KV cache（use_cache=True）")
        if batch_size != 1 or seq_length != 1:
            raise RuntimeError(
                "GLM TLI 无 BatchCtx 时仅支持 B=1、q_len=1 decode；"
                f"当前 B={batch_size}、q_len={seq_length}，拒绝 chunked prefill/批量误算")
        if (attention_mask is not None and is_static_cache
                and _can_drop_static_mask(
                    past_key_values, attention_mask, cur_len, prefill=False)):
            attention_mask = None
        if attention_mask is not None and (
                attention_mask.dim() != 2 or attention_mask.dtype != torch.bool
                or tuple(attention_mask.shape) != (1, key_states.shape[-2])
                or not bool(attention_mask[0, -1])):
            raise RuntimeError(
                "GLM TLI 单序列 decode 的 attention_mask 须为 [1,S] 2D bool "
                "padding mask，且当前 query 有效；其他 mask 显式拒绝")

        indexer: Indexer = self.indexer
        query_states, key_states, value_states = (
            x.transpose(1, 2) for x in (query_states, key_states, value_states))
        if attention_mask is not None:
            cu_seqlens_k = prepare_cu_seqlens_from_mask(attention_mask)
            key_states, value_states = (
                unpad_tensor(x, attention_mask) for x in (key_states, value_states))
        else:
            cu_seqlens_k = prepare_cu_seqlens(key_states)
            query_states, key_states, value_states = (
                x.view(1, -1, *x.shape[-2:])
                for x in (query_states, key_states, value_states))
        query_position_ids = prepare_seqlens(cu_seqlens_k) - 1
        # 多卡 device_map 下当前 CUDA device 不一定等于本层 device；Triton
        # 与 ATen 必须落在同一设备流，否则候选索引写入可被 gather 提前消费。
        with torch.cuda.device(key_states.device):
            mask, block_size = indexer.prepare_mask(
                query_states, query_position_ids, key_states, cu_seqlens_k,
                softmax_scale=self.scaling)
            attn_output = eager_decoding_attn(
                query_states,
                key_states,
                value_states,
                mask,
                block_size,
                cu_seqlens_k,
                softmax_scale=self.scaling,
                allow_kv_hetero=False,
            )
        _paper_route(self.layer_idx, "sparse-decode")
        attn_weights = None

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
