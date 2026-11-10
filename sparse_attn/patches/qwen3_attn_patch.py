import sys

import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from typing import Optional


def _paper_route(layer, route):
    observer = sys.modules.get('exp.paper.receipt')
    if observer is not None:
        observer.record_qwen_route(layer, route)


def qwen3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """Qwen3 attention with the single GraphReadyPagedRegionCache path."""
    from ..cache import GraphReadyPagedRegionCache
    from ..ops.selector_options import validate_selector_options

    if getattr(self, 'layer_idx', 0) == 0:
        capacity = (past_key_values.get_max_cache_shape()
                    if past_key_values is not None else None)
        validate_selector_options(
            capacity if isinstance(capacity, int) and capacity > 0 else None)
    if not isinstance(past_key_values, GraphReadyPagedRegionCache):
        raise ValueError(
            "NFA requires GraphReadyPagedRegionCache; use prefill_frozen_request")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if input_shape[0] == 1 and input_shape[1] == 1:
        # iter89：仅 Qwen3-8B B1 decode 使用 Q/K 单 launch；其他几何在
        # wrapper 内回退 HF。prefill 与 B>1 始终保持原 apply_rotary_pos_emb。
        from ..ops.qwen3_rope_triton import qwen3_decode_rope
        query_states, key_states = qwen3_decode_rope(
            query_states, key_states, cos, sin, owner=self)
    else:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

    if input_shape[1] != 1:
        request_id = past_key_values.active_request_id
        if request_id is None:
            raise ValueError('GraphReady prefill requires an active request id')
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
            sliding_window=self.sliding_window,
            **kwargs,
        )
        _paper_route(self.layer_idx, 'graph-ready-prefill')
        return self.o_proj(
            attn_output.reshape(*input_shape, -1).contiguous()), attn_weights

    if self.layer_idx == 0:
        # 每次 model forward（包括 Graph capture）只刷新一次执行上下文；
        # 后续 KV write、selector 与 gather 均复用同一 stream/hooks。
        from triton import knobs
        workspace = getattr(past_key_values, '_nfa_selector_workspace', None)
        if workspace is None:
            workspace = past_key_values._nfa_selector_workspace = {}
        workspace['_nfa_launch_context'] = (
            torch.cuda.current_stream().cuda_stream,
            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
            knobs.runtime.debug,
        )

    slot_mapping = past_key_values.slot_mapping_for_layer(
        self.layer_idx)[:input_shape[0]]

    # 当前 token 的 post-RoPE Q 必须在本层内完成 L1/L2 选择；禁止提前到
    # 整步 forward 前读取上一帧 Q。生产子空间路径由 selector 的页统计 owner
    # 同步写入当前 K/V；其他几何在 selector 内保留原独立写入。
    from ..indexer.nfa_layer_batch_selector import select_topk_for_layer_batched
    from ..ops.paged_region_kernels_nfa import (
        build_protected_gather, decode_graph_ready, split_gather_buffers)

    operator_mode = getattr(self.indexer.args, 'nfa_operator_mode', 'off')
    if operator_mode == 'off':
        select_topk_for_layer_batched(
            past_key_values, self.layer_idx,
            past_key_values.active_req_ids, query_states[:, :, 0], attention=self,
            decode_write=(key_states, value_states, slot_mapping))
        result = decode_graph_ready(
            self,
            query_states,
            past_key_values,
            gather_indices=past_key_values.gather_indices_for_layer(self.layer_idx),
            visible_lens=past_key_values.visible_lens_for_layer(self.layer_idx),
            query_to_req=past_key_values.query_to_req_buf,
        )
    else:
        # disjoint 两臂先显式写当前 K/V，保护集与 selector 随后都只读同一
        # 已提交 arena；serial 与 overlap 仅执行流不同，归并顺序完全一致。
        from ..ops.paged_region_kernels import (
            sparse_paged_attention_gather_partial,
            sparse_paged_attention_regions_combine,
        )
        past_key_values.write_decode(
            self.layer_idx, key_states, value_states, slot_mapping)
        protected_gather, protected_lens, middle_gather, middle_lens = (
            split_gather_buffers(past_key_values, self.layer_idx, input_shape[0]))
        workspace = past_key_values._nfa_selector_workspace
        q_flat = query_states[:, :, 0]
        q_flat = q_flat if q_flat.is_contiguous() else q_flat.contiguous()

        def _protected_partial():
            build_protected_gather(
                past_key_values, self.layer_idx, self.indexer,
                protected_gather, protected_lens)
            return sparse_paged_attention_gather_partial(
                past_key_values.layers[self.layer_idx], q_flat,
                protected_gather, protected_lens,
                past_key_values.query_to_req_buf,
                scale=self.scaling, workspace=workspace,
                name='protected', nsplit=1)

        if operator_mode == 'overlap':
            primitives_key = 'nfa_overlap_primitives'
            primitives = workspace.get(primitives_key)
            if primitives is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('overlap streams must be warmed before capture')
                primitives = (torch.cuda.Stream(), torch.cuda.Event(),
                              torch.cuda.Event())
                workspace[primitives_key] = primitives
            side_stream, fork_event, join_event = primitives
            main_stream = torch.cuda.current_stream()
            fork_event.record(main_stream)
            side_stream.wait_event(fork_event)
            context = workspace['_nfa_launch_context']
            try:
                workspace['_nfa_launch_context'] = (
                    side_stream.cuda_stream, context[1], context[2], context[3])
                with torch.cuda.stream(side_stream):
                    protected_parts = _protected_partial()
                    join_event.record(side_stream)
            finally:
                workspace['_nfa_launch_context'] = context
            select_topk_for_layer_batched(
                past_key_values, self.layer_idx,
                past_key_values.active_req_ids, q_flat, attention=self,
                middle_only=True,
                selection_output=(middle_gather, middle_lens))
            middle_parts = sparse_paged_attention_gather_partial(
                past_key_values.layers[self.layer_idx], q_flat,
                middle_gather, middle_lens, past_key_values.query_to_req_buf,
                scale=self.scaling, workspace=workspace, name='middle', nsplit=4)
            main_stream.wait_event(join_event)
        elif operator_mode == 'serial':
            protected_parts = _protected_partial()
            select_topk_for_layer_batched(
                past_key_values, self.layer_idx,
                past_key_values.active_req_ids, q_flat, attention=self,
                middle_only=True,
                selection_output=(middle_gather, middle_lens))
            middle_parts = sparse_paged_attention_gather_partial(
                past_key_values.layers[self.layer_idx], q_flat,
                middle_gather, middle_lens, past_key_values.query_to_req_buf,
                scale=self.scaling, workspace=workspace, name='middle', nsplit=4)
        else:
            raise ValueError(f'unknown NFA operator mode: {operator_mode}')
        result = sparse_paged_attention_regions_combine(
            q_flat, protected_parts, middle_parts, workspace=workspace)
    return self.o_proj(result.reshape(*input_shape, -1).contiguous()), None
