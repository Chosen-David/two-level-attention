# -*- coding: utf-8 -*-
"""Operator pipeline (Phase 4: Serial Graph).

P0-P3 完成：compact Selection 接口（TokenSelection）、Partial Attention
算子（SIMT backend）、compute_selection_graph ↔ compute_mask_graph 逐位对拍。

Phase 4 实现 decode_indexed：将 compute_selection_graph 的紧凑输出直接喂给
partial_attention（SIMT backend），替换 full-width mask 路径。
  'off'     -> legacy qwen3_attn_patch path（默认）
  'serial'  -> compact selection → partial attention (SIMT) decode
  'overlap' -> fork/join graph with protected-stream overlap
"""
import os
import torch


_PROTECTED_STREAMS = {}


def _protected_stream(device):
    key = str(device)
    if key not in _PROTECTED_STREAMS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up overlap before CUDA graph capture")
        _PROTECTED_STREAMS[key] = torch.cuda.Stream(device=device)
    return _PROTECTED_STREAMS[key]


def operator_mode() -> str:
    return os.environ.get("TLI_OPERATOR_MODE", "off")



def _ordered_selection_enabled():
    value = os.environ.get('TLI_OPERATOR_ORDERED_SELECTION', '1')
    if value not in ('0', '1'):
        raise ValueError('TLI_OPERATOR_ORDERED_SELECTION must be 0 or 1')
    return value == '1'

def validate_operator_inputs(q, k, v):
    """Reject unsupported HF-format decode inputs before any KV mutation."""
    mode = operator_mode()
    if mode == 'off':
        return
    _ordered_selection_enabled()
    backend = os.environ.get('TLI_OPERATOR_BACKEND', 'simt')
    stages = int(os.environ.get('TLI_OPERATOR_SELECTED_STAGES', '1'))
    if mode not in ('serial', 'overlap') or backend not in ('simt', 'tc') or stages not in (1, 2, 3):
        raise ValueError('invalid operator mode/backend/stages')
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]
            or q.shape[2] != 1 or k.shape[2] != 1 or k.shape[1] < 1
            or q.shape[1] % k.shape[1] or q.shape[-1] != k.shape[-1]
            or min(q.shape[-1], v.shape[-1]) < 1):
        raise ValueError('incompatible operator decode dimensions')
    if (q.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or any(t.dtype != q.dtype or t.device != q.device or t.stride(-1) != 1
                   for t in (q, k, v))):
        raise ValueError('operator requires matching floating Q/K/V and unit feature stride')
    if backend == 'tc' and q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('TC backend requires fp16/bf16')
    from .partial_attention import triton
    if triton is None or not q.is_cuda:
        raise RuntimeError('GPU partial attention requires CUDA and Triton; no implicit fallback')


def decode_indexed(indexer, _, q, k, v, cu, vl, softmax_scale):
    """Compact decode, optionally forking protected attention before indexing.

    Args:
        indexer: TLI indexer（expose prepare_index + compute_selection_graph）
        _: reserved（unused，与 placer 占位符对齐）
        q [B,1,Hq,D], k/v [B,S_cap,Hkv,D]
        cu [0, S_cap] int32, vl [B] int64 valid lengths
        softmax_scale: 缩放因子（已含 1/sqrt(d)）
    Returns:
        [B,1,Hq,Dv]
    """
    from ..ops.partial_attention import partial_attention, merge_partials

    # Explicit experimental routing; keep the accepted default unchanged.
    backend = os.environ.get('TLI_OPERATOR_BACKEND', 'simt')
    stages = int(os.environ.get('TLI_OPERATOR_SELECTED_STAGES', '1'))
    if backend not in ('simt', 'tc'):
        raise ValueError('TLI_OPERATOR_BACKEND must be simt or tc')
    if stages not in (1, 2, 3):
        raise ValueError('TLI_OPERATOR_SELECTED_STAGES must be 1, 2 or 3')
    if backend == 'tc' and q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('TC operator decode requires fp16/bf16')

    # Fork before index preparation: protected positions depend only on lengths.
    overlap = operator_mode() == 'overlap'
    if overlap and not q.is_cuda:
        raise RuntimeError('overlap requires CUDA')
    protected_state = None
    if overlap:
        current = torch.cuda.current_stream(q.device)
        side = _protected_stream(q.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            empty_ids = torch.empty((q.shape[0], k.shape[2], 0),
                                    dtype=torch.int32, device=q.device)
            empty_counts = torch.empty((q.shape[0], k.shape[2]),
                                       dtype=torch.int32, device=q.device)
            protected_state = partial_attention(
                q, k, v, (empty_ids, empty_counts), vl,
                sink=indexer.sink_tokens, swa=indexer.sliding_window_size,
                scale=softmax_scale, protected=True, splits=4, backend=backend)
            # Inputs may be freed by the caller before the side stream finishes.
            for tensor in (q, k, v, vl):
                tensor.record_stream(side)

    index_dict = indexer.prepare_index(k, cu)
    selection = indexer.compute_selection_graph(q, index_dict, softmax_scale, vl)
    ids_counts = (selection.indices, selection.counts)
    if not overlap:
        protected_state = partial_attention(
            q, k, v, ids_counts, vl,
            sink=indexer.sink_tokens, swa=indexer.sliding_window_size,
            scale=softmax_scale, protected=True, splits=4, backend=backend)

    # 4. selected attention（中段紧凑候选）
    selected_state = partial_attention(
        q, k, v, ids_counts, vl,
        sink=indexer.sink_tokens, swa=indexer.sliding_window_size,
        scale=softmax_scale, protected=False, splits=4, backend=backend,
        pipeline_stages=stages)

    # Join only after selected attention has been submitted on the main stream.
    if overlap:
        current.wait_stream(side)
        for tensor in protected_state:
            tensor.record_stream(current)

    # 5. merge
    output = merge_partials(protected_state, selected_state,
                            dtype=q.dtype, backend='simt')
    indexer._graph_attention_route = f"linear_{'overlap' if overlap else 'serial'}_{backend}"
    return output


def decode_ring(workspace, q, k, v, lengths, softmax_scale):
    """Consume far-skip logical IDs directly from ring KV, without KV gather."""
    torch._assert_async(((lengths >= 1) & (lengths < torch.iinfo(torch.int32).max)).all(),
                        "ring logical lengths must fit positive int32 IDs")
    return decode_selected(q, k, v, lengths, softmax_scale, cfg=workspace.cfg,
                           layout=workspace.layout, select=workspace.select,
                           ordered_unique=getattr(workspace, 'ordered_unique_selection', False))


def decode_selected(q, k, v, lengths, softmax_scale, *, cfg, layout, select, ordered_unique=False):
    """Shared split attention consumer for graph and eager compact selectors."""
    from .partial_attention import partial_attention, merge_partials
    from .token_selection import middle_selection
    ordered_enabled = _ordered_selection_enabled()
    if q.shape[0] != 1:
        raise ValueError('ring workspace owns a single request')
    backend = os.environ.get('TLI_OPERATOR_BACKEND', 'simt')
    stages = int(os.environ.get('TLI_OPERATOR_SELECTED_STAGES', '1'))
    if backend not in ('simt', 'tc') or stages not in (1, 2, 3):
        raise ValueError('invalid operator backend or selected stages')
    overlap = operator_mode() == 'overlap'
    if overlap and not q.is_cuda:
        raise RuntimeError('overlap requires CUDA')
    torch._assert_async(((lengths >= 1) & (lengths < torch.iinfo(torch.int32).max)).all(),
                        'ring logical lengths must fit positive int32 IDs')
    def protected():
        ids = torch.empty((1, k.shape[2], 0), dtype=torch.int32, device=q.device)
        counts = torch.empty((1, k.shape[2]), dtype=torch.int32, device=q.device)
        return partial_attention(q, k, v, (ids, counts), lengths,
            sink=cfg.sink_tokens, swa=cfg.window_tokens, scale=softmax_scale,
            protected=True, splits=4, backend=backend, layout=layout,
            round_scaled_query=False)
    if overlap:
        current = torch.cuda.current_stream(q.device)
        side = _protected_stream(q.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            pstate = protected()
            for tensor in (q, k, v, lengths):
                tensor.record_stream(side)
    positions, valid = select(q, k, lengths, softmax_scale)
    # Ring positions are absolute: physical capacity is not an ID bound.
    if ordered_unique and ordered_enabled:
        from .ordered_selection import ordered_middle_selection
        selection = ordered_middle_selection(positions[None], valid[None], lengths,
            cfg.sink_tokens, cfg.window_tokens, torch.iinfo(torch.int32).max)
    else:
        selection = middle_selection(positions[None], valid[None], lengths,
            cfg.sink_tokens, cfg.window_tokens, torch.iinfo(torch.int32).max)
    if not overlap:
        pstate = protected()
    sstate = partial_attention(q, k, v, selection, lengths,
        sink=cfg.sink_tokens, swa=cfg.window_tokens, scale=softmax_scale,
        protected=False, splits=4, backend=backend, layout=layout,
        pipeline_stages=stages, round_scaled_query=False)
    if overlap:
        current.wait_stream(side)
        for tensor in pstate:
            tensor.record_stream(current)
    return merge_partials(pstate, sstate, q.dtype)
