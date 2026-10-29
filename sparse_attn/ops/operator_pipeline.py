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
                scale=softmax_scale, protected=True, splits=4, backend='simt')
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
            scale=softmax_scale, protected=True, splits=4, backend='simt')

    # 4. selected attention（中段紧凑候选）
    selected_state = partial_attention(
        q, k, v, ids_counts, vl,
        sink=indexer.sink_tokens, swa=indexer.sliding_window_size,
        scale=softmax_scale, protected=False, splits=4, backend='simt')

    # Join only after selected attention has been submitted on the main stream.
    if overlap:
        current.wait_stream(side)
        for tensor in protected_state:
            tensor.record_stream(current)

    # 5. merge
    return merge_partials(protected_state, selected_state,
                          dtype=q.dtype, backend='simt')


def decode_ring(*args, **kwargs):
    raise NotImplementedError(
        "decode_ring is not available when TLI_OPERATOR_MODE != 'overlap'")
