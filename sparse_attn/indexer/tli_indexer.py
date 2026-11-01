"""Compatibility surface for the renamed TLI (now NFA) indexer.

The repository refactor renamed TLI to NFA.  GraphReadyPagedRegionCache still
calls this module so the Top-K fine-selection step has a stable entry point.
"""
from .nfa_indexer import NFAIndexer
from ..ops.paged_region_kernels_nfa import (
    _GraphReadyLayerView, region_selection_compact)

TLIIndexer = NFAIndexer


def select_graph_ready_ids(cache, layer_idx, request_id, indexer,
                           q_row, scale, length):
    """Return ``(padded_ids, counts)`` without any host ``.item()`` sync.

    ``padded_ids`` is the fixed-width GPU selection row and ``counts`` is the
    int32 valid count.  The caller copies both directly into fixed metadata
    buffers, so the selector never waits for a scalar device-to-host read.
    """
    layer_view = _GraphReadyLayerView(cache, layer_idx, request_id, length)
    selection, _ = region_selection_compact(
        indexer, q_row, layer_view, scale, build_receipt=False)
    ids = selection.indices[0, 0].contiguous()
    counts = selection.counts[0, 0].contiguous()
    return ids, counts
