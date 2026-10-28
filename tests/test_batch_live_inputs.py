"""CPU checks of real selector inputs; CUDA replay remains a separate gate."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparse_attn.baseline_runtime import BatchCtx
from sparse_attn.ops.cuda_graph_state import graph_state, disable_graph
from sparse_attn.patches.qwen3_attn_patch import _batch_device_lengths, _tl_decode_batched
from test_far_skip_hybrid_cache import make_model


def test_host_lengths_refresh_without_reallocating():
    ctx = BatchCtx(2, [10, 20], [0, 0], [True, True], 0)
    first = _batch_device_lengths(ctx, torch.device('cpu'))
    ctx.seq_lens[:] = [11, 20]
    second = _batch_device_lengths(ctx, torch.device('cpu'))
    assert second.tolist() == [11, 20]
    assert first.data_ptr() == second.data_ptr()


@pytest.mark.parametrize('graph', [False, True])
@torch.no_grad()
def test_parallel_selector_tracks_live_lengths(graph, monkeypatch):
    monkeypatch.setenv('TLI_BATCH_INDEXER', 'parallel')
    monkeypatch.setenv('TLI_BATCH_SCORE', 'torch')
    monkeypatch.setenv('TLI_BATCH_FUSIONS', '0')
    monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE', '0')
    idx = make_model().model.layers[0].self_attn.indexer
    idx.far_skip_backend = 'legacy'
    idx._skip_ids = None
    idx.skip_far = False
    ctx = BatchCtx(2, [23, 31], [0, 0], [True, True], 0)
    q = torch.randn(2, 2, 1, 128)
    k, v = torch.randn(2, 2, 1, 64, 128)
    module = SimpleNamespace(scaling=128**-.5, layer_idx=0)
    gs = graph_state()
    gs.enabled, gs.bs = graph, 2
    gs.valid_len_buf = torch.tensor(ctx.seq_lens)
    seen = []
    def consume(q, k, v, mask, *args, **kwargs):
        seen.append(mask.clone())
        return torch.zeros_like(q)
    try:
        with patch('sparse_attn.patches.qwen3_attn_patch.eager_decoding_attn', consume):
            _tl_decode_batched(module, idx, q, k, v, ctx)
            if graph:
                # Replay changes device inputs; capture's Python state stays old.
                gs.valid_len_buf.copy_(torch.tensor([24, 32]))
            else:
                ctx.seq_lens[:] = [24, 32]
            _tl_decode_batched(module, idx, q, k, v, ctx)
        assert idx._batch_tli.previous.tolist() == [24, 32]
        assert seen[-1][0, ..., 23].all()
        assert seen[-1][1, ..., 31].all()
        assert not seen[-1][0, ..., 24:].any()
    finally:
        disable_graph()


@pytest.mark.parametrize('setting', ['dyn_dgate', 'static_pair'])
@torch.no_grad()
def test_legacy_batch_rejects_request_local_prefill_state(setting, monkeypatch):
    monkeypatch.setenv('TLI_BATCH_INDEXER', 'legacy')
    idx = make_model().model.layers[0].self_attn.indexer
    idx.far_skip_backend = 'legacy'
    idx._skip_ids = None
    setattr(idx, setting, True)
    idx._dyn_skip_pending = True
    q = torch.randn(1, 2, 1, 128)
    k, v = torch.randn(2, 1, 1, 64, 128)
    ctx = BatchCtx(1, [23], [0], [True], 0)
    with pytest.raises(ValueError, match='request-local'):
        _tl_decode_batched(SimpleNamespace(scaling=128**-.5), idx, q, k, v, ctx)
    assert idx._dyn_skip_pending  # rejection precedes state reset


@torch.no_grad()
def test_graph_rejects_disabled_boundary_bypass():
    idx = make_model().model.layers[0].self_attn.indexer
    idx.skip_far = False
    idx.boundary_bypass = False
    with pytest.raises(NotImplementedError, match='boundary_bypass'):
        idx.compute_mask_graph(torch.zeros(1, 1, 2, 128), {}, 128**-.5, torch.tensor([23]))
