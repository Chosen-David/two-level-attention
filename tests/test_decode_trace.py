import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location('decode_trace', Path(__file__).parents[1] / 'exp/bench/decode_trace.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_observer_preserves_output_and_restores_on_failure(tmp_path, monkeypatch):
    from sparse_attn.patches import qwen3_attn_patch as patch
    from sparse_attn.ops import cuda_graph_state as state
    expected = torch.ones(1, 1, 2, 4)
    fn = lambda *a, **kw: expected
    monkeypatch.setattr(patch, 'eager_decoding_attn', fn)
    monkeypatch.setattr(state, 'graph_state', lambda: SimpleNamespace(enabled=False))
    original_replay = torch.cuda.CUDAGraph.replay
    q = torch.zeros_like(expected)
    mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    with pytest.raises(RuntimeError, match='injected'):
        with diag.record_decode_trace(tmp_path / 'trace'):
            assert patch.eager_decoding_attn(q, q, q, mask) is expected
            raise RuntimeError('injected')
    assert patch.eager_decoding_attn is fn
    assert torch.cuda.CUDAGraph.replay is original_replay
    record = torch.load(tmp_path / 'trace/eager_000000.pt', weights_only=True)[0]
    assert torch.equal(record['output'], expected)
    assert torch.equal(record['mask'], mask)


def test_capture_records_refresh_after_each_replay(tmp_path, monkeypatch):
    from sparse_attn.patches import qwen3_attn_patch as patch
    from sparse_attn.ops import cuda_graph_state as state
    q = torch.zeros(1, 1, 2, 4)
    mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    monkeypatch.setattr(patch, 'eager_decoding_attn', lambda *a, **kw: q)
    monkeypatch.setattr(state, 'graph_state', lambda: SimpleNamespace(enabled=True))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    monkeypatch.setattr(torch.cuda.CUDAGraph, 'replay', lambda self: None)
    # CPU mock checks capture lifecycle/file ordering, not actual CUDA refresh.
    with diag.record_decode_trace(tmp_path / 'trace', max_steps=1):
        patch.eager_decoding_attn(q, q, q, mask)
        graph = object()
        torch.cuda.CUDAGraph.replay(graph)
        torch.cuda.CUDAGraph.replay(graph)
    assert len(list((tmp_path / 'trace').glob('*.pt'))) == 1
    assert len(torch.load(tmp_path / 'trace/graph_000_000.pt', weights_only=True)) == 1
