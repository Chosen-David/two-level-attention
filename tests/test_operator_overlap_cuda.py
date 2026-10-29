"""Native fork/join graph regression; run on a reserved GPU."""
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops.operator_pipeline import decode_indexed


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_overlap_graph_replay_matches_serial(monkeypatch):
    torch.manual_seed(42)
    q = torch.randn(1, 1, 8, 32, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 1024, 2, 32, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    vl = torch.tensor([1024], device='cuda', dtype=torch.int64)
    cu = torch.tensor([0, 1024], device='cuda', dtype=torch.int32)
    ids = torch.arange(4, 260, device='cuda', dtype=torch.int32)[None, None].expand(1, 2, -1).contiguous()
    counts = torch.full((1, 2), 256, device='cuda', dtype=torch.int32)
    indexer = SimpleNamespace(
        sink_tokens=4, sliding_window_size=64,
        prepare_index=lambda k, cu: k.square().sum(-1),
        compute_selection_graph=lambda *args: SimpleNamespace(indices=ids, counts=counts))
    def run(mode):
        monkeypatch.setenv('TLI_OPERATOR_MODE', mode)
        return decode_indexed(indexer, None, q, k, v, cu, vl, 32 ** -.5)
    run('overlap')
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run('overlap')
    for length, count in [(1024, 256), (512, 37), (128, 0), (1024, 256)]:
        vl.fill_(length)
        counts.fill_(count)
        q.normal_()
        graph.replay()
        expected = run('serial')
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_overlap_allocator_lifetime(monkeypatch, dtype):
    """Drop caller-owned inputs while side-stream work is pending; churn storage."""
    monkeypatch.setenv('TLI_OPERATOR_MODE', 'overlap')
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    results = []
    for seed in range(12):
        with torch.cuda.stream(producer):
            torch.manual_seed(seed)
            q = torch.randn(1, 1, 8, 32, device='cuda', dtype=dtype)
            k = torch.randn(1, 1024, 2, 32, device='cuda', dtype=dtype)
            v = torch.randn_like(k)
            vl = torch.tensor([1024], device='cuda', dtype=torch.int64)
            cu = torch.tensor([0, 1024], device='cuda', dtype=torch.int32)
            ids = torch.arange(4, 260, device='cuda', dtype=torch.int32)[None, None].expand(1, 2, -1).contiguous()
            counts = torch.full((1, 2), 256, device='cuda', dtype=torch.int32)
            indexer = SimpleNamespace(sink_tokens=4, sliding_window_size=64,
                prepare_index=lambda k, cu: k.square().sum(-1),
                compute_selection_graph=lambda *args: SimpleNamespace(indices=ids, counts=counts))
            monkeypatch.setenv('TLI_OPERATOR_MODE', 'serial')
            expected = decode_indexed(indexer, None, q, k, v, cu, vl, 32 ** -.5)
            monkeypatch.setenv('TLI_OPERATOR_MODE', 'overlap')
            actual = decode_indexed(indexer, None, q, k, v, cu, vl, 32 ** -.5)
            results.append((actual, expected))
            del q, k, v, vl, cu, ids, counts, indexer
            # Reuse freed blocks on the producing stream before a host sync.
            for _ in range(8):
                torch.empty((1, 1024, 2, 32), device='cuda', dtype=dtype).fill_(123.)
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        for actual, expected in results:
            actual.record_stream(consumer)
            expected.record_stream(consumer)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    consumer.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_real_indexer_compact_and_overlap_graph(monkeypatch, dtype):
    """Exercise production geometry (scalar quota), selection and native graphs."""
    import importlib.util
    from pathlib import Path
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    path = Path(__file__).resolve().parents[1] / 'exp/bench/test_tli_graph_gates.py'
    spec = importlib.util.spec_from_file_location('operator_prod_args', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(43)
    capacity = 16384
    q = torch.randn(1, 1, 32, 128, device='cuda', dtype=dtype)
    k = torch.randn(1, capacity, 8, 128, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    vl = torch.tensor([capacity], device='cuda', dtype=torch.int64)
    cu = torch.tensor([0, capacity], device='cuda', dtype=torch.int32)
    idx = TLIIndexer(module._make_prod_args())
    idx.layer_idx = 2
    idx.clear()
    index = idx.prepare_index(k, cu)
    for length in (132, 4097, capacity):
        vl.fill_(length)
        sel = idx.compute_selection_graph(q, index, 128 ** -.5, vl)
        mask = idx.compute_mask_graph(q, index, 128 ** -.5, vl)
        positions = torch.arange(capacity, device='cuda')
        protected = (positions < min(length, idx.sink_tokens)) | ((positions >= max(idx.sink_tokens, length-idx.sliding_window_size)) & (positions < length))
        torch.testing.assert_close(sel.to_mask(capacity) | protected, mask[:, 0], atol=0, rtol=0)
    def run(mode):
        monkeypatch.setenv('TLI_OPERATOR_MODE', mode)
        return decode_indexed(idx, None, q, k, v, cu, vl, 128 ** -.5)
    run('overlap')
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run('overlap')
    for length in (132, 4097, capacity, 512):
        vl.fill_(length)
        q.normal_()
        graph.replay()
        expected = run('serial')
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
