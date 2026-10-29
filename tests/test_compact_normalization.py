"""Exact normalization and real selector replay against the accepted path."""
import importlib.util
from pathlib import Path
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

@pytest.mark.parametrize('fused_mean', ['0', '1'])
@pytest.mark.parametrize('capacity', [32768, 32769, 33729])
@pytest.mark.parametrize('group', [1, 4, 8])
def test_compact_probability_replay(capacity, group, fused_mean, monkeypatch):
    monkeypatch.setenv("TLI_COMPACT_FUSED_MEAN", fused_mean)
    from sparse_attn.ops.compact_normalization import compact_probabilities
    from sparse_attn.indexer.tli_indexer import _ako_gqa_mean_kernel
    import triton
    torch.manual_seed(619)
    hk, width = 3, 769
    ids = torch.randint(capacity, (hk, width), device='cuda')
    keep = torch.ones_like(ids, dtype=torch.bool)
    token = torch.randn(hk*group, capacity, device='cuda', dtype=torch.bfloat16)
    scores = token.gather(1, ids.repeat_interleave(group, 0))
    def run():
        return compact_probabilities(scores, ids, keep, capacity, group)
    run(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for kind in ('random', 'ties', 'empty', 'inf', 'negative_inf', 'nan', 'underflow'):
        ids.random_(capacity); token.normal_(); keep.fill_(kind != 'empty')
        ids[:, -3:] = ids[:, :3]
        if kind == 'ties': token.zero_()
        if kind == 'inf': token.fill_(float('inf'))
        if kind == 'negative_inf': token.fill_(-torch.inf)
        if kind == 'nan': token[0].fill_(float('nan'))
        if kind == 'underflow': token.mul_(100)
        scores.copy_(token.gather(1, ids.repeat_interleave(group, 0)))
        graph.replay()
        full = torch.full((hk*group, capacity+1), -torch.inf, device='cuda', dtype=scores.dtype)
        safe = torch.where(keep, ids, capacity).repeat_interleave(group, 0)
        full.scatter_(1, safe, scores)
        sm = full[:, :capacity].softmax(-1)
        expected = torch.empty((hk, capacity), device='cuda', dtype=scores.dtype)
        _ako_gqa_mean_kernel[(hk, triton.cdiv(capacity, 1024))](sm, expected, capacity, 0, 0, G=group, BLOCK=1024, num_warps=4)
        assert torch.equal(actual, expected), kind
        assert torch.equal(actual.topk(32).indices, expected.topk(32).indices), kind


def test_real_selector_compact_replay(monkeypatch):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    spec = importlib.util.spec_from_file_location('prod_args', Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.manual_seed(620)
    capacity = 32832
    q = torch.randn(1, 1, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, capacity, 8, 128, device='cuda', dtype=q.dtype)
    vl = torch.tensor([capacity], device='cuda', dtype=torch.int64)
    cu = torch.tensor([0, capacity], device='cuda', dtype=torch.int32)
    idx = TLIIndexer(module._make_prod_args()); idx.layer_idx = 2; idx.clear()
    index = idx.prepare_index(k, cu)
    def run(): return idx.compute_selection_graph(q, index, 128**-.5, vl)
    monkeypatch.setenv('TLI_COMPACT_NORMALIZATION', '1')
    run(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual = run()
    for length in (132, 4097, 32768, capacity, 512, capacity):
        vl.fill_(length); q.normal_(); graph.replay()
        monkeypatch.setenv('TLI_COMPACT_NORMALIZATION', '0')
        expected = run()
        assert torch.equal(actual.indices, expected.indices), length
        assert torch.equal(actual.counts, expected.counts), length


def test_device_guard_before_launch():
    from types import SimpleNamespace
    from sparse_attn.ops.compact_normalization import supported, probabilities
    scores = SimpleNamespace(dtype=torch.bfloat16, device=torch.device('cuda:0'))
    ids = SimpleNamespace(device=torch.device('cuda:1'))
    keep = SimpleNamespace(device=torch.device('cuda:0'))
    assert not supported(scores, ids, keep, 32768, 4)
    with pytest.raises(ValueError, match='same CUDA device'):
        probabilities(None, scores, ids, keep, 32768, 4)
