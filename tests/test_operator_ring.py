"""Ring direct-ID attention against the workspace's gather/SDPA reference."""
from functools import partial
import pytest
import torch
from sparse_attn.ops.far_skip import FarSkipConfig, FarSkipLayout
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
from sparse_attn.ops.operator_pipeline import decode_ring


@pytest.mark.parametrize('length', [2, 7, 31, 97])
@pytest.mark.parametrize('budget', [7, 12, 40])
def test_ring_cpu_reference(monkeypatch, length, budget):
    import sparse_attn.ops.partial_attention as ops
    monkeypatch.setenv('TLI_OPERATOR_MODE', 'serial')
    real_partial, real_merge = ops.partial_attention, ops.merge_partials
    def reference_partial(*args, **kwargs):
        kwargs['backend'] = 'torch'
        return real_partial(*args, **kwargs)
    monkeypatch.setattr(ops, 'partial_attention', reference_partial)
    monkeypatch.setattr(ops, 'merge_partials', partial(real_merge, backend='torch'))
    torch.manual_seed(881)
    cfg = FarSkipConfig(4, 16, 2, budget, 3, 4, 'avg')
    layout = FarSkipLayout(3, 23)
    q = torch.randn(1, 1, 8, 32)
    k, v = torch.randn(2, 1, layout.capacity, 2, 32)
    vl = torch.tensor([length])
    dims = torch.arange(32)
    ws = FarSkipGraphWorkspace(cfg, layout, 2, dims, dims)
    ws.initialize(k, length)
    reference = FarSkipGraphWorkspace(cfg, layout, 2, dims.clone(), dims.clone())
    rk, rv = k.clone(), v.clone()
    reference.initialize(rk, length)
    positions, valid = reference.select(q, rk, vl, 32**-.5)
    expected = reference.attend(q, rk, rv, positions, valid, 32**-.5)
    actual = decode_ring(ws, q, k, v, vl, 32**-.5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('backend', ['simt', 'tc'])
@pytest.mark.parametrize('mode', ['serial', 'overlap'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_ring_cuda_graph(monkeypatch, backend, mode, dtype):
    monkeypatch.setenv('TLI_OPERATOR_MODE', mode)
    monkeypatch.setenv('TLI_OPERATOR_BACKEND', backend)
    torch.manual_seed(882)
    cfg = FarSkipConfig(4, 16, 2, 12, 3, 4, 'avg')
    layout = FarSkipLayout(3, 23)
    q = torch.randn(1, 1, 8, 32, device='cuda', dtype=dtype)
    k, v = torch.randn(2, 1, layout.capacity, 2, 32, device='cuda', dtype=dtype)
    vl = torch.tensor([31], device='cuda')
    dims = torch.arange(32, device='cuda')
    ws = FarSkipGraphWorkspace(cfg, layout, 2, dims, dims)
    ws.initialize(k, 31)
    reference = FarSkipGraphWorkspace(cfg, layout, 2, dims.clone(), dims.clone())
    rk, rv = k.clone(), v.clone()
    reference.initialize(rk, 31)
    run = lambda: decode_ring(ws, q, k, v, vl, 32**-.5)
    run(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual = run()
    for length in range(31, 98):
        vl.fill_(length); q.normal_()
        # Simulate one append at its mapped slot, preserving all other KV.
        slot = layout.slot(length-1)
        k[:, slot].normal_(); v[:, slot].normal_()
        rk[:, slot].copy_(k[:, slot]); rv[:, slot].copy_(v[:, slot])
        graph.replay()
        positions, valid = reference.select(q, rk, vl, 32**-.5)
        expected = reference.attend(q, rk, rv, positions, valid, 32**-.5)
        torch.testing.assert_close(actual, expected, atol=1e-2 if dtype==torch.bfloat16 else 2e-3, rtol=1e-2)


@pytest.mark.parametrize('length', [0, 2**31-1, 2**31])
def test_ring_rejects_unrepresentable_lengths(monkeypatch, length):
    from types import SimpleNamespace
    monkeypatch.setenv('TLI_OPERATOR_MODE', 'serial')
    q = torch.zeros(1, 1, 1, 32)
    with pytest.raises(RuntimeError, match='positive int32'):
        decode_ring(SimpleNamespace(), q, q, q, torch.tensor([length]), 1.0)
