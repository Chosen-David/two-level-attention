"""Exact projection writes; interpreter evidence is not CUDA validation."""
import os
from dataclasses import replace
import pytest
import torch
from sparse_attn.ops import far_skip_accel as accel
from sparse_attn.ops.far_skip import FarSkipConfig, FarSkipLayout
from sparse_attn.ops.far_skip_subspace import FarSkipFineCache

@pytest.fixture(params=['torch', 'interpreter', 'cuda'])
def backend(request, monkeypatch):
    mode = request.param
    interpreting = os.environ.get('TRITON_INTERPRET') == '1'
    if mode == 'interpreter' and not interpreting: pytest.skip('requires TRITON_INTERPRET=1')
    if mode == 'cuda' and (not torch.cuda.is_available() or interpreting): pytest.skip('requires CUDA')
    monkeypatch.setattr(accel, 'enabled', lambda _: mode != 'torch')
    return mode, 'cuda' if mode == 'cuda' else 'cpu'

def setup(device, dtype, heads=3, tail=23):
    cfg = FarSkipConfig(4, 12, 2, 10, 3, 4, quantize=False)
    layout = FarSkipLayout(3, tail)
    k = torch.randn(1, heads, layout.capacity, 32, device=device, dtype=dtype).transpose(1, 2)[..., ::2]
    dims = torch.tensor([15, 99, 1, 99, 7, 99, 0, 99, 3, 99], device=device)[::2]
    fine = FarSkipFineCache(cfg, layout, dims)
    fine.initialize(k, 1)
    fine.keys = torch.zeros(heads, layout.capacity, 10, device=device, dtype=dtype)[..., ::2]
    return k, fine

def reference(k, expected, dims, layout, position):
    slot = layout.slot(position)
    expected[:, slot].copy_(k[0, slot].index_select(-1, dims))

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tensor_position', [False, True])
@torch.inference_mode()
def test_exact_wrap_rewrite_and_strides(backend, dtype, tensor_position, monkeypatch):
    mode, device = backend
    if mode == 'interpreter' and dtype == torch.bfloat16: pytest.skip('BF16 requires CUDA validation')
    torch.manual_seed(71)
    k, fine = setup(device, dtype)
    expected = fine.keys.clone()
    ptr = fine.keys.data_ptr()
    pos = torch.zeros(1, device=device, dtype=torch.long)
    calls = []
    original = accel.project_token
    def tracked(*args):
        calls.append(args[3]); return original(*args)
    monkeypatch.setattr(accel, 'project_token', tracked)
    positions = list(range(1, 80))
    if tensor_position: positions += [79, 79, 2**33, 2**33]
    for position in positions:
        slot = fine.layout.slot(position)
        k[0, slot].normal_()
        reference(k, expected, fine.dims, fine.layout, position)
        if tensor_position:
            pos.fill_(position); fine.update_tensor(k, pos)
        else: fine.append(k, position+1)
        torch.testing.assert_close(fine.keys, expected, rtol=0, atol=0)
    assert fine.keys.data_ptr() == ptr
    assert len(calls) == (0 if mode == 'torch' else len(positions))
    _, second = setup(device, dtype, tail=31)
    assert second.keys.data_ptr() != ptr

@torch.inference_mode()
def test_quantized_path_does_not_use_unquantized_kernel(monkeypatch):
    k, fine = setup('cpu', torch.float32)
    fine.cfg = replace(fine.cfg, quantize=True)
    monkeypatch.setattr(accel, 'enabled', lambda _: True)
    def forbidden(*args): raise AssertionError('quantized values must use the original QDQ')
    monkeypatch.setattr(accel, 'project_token', forbidden)
    fine.append(k, 2)
    fine.update_tensor(k, torch.tensor([1]))
    torch.testing.assert_close(fine.keys[:, 1], fine._encode(k[0, 1].index_select(-1, fine.dims)), rtol=0, atol=0)

@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET') == '1', reason='requires CUDA capture')
@torch.inference_mode()
def test_cuda_multiple_rows_replay_and_new_capture(monkeypatch):
    monkeypatch.setenv('TLI_FAR_SKIP_KERNELS', 'triton')
    for request in range(2):
        rows = [setup('cuda', torch.bfloat16, heads=8, tail=17+b+request) for b in range(8)]
        expected = [f.keys.clone() for _, f in rows]
        pos = torch.ones(8, device='cuda', dtype=torch.long)
        def run():
            for b, (k, fine) in enumerate(rows): fine.update_tensor(k, pos[b:b+1])
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): run(); run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): run()
        for (k, f), ref in zip(rows, expected): reference(k, ref, f.dims, f.layout, 1)
        for step in range(100):
            for b, ((k, fine), ref) in enumerate(zip(rows, expected)):
                absolute = b + min(step, 31+b) + 1
                pos[b].fill_(absolute)
                if step <= 31+b: k[0, fine.layout.slot(absolute)].normal_()
                reference(k, ref, fine.dims, fine.layout, absolute)
            graph.replay()
            for (_, fine), ref in zip(rows, expected): torch.testing.assert_close(fine.keys, ref, rtol=0, atol=0)

@torch.inference_mode()
def test_graph_selector_projection_integration(backend, monkeypatch):
    from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
    mode, device = backend
    if mode == 'torch': pytest.skip('covered by reference graph suite')
    torch.manual_seed(109)
    k, fine = setup(device, torch.float32)
    q = torch.randn(1, 1, 6, 16, device=device)
    workspaces = [FarSkipGraphWorkspace(fine.cfg, fine.layout, 3, fine.dims, fine.dims) for _ in range(2)]
    # Both caches are seeded from the same retained values before comparing routes.
    monkeypatch.setattr(accel, 'enabled', lambda _: False)
    for ws in workspaces: ws.initialize(k, 30)
    length = torch.tensor([31], device=device)
    for current in range(31, 75):
        k[0, fine.layout.slot(current-1)].normal_()
        length.fill_(current)
        monkeypatch.setattr(accel, 'enabled', lambda _: False)
        expected = workspaces[0].select(q, k, length, 0.25)
        monkeypatch.setattr(accel, 'enabled', lambda _: True)
        actual = workspaces[1].select(q, k, length, 0.25)
        for x, y in zip(actual, expected): torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(workspaces[1].fine_cache.keys, workspaces[0].fine_cache.keys, rtol=0, atol=0)
