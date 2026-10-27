"""Linear attention parity; interpreter checks are not GPU measurements."""
import os
import pytest
import torch
from sparse_attn.ops import far_skip_accel as accel
from sparse_attn.ops.hybrid_batch_cache import _attend_linear
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    interpreting = os.environ.get('TRITON_INTERPRET') == '1'
    if request.param == 'cpu' and not interpreting:
        pytest.skip('CPU fused logic requires Triton interpreter')
    if request.param == 'cuda' and (interpreting or not torch.cuda.is_available()):
        pytest.skip('requires CUDA without interpreter')
    return request.param


@pytest.mark.parametrize('length,budget', [(7, 12), (65, 12), (257, 33)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_linear_gather_matches_original_order_padding_and_strides(device, length, budget, dtype, monkeypatch):
    if device == 'cpu' and dtype == torch.bfloat16:
        pytest.skip('BF16 needs device verification')
    torch.manual_seed(273)
    q = torch.randn(1, 1, 6, 16, device=device, dtype=dtype)
    k = torch.randn(1, 2, length, 16, device=device, dtype=dtype).transpose(1, 2)
    v = torch.randn(1, length, 2, 32, device=device, dtype=dtype)[..., ::2]
    mask = torch.zeros(1, 1, 2, length, device=device, dtype=torch.bool)
    workspace = FarSkipWorkspace()
    monkeypatch.setattr(accel, 'enabled', lambda _: True)
    pointers = None
    for count in (0, 1, min(4, length), length):
        mask.zero_(); mask[..., :count] = True
        # Poison excluded slots, requiring the fused kernel to suppress loads.
        k[:, count:] = torch.nan; v[:, count:] = torch.nan
        k[:, :count].normal_(); v[:, :count].normal_()
        expected = _attend_linear(q, k, v, mask, budget, 0.25)
        actual = _attend_linear(q, k, v, mask, budget, 0.25, workspace)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        a, positions = mask[0, 0].int().topk(min(length, budget), sorted=False)
        kk, vv = accel.gather(k, positions, None, workspace, other=v, valid=a.bool())
        current = kk.data_ptr(), vv.data_ptr()
        if pointers is not None:
            assert current == pointers
        pointers = current


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET') == '1',
                    reason='real CUDA Graph required')
@torch.inference_mode()
def test_cuda_linear_capture_reuses_buffers_and_changes_mask(monkeypatch):
    monkeypatch.setenv('TLI_FAR_SKIP_KERNELS', 'triton')
    for length in (65, 129):
        q = torch.randn(1, 1, 4, 128, device='cuda', dtype=torch.bfloat16)
        k, v = [torch.randn(1, length, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        mask = torch.ones(1, 1, 2, length, device='cuda', dtype=torch.bool)
        workspace = FarSkipWorkspace()
        def run():
            return _attend_linear(q, k, v, mask, 33, 128**-.5, workspace)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(); run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for count in (0, 1, 12, length):
            mask.zero_(); mask[..., :count] = True; q.normal_()
            graph.replay()
            expected = _attend_linear(q, k, v, mask, 33, 128**-.5)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
