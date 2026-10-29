import pytest
import torch
from sparse_attn.ops.batch_coarse_score import score_coarse_batched


def test_cpu_rejected_before_launch():
    k = torch.empty(2, 17, 2, 32, dtype=torch.bfloat16)
    q = torch.empty(2, 8, 32)
    with pytest.raises(ValueError, match='CUDA'):
        score_coarse_batched(k, k, q, out_dtype=k.dtype)
    with pytest.raises(ValueError, match='power-of-two'):
        score_coarse_batched(k, k, torch.empty(2, 6, 32), out_dtype=k.dtype, return_mean=True)
    with pytest.raises(ValueError, match='shape/dtype/stride'):
        score_coarse_batched(k, k, q.double(), out_dtype=k.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('dim,group,mean', [(32, 3, False), (128, 4, True)])
def test_primary_kernel_exact_and_live_graph(batch, dtype, dim, group, mean, tokens=131):
    from sparse_attn.indexer.tli_indexer import TLIIndexer, _score_coarse_kernel
    import triton
    torch.manual_seed(8726)
    heads = 2
    # Separate, noncontiguous min/max storage strides exercise both batch bases.
    lo = torch.randn(batch, tokens * 2, heads, dim, device='cuda', dtype=dtype)[:, ::2]
    hi = torch.randn(batch, tokens * 3, heads, dim, device='cuda', dtype=dtype)[:, ::3]
    q = (torch.randn(batch, heads * group, dim, device='cuda', dtype=dtype) * .125).float()
    def run():
        return score_coarse_batched(lo, hi, q, out_dtype=dtype, return_mean=mean)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    tile, warps = TLIIndexer._TILE_COARSE.get(dim, (64, 4))
    for scale in (0., 1., -2.):
        q.copy_(torch.randn_like(q) * scale)
        lo.normal_(); hi.normal_()
        graph.replay()
        scores, means = actual if mean else (actual, None)
        for row in range(batch):
            expected = torch.empty(heads * group, tokens, device='cuda', dtype=dtype)
            expected_mean = torch.empty(heads, tokens, device='cuda', dtype=dtype) if mean else expected
            _score_coarse_kernel[(triton.cdiv(tokens, tile), heads)](
                lo[row], hi[row], q[row], expected, expected_mean, tokens,
                lo.stride(1), lo.stride(2), hi.stride(1), hi.stride(2), q.stride(1),
                expected.stride(0), expected.stride(1), expected_mean.stride(0), expected_mean.stride(1),
                Hkv=heads, G=group, D=dim, BT=tile, HAS_MEAN=mean, num_warps=warps)
            assert torch.equal(scores[row], expected)
            if mean:
                assert torch.equal(means[row], expected_mean)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_long_token_address_domain():
    test_primary_kernel_exact_and_live_graph(2, torch.bfloat16, 32, 3, False, tokens=32769)


@pytest.mark.parametrize('bad', ['max_shape', 'max_dtype', 'feature_stride', 'query_stride', 'heads', 'empty_tokens'])
def test_metadata_guard(bad):
    lo = torch.empty(2, 17, 2, 32, dtype=torch.bfloat16)
    hi = torch.empty_like(lo)
    q = torch.empty(2, 8, 32)
    if bad == 'max_shape': hi = hi[:, :-1]
    if bad == 'max_dtype': hi = hi.float()
    if bad == 'feature_stride': lo = torch.empty(2, 17, 2, 64, dtype=lo.dtype)[..., ::2]
    if bad == 'query_stride': q = torch.empty(2, 8, 64)[..., ::2]
    if bad == 'heads': q = torch.empty(2, 7, 32)
    if bad == 'empty_tokens': lo, hi = lo[:, :0], hi[:, :0]
    with pytest.raises(ValueError, match='shape/dtype/stride'):
        score_coarse_batched(lo, hi, q, out_dtype=lo.dtype)
