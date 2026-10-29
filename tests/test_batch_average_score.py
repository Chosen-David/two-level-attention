import pytest
import torch
from sparse_attn.ops.batch_coarse_score import score_average_batched


def test_average_metadata_guards():
    k = torch.empty(2, 17, 2, 32, dtype=torch.bfloat16)
    q = torch.empty(2, 8, 32)
    with pytest.raises(ValueError, match='CUDA'):
        score_average_batched(k, q, out_dtype=k.dtype)
    with pytest.raises(ValueError, match='power-of-two'):
        score_average_batched(k, q[:, :6].contiguous(), out_dtype=k.dtype, return_mean=True)
    with pytest.raises(ValueError, match='shape/dtype/stride'):
        score_average_batched(k, q.double(), out_dtype=k.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('dim,group,mean', [(32, 3, False), (128, 4, True)])
def test_average_primary_kernel_exact_graph(batch, dtype, dim, group, mean):
    from sparse_attn.indexer.tli_indexer import TLIIndexer, _score_fine_kernel
    import triton
    torch.manual_seed(8729)
    tokens, heads = 131, 2
    k = torch.randn(batch, tokens * 2, heads, dim, device='cuda', dtype=dtype)[:, ::2]
    q = (torch.randn(batch, heads * group, dim, device='cuda', dtype=dtype) * .125).float()
    def run():
        return score_average_batched(k, q, out_dtype=dtype, return_mean=mean)
    run(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual = run()
    tile, warps = TLIIndexer._TILE_FINE.get(dim, (64, 4))
    for scale in (0., 1., -2.):
        q.copy_(torch.randn_like(q) * scale); k.normal_()
        graph.replay()
        scores, means = actual if mean else (actual, None)
        for row in range(batch):
            expected = torch.empty(heads * group, tokens, device='cuda', dtype=dtype)
            expected_mean = torch.empty(heads, tokens, device='cuda', dtype=dtype) if mean else expected
            _score_fine_kernel[(triton.cdiv(tokens, tile), heads)](
                k[row], q[row], expected, expected_mean, tokens,
                k.stride(1), k.stride(2), q.stride(1), expected.stride(0), expected.stride(1),
                expected_mean.stride(0), expected_mean.stride(1),
                Hkv=heads, G=group, D=dim, BT=tile, HAS_MEAN=mean, num_warps=warps)
            assert torch.equal(scores[row], expected)
            if mean: assert torch.equal(means[row], expected_mean)
