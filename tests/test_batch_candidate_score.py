import pytest
import torch
from sparse_attn.ops.batch_candidate_score import score_candidates_batched


def test_cpu_rejected_before_launch():
    with pytest.raises(ValueError,match='CUDA'):
        score_candidates_batched(torch.empty(2,17,2,32,dtype=torch.bfloat16),
            torch.empty(2,8,32),torch.empty(2,2,65,dtype=torch.int64),
            torch.empty(2,2,dtype=torch.int64),out_dtype=torch.bfloat16)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('batch',[1,2,4,8])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('dim,group',[(32,3),(128,4)])
def test_exact_score_and_live_count_graph(batch,dtype,dim,group):
    from sparse_attn.indexer.tli_indexer import _score_fine_cand_kernel,TLIIndexer
    import triton
    torch.manual_seed(9021)
    cap,h,width=257,2,130
    # Noncontiguous batch/token strides, contiguous feature dimension.
    storage=torch.randn(batch,cap*2,h,dim,device='cuda',dtype=dtype)
    k=storage[:,::2]
    q=(torch.randn(batch,h*group,dim,device='cuda',dtype=dtype)*.125).float()
    ids=torch.randint(cap,(batch,h,width),device='cuda',dtype=torch.int64)
    counts=torch.full((batch,h),width,device='cuda',dtype=torch.int64)
    def run():return score_candidates_batched(k,q,ids,counts,out_dtype=dtype)
    run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    tile,warps=TLIIndexer._TILE_FINE.get(dim,(64,4))
    for active in (0,17,width,3):
        live=(torch.arange(batch*h,device='cuda').reshape(batch,h)+active).clamp(max=width)
        if active==0:live.zero_()
        counts.copy_(live)
        valid=torch.arange(width,device='cuda')<counts[...,None]
        ids.copy_(torch.where(valid,torch.randint(cap,ids.shape,device='cuda'),-999999))
        graph.replay()
        for row in range(batch):
            safe=ids[row].clamp(min=0,max=cap-1).contiguous()
            expected=torch.empty(h*group,width,device='cuda',dtype=dtype)
            _score_fine_cand_kernel[(triton.cdiv(width,tile),h)](
                k[row],q[row],safe,counts[row],expected,width,k.stride(1),k.stride(2),q.stride(1),
                safe.stride(0),safe.stride(1),expected.stride(0),expected.stride(1),
                Hkv=h,G=group,D=dim,BT=tile,num_warps=warps)
            assert torch.equal(actual[row],expected)
    assert actual.shape==(batch,h*group,width)
