"""Pinned CUDA reference gates for undispatched candidate-domain prototypes."""
import pytest
import torch
import triton
from sparse_attn.ops.sparse_normalization import candidate_probabilities
from sparse_attn.ops.candidate_ranking import rank_candidates
from sparse_attn.indexer.tli_indexer import _ako_gqa_mean_kernel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


def dense_reference(scores,ids,keep,capacity,group):
    hk,r=ids.shape
    full=torch.full((hk*group,capacity+1),-torch.inf,device='cuda',dtype=scores.dtype)
    full.scatter_(1,torch.where(keep,ids,capacity).repeat_interleave(group,0),scores)
    sm=full[:,:capacity].softmax(-1)
    out=torch.empty((hk,capacity),device='cuda',dtype=scores.dtype)
    _ako_gqa_mean_kernel[(hk,triton.cdiv(capacity,1024))](sm,out,capacity,0,0,G=group,BLOCK=1024,num_warps=4)
    return out


@pytest.mark.parametrize('shared_dedup',[False,True])
@pytest.mark.parametrize('capacity',[9224,16384,16400,24568,24576,32768,32832,32840,131072])
@pytest.mark.parametrize('group',[1,4])
@pytest.mark.parametrize('hk',[3,8])
def test_sparse_native_reduction_and_ranking(capacity,group,hk,shared_dedup):
    torch.manual_seed(886)
    r,width=8448,1024
    ids=torch.randint(capacity,(hk,r),device='cuda');keep=torch.ones_like(ids,dtype=torch.bool)
    token=torch.randn(hk*group,capacity,device='cuda',dtype=torch.bfloat16)
    scores=token.gather(1,ids.repeat_interleave(group,0))
    low=torch.tensor(0,device='cuda');high=torch.tensor(capacity-128,device='cuda');quota=torch.tensor(892,device='cuda')
    def run():
        p,unique=candidate_probabilities(scores,ids,keep,capacity,group,return_unique=True)
        sel=rank_candidates(ids,p,unique if shared_dedup else keep,low,high,quota,
                            capacity=capacity,width=width,sink=4,deduplicated=shared_dedup)
        return p,sel
    run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual,sel=run()
    for kind,q in [('random',892),('ties',892),('zero',1024),('few_positive',17),('underflow',892),('inf',1),('nan',0)]:
        ids.random_(capacity);ids[:,-1024:]=ids[:,:1024];token.normal_();keep.fill_(True)
        if kind=='ties':token.zero_()
        if kind=='zero':keep.zero_()
        if kind=='few_positive':keep[:,64:]=False
        if kind=='underflow':token.mul_(100)
        if kind=='inf':token.fill_(torch.inf)
        if kind=='nan':token.fill_(torch.nan)
        scores.copy_(token.gather(1,ids.repeat_interleave(group,0)));quota.fill_(q)
        graph.replay()
        ref=dense_reference(scores,ids,keep,capacity,group)
        expected_p=ref.gather(-1,ids).masked_fill(~keep,0)
        assert torch.equal(actual,expected_p),(capacity,group,kind,'probabilities')
        ref[:,capacity-128:]=-torch.inf
        top=ref.topk(width,-1).indices[:,:q]
        for h in range(hk):
            expected=top[h][top[h]>=4].sort().values.to(torch.int32)
            assert torch.equal(sel.indices[0,h,:sel.counts[0,h]],expected),(capacity,group,kind,'ranking')
