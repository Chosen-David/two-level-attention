"""Batch primitives vs independent per-request pinned exact primitives."""
import pytest
import torch
from sparse_attn.ops.sparse_normalization import candidate_probabilities,candidate_probabilities_batched
from sparse_attn.ops.unique_ranking import rank_unique,rank_unique_batched

@pytest.mark.parametrize('group',[0,3,1.5])
def test_invalid_group_rejected(group):
    with pytest.raises(ValueError):
        candidate_probabilities_batched(torch.empty(2,4,8),torch.empty(2,2,8,dtype=torch.int64),
                                       torch.ones(2,2,8,dtype=torch.bool),32768,group)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA required')
@pytest.mark.parametrize('batch',[1,2,4,8])
def test_batch_normalization_exact_rows(batch):
    torch.manual_seed(32030)
    h,g,r,c=2,4,2048,32768
    ids=torch.randint(c,(batch,h,r),device='cuda')
    keep=torch.rand(batch,h,r,device='cuda')>.2
    raw=torch.randn(batch,h*g,c,device='cuda',dtype=torch.bfloat16)
    scores=raw.gather(-1,ids.repeat_interleave(g,1)).contiguous()
    p,u=candidate_probabilities_batched(scores,ids,keep,c,g,return_unique=True)
    for b in range(batch):
        expected,unique=candidate_probabilities(scores[b],ids[b],keep[b],c,g,return_unique=True)
        assert torch.equal(p[b],expected) and torch.equal(u[b],unique)
        # Dense native CUDA softmax reference, with duplicate scores identical.
        dense=torch.full((h*g,c+1),-torch.inf,device='cuda',dtype=scores.dtype)
        expanded=ids[b].repeat_interleave(g,0)
        active=keep[b].repeat_interleave(g,0)
        dense.scatter_(1,torch.where(active,expanded,c),torch.where(active,scores[b],-torch.inf))
        mean=dense[:,:c].softmax(-1).reshape(h,g,c).mean(1)
        gathered=mean.gather(1,ids[b]).masked_fill(~keep[b],0)
        assert torch.equal(p[b],gathered)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA Graph required')
@pytest.mark.parametrize('batch',[1,2,4,8])
@pytest.mark.parametrize('kind',['random','ties','zeros'])
def test_batch_ranking_mutable_bounds(batch,kind):
    torch.manual_seed(32031)
    h,r,c=2,2048,4096
    ids=torch.stack([torch.randperm(c,device='cuda')[:r] for _ in range(batch*h)]).reshape(batch,h,r)
    p=torch.rand(batch,h,r,device='cuda',dtype=torch.bfloat16)
    if kind=='ties':p.fill_(.5)
    if kind=='zeros':p.zero_()
    valid=torch.rand(batch,h,r,device='cuda')>.2
    lower=torch.arange(batch,device='cuda',dtype=torch.int64)*4
    upper=torch.full_like(lower,c)-lower
    quota=torch.full_like(lower,1024)
    def run():return rank_unique_batched(ids,p,valid,lower,upper,quota,capacity=c,sink=4)
    run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    for step in range(3):
        quota.copy_(torch.tensor([(0,17,1024)[(b+step)%3] for b in range(batch)],device='cuda'))
        upper.copy_(torch.tensor([c if b%2 else int(lower[b])+3 for b in range(batch)],device='cuda'))
        graph.replay()
        for b in range(batch):
            reference=rank_unique(ids[b],p[b],valid[b],lower[b],upper[b],quota[b],capacity=c,width=1024,sink=4)
            assert torch.equal(actual.indices[b],reference.indices[0])
            assert torch.equal(actual.counts[b],reference.counts[0])
