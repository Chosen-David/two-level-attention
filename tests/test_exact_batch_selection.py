"""生产 unique ranking 的批量与独立单请求精确对拍。"""
import pytest
import torch
from sparse_attn.indexer.selection.unique_ranking import rank_unique,rank_unique_batched

@pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA Graph required')
@pytest.mark.parametrize('batch',[1,2,4,8])
@pytest.mark.parametrize('kind',['random','ties','zeros'])
@pytest.mark.parametrize('r',[2048,4096])
def test_batch_ranking_mutable_bounds(batch,kind,r):
    torch.manual_seed(32031)
    h,c=2,4096
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
