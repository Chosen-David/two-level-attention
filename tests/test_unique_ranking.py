"""Fused unique ranking: implicit zeros, ties, and dynamic quota under Graph."""
import pytest
import torch
from sparse_attn.ops.candidate_ranking import rank_candidates

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('kind',['random','ties','empty','few_positive'])
def test_fused_unique_ranking(dtype,kind):
    torch.manual_seed(912)
    capacity,r=2048,1536
    ids=torch.stack([torch.randperm(capacity)[:r] for _ in range(3)]).cuda()
    p=torch.rand(3,r,device='cuda').to(dtype)
    if kind=='ties':p.fill_(.5)
    if kind=='empty':p.zero_()
    if kind=='few_positive':p[:,8:]=0
    valid=torch.ones_like(ids,dtype=torch.bool);valid[:,-32:]=False
    low=torch.tensor(0,device='cuda');high=torch.tensor(capacity,device='cuda')
    quota=torch.tensor(1024,device='cuda')
    def run():
        return rank_candidates(ids,p,valid,low,high,quota,capacity=capacity,
                               width=1024,sink=4,deduplicated=True)
    run();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=run()
    for q in [0,1,17,1024]:
        quota.fill_(q);g.replay()
        actual=out.indices.cpu();counts=out.counts.cpu()
        for h in range(3):
            dense=[0.]*capacity
            for token,score,keep in zip(ids[h].tolist(),p[h].tolist(),valid[h].tolist()):
                if keep:dense[token]=score
            expected=sorted(i for i in sorted(range(capacity),key=lambda i:(-dense[i],i))[:q] if i>=4)
            assert counts[0,h].item()==len(expected)
            assert actual[0,h,:len(expected)].tolist()==expected


@pytest.mark.parametrize('r',[1024,2048,2049,4096,4097,8448])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_tree_topk_tail_and_merge_groups(r,dtype):
    from sparse_attn.ops.unique_ranking import tree_topk
    torch.manual_seed(913)
    capacity=r+257
    ids=torch.stack([torch.randperm(capacity)[:r] for _ in range(3)]).cuda()
    p=torch.rand(3,r,device='cuda').to(dtype)
    valid=torch.rand(3,r,device='cuda')>.1
    low=torch.tensor(4,device='cuda');high=torch.tensor(capacity-17,device='cuda')
    quota=torch.tensor(1024,device='cuda')
    top=tree_topk(ids,p,valid,low,high,quota,capacity,1024)
    active=valid&(ids>=low)&(ids<high)&(p>0)
    key=(p.view(torch.int16).to(torch.int32)<<16)+(capacity-ids).to(torch.int32)
    expected=torch.where(active,key,0).topk(1024,-1).values
    assert torch.equal(top,expected)
