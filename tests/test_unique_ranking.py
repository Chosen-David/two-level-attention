"""生产 unique ranking：implicit zeros、ties 与动态 quota 的 Graph 对拍。"""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('r',[1024,2048,2049,4096,4097,8448])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_tree_topk_tail_and_merge_groups(r,dtype):
    from sparse_attn.indexer.selection.unique_ranking import tree_topk
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
