from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops.regional_score import near_page_scores


@pytest.mark.parametrize('length', [1,3,4,7,15,16,17,63,64,65,128,131])
@pytest.mark.parametrize('near,swa', [(8,0),(8,4),(9,3),(64,17)])
def test_regional_avg_equals_masked_full_scores(length,near,swa):
    block, pages, hkv, groups, d = 4, 33, 2, 3, 8
    torch.manual_seed(length)
    averages = torch.randn(pages,hkv,d)
    q = torch.randn(hkv*groups,d)
    start = torch.tensor(max(0,(length-swa-max(block,near))//block))
    stop = max(int(start),(length+block-1)//block-max(1,swa//block))
    pool = (torch.arange(pages)>=start) & (torch.arange(pages)<stop)
    widths = []
    def score(k,q,dtype):
        widths.append(k.shape[0])
        # Independent pointwise dot, unaffected by GEMM size heuristics.
        return (k[:, :, None]*q.reshape(hkv,groups,d)[None]).sum(-1).permute(1,2,0).reshape(hkv*groups,-1).to(dtype)
    expected = score(averages,q,q.dtype).reshape(hkv,groups,pages).mean(1).masked_fill(~pool,-torch.inf)
    widths.clear()
    actual = near_page_scores(SimpleNamespace(_fast_score_fine=score),averages,q,q.dtype,
                             start,pool,max(block,near),block,hkv,groups)
    torch.testing.assert_close(actual[0,0],expected,rtol=0,atol=0)
    assert widths == [min(pages,(max(block,near)+block-1)//block+2)]
    assert torch.equal(actual[0,0].topk(4).indices,expected.topk(4).indices)


def test_128k_avg_work_is_bounded_to_66_pages():
    seen=[]
    def score(k,q,dtype):
        seen.append(k.shape[0])
        return torch.ones(32,k.shape[0])
    pool=torch.arange(2048)>=1984
    near_page_scores(SimpleNamespace(_fast_score_fine=score),torch.zeros(2048,8,32),
                     torch.zeros(32,32),torch.float32,torch.tensor(1984),pool,4096,64,8,4)
    assert seen==[66]
