import pytest
import torch
from sparse_attn.ops.candidate_ranking import rank_candidates


@pytest.mark.parametrize('kind', ['random', 'plateau', 'zero', 'few_positive'])
@pytest.mark.parametrize('quota', [0, 1, 17, 64])
def test_candidate_ranking_implicit_zeros_and_duplicates(kind, quota):
    torch.manual_seed(885)
    s, hk, width = 257, 3, 64
    ids = torch.randint(s, (hk, 93))
    ids[:, -7:] = ids[:, :7]
    tokenp = torch.rand(hk,s).to(torch.bfloat16)
    if kind == 'plateau': tokenp.fill_(0.5)
    if kind == 'zero': tokenp.zero_()
    if kind == 'few_positive': tokenp[:, 10:] = 0
    p = tokenp.gather(-1, ids)
    valid = torch.rand_like(p.float()) > .1
    lo, hi = 4, 217
    result = rank_candidates(ids,p,valid,torch.tensor(lo),torch.tensor(hi),torch.tensor(quota),capacity=s,width=width,sink=8)
    for h in range(hk):
        dense = [0.0] * s
        for i in range(ids.shape[-1]):
            if valid[h,i]: dense[int(ids[h,i])] = float(p[h,i])
        chosen = sorted(range(lo,hi), key=lambda i:(-dense[i],i))[:quota]
        expected = sorted(i for i in chosen if i>=8)
        assert result.counts[0,h].item()==len(expected)
        assert result.indices[0,h,:len(expected)].tolist()==expected


@pytest.mark.parametrize('capacity', [65535, 65536, 131072])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_ranking_key_width_boundary(capacity, dtype):
    ids = torch.tensor([[capacity-2, capacity-5, capacity-2, capacity-3]])
    p = torch.tensor([[.5, 1., .5, .5]], dtype=dtype)
    result = rank_candidates(ids, p, torch.ones_like(ids, dtype=torch.bool),
        torch.tensor(capacity-7), torch.tensor(capacity), torch.tensor(4),
        capacity=capacity, width=4)
    # Three unique positives then the first implicit zero, canonical ID order.
    assert result.counts.tolist() == [[4]]
    assert result.indices.tolist() == [[[capacity-7, capacity-5, capacity-3, capacity-2]]]
