"""Owner stats preserve candidate probabilities without traversal sorting."""
import pytest
import torch
from sparse_attn.ops.sparse_normalization import candidate_probabilities

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


@pytest.mark.parametrize('capacity',[16400,32840])
@pytest.mark.parametrize('group',[1,4])
def test_owner_invalid_duplicates_and_no_sort(monkeypatch,capacity,group):
    torch.manual_seed(901)
    ids=torch.randint(capacity,(3,2048),device='cuda')
    ids[:,-256:]=ids[:,:256]
    ids[:,0]=-1;ids[:,1]=capacity
    keep=torch.rand(ids.shape,device='cuda')>.2
    token=torch.randn(3*group,capacity,device='cuda',dtype=torch.bfloat16)
    scores=token.gather(1,ids.clamp(0,capacity-1).repeat_interleave(group,0))
    monkeypatch.setenv('TLI_SPARSE_STATS','sorted')
    expected,expected_unique=candidate_probabilities(scores,ids,keep,capacity,group,return_unique=True)
    monkeypatch.setenv('TLI_SPARSE_STATS','owner')
    def forbidden(*args,**kwargs):
        raise AssertionError('owner stats must not sort candidates')
    monkeypatch.setattr(torch.Tensor,'sort',forbidden)
    actual,unique=candidate_probabilities(scores,ids,keep,capacity,group,return_unique=True)
    assert torch.equal(actual,expected)
    for h in range(3):
        assert set(ids[h,unique[h]].tolist())==set(ids[h,expected_unique[h]].tolist())
        assert unique[h].sum()==expected_unique[h].sum()
