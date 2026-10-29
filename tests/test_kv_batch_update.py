import pytest
import torch
from sparse_attn.ops.kv_row_update import append_kv_batch

@pytest.mark.parametrize('batch',[1,2,4,8])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_ragged_writes(batch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(9)
    # Noncontiguous batch/head/time strides, contiguous feature dimension.
    k=torch.randn(batch,3,34,40,device=device)[:,:,::2,:32]
    v=torch.randn_like(k);nk=torch.randn(batch,3,1,32,device=device);nv=torch.randn_like(nk)
    slots=torch.tensor([(b*7)%17 for b in range(batch)],device=device)
    alive=torch.tensor([b%3!=0 for b in range(batch)],device=device)
    expected_k,expected_v=k.clone(),v.clone()
    for b in range(batch):
        if alive[b]:
            expected_k[b,:,slots[b]:slots[b]+1]=nk[b]
            expected_v[b,:,slots[b]:slots[b]+1]=nv[b]
    append_kv_batch(k,v,nk,nv,slots,alive)
    torch.testing.assert_close(k,expected_k,rtol=0,atol=0)
    torch.testing.assert_close(v,expected_v,rtol=0,atol=0)

def test_invalid_metadata_does_not_mutate():
    k=torch.zeros(2,1,8,16);v=k.clone();nk=torch.ones(2,1,1,16)
    with pytest.raises(ValueError):append_kv_batch(k,v,nk,nk,torch.tensor([1]),torch.tensor([True,False]))
    assert not k.any() and not v.any()
