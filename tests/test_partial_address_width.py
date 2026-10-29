"""Native probes whose valid KV offsets exceed signed int32, without full scans.

Run only on the reserved idle large-memory GPU. These allocate up to 16 GiB of
KV but initialize/read only selected positions. Expected attention is exactly V
because each head selects one token; no duplicate attention reference is used.
"""
import pytest
import torch
from sparse_attn.ops.partial_attention import partial_attention, merge_partials

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def check_one_token(q,k,v,token,expected,backend):
    b,_,hq,_=q.shape;h=k.shape[2]
    ids=torch.full((b,h,1),token,device=q.device,dtype=torch.int32)
    counts=torch.ones((b,h),device=q.device,dtype=torch.int32)
    lengths=torch.full((b,),k.shape[1],device=q.device,dtype=torch.int64)
    args=dict(sink=0,swa=0,scale=128**-.5,splits=1,backend=backend)
    selected=partial_attention(q,k,v,(ids,counts),lengths,**args)
    neutral=partial_attention(q,k,v,(ids,counts),lengths,protected=True,**args)
    actual=merge_partials(neutral,selected,q.dtype,backend=backend)
    assert torch.equal(actual,expected)
    torch.cuda.synchronize()


@pytest.mark.parametrize('backend',['simt','tc'])
def test_batch_base_above_int32(backend):
    b,cap,h,d,g=64,65528,8,128,4
    k=torch.empty((b,cap,h,d),device='cuda',dtype=torch.bfloat16)
    v=torch.empty_like(k)
    q=torch.zeros((b,1,h*g,d),device='cuda',dtype=k.dtype)
    assert (b-1)*k.stride(0)>2**31
    values=(torch.arange(b,device='cuda')[:,None,None]+torch.arange(h,device='cuda')[None,:,None]/16).expand(b,h,d).to(k.dtype)
    k[:,-1].zero_();v[:,-1].copy_(values)
    expected=values.repeat_interleave(g,dim=1)[:,None]
    check_one_token(q,k,v,cap-1,expected,backend)


@pytest.mark.parametrize('backend',['simt','tc'])
def test_token_stride_product_above_int32(backend):
    stride=2**30;d=128
    kbase=torch.empty(2*stride+d,device='cuda',dtype=torch.bfloat16)
    vbase=torch.empty_like(kbase)
    k=kbase.as_strided((1,3,1,d),(0,stride,d,1))
    v=vbase.as_strided((1,3,1,d),(0,stride,d,1))
    q=torch.zeros((1,1,1,d),device='cuda',dtype=k.dtype)
    assert 2*k.stride(1)==2**31
    k[:,2].zero_();v[:,2].fill_(.625)
    check_one_token(q,k,v,2,torch.full_like(q,.625),backend)
