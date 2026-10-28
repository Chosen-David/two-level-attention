"""Triton interpreter is a CPU logic check, never GPU evidence."""
import os
import pytest
import torch

INTERPRETER = os.environ.get('TRITON_INTERPRET') == '1'
pytestmark = pytest.mark.skipif(not INTERPRETER and not torch.cuda.is_available(), reason='needs Triton interpreter or real CUDA')


@pytest.mark.parametrize('head_major',[False,True])
@pytest.mark.parametrize('mode', ['avg','minmax','candidate'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('dim', [32,128])
@torch.no_grad()
def test_batch_score(mode,dtype,dim,head_major):
    if INTERPRETER and dtype == torch.bfloat16:
        pytest.skip('BF16 interpreter unsupported; real CUDA test retained')
    from sparse_attn.indexer.tli_batch_kernels import score
    dev='cpu' if INTERPRETER else 'cuda'
    torch.manual_seed(861)
    B,T,H,G,R=3,97,2,4,43
    k=torch.randn(B,T,H,dim,device=dev,dtype=dtype)
    if head_major:
        k=k.transpose(1,2).contiguous().transpose(1,2)
    hi=k+1
    q=torch.randn(B,H,G,dim,device=dev)
    kw={}
    if mode=='candidate':
        ids=torch.randint(0,T,(B,H,R),device=dev)
        keep=torch.rand(B,H,R,device=dev)>.25
        # Invalid slots may be wildly out of range: kernel must not load them.
        ids[~keep]=2**40
        safe=ids.masked_fill(~keep,0)
        x=k[torch.arange(B,device=dev)[:,None,None],safe,torch.arange(H,device=dev)[None,:,None]]
        expected=torch.einsum('bhgd,bhrd->bhgr',q,x.float()).to(dtype)
        expected.masked_fill_(~keep[:,:,None],-torch.inf)
        kw=dict(ids=ids,keep=keep)
    elif mode=='minmax':
        expected=(torch.einsum('bhgd,bthd->bhgt',q.clamp(max=0),k.float())+
                  torch.einsum('bhgd,bthd->bhgt',q.clamp(min=0),hi.float())).to(dtype)
        kw=dict(kmax=hi)
    else:
        expected=torch.einsum('bhgd,bthd->bhgt',q,k.float()).to(dtype)
    actual=score(k,q,dtype,**kw)
    torch.testing.assert_close(actual,expected,atol=.04 if dtype==torch.bfloat16 else .008 if dtype==torch.float16 else 2e-5,rtol=.02 if dtype==torch.bfloat16 else 2e-3 if dtype==torch.float16 else 1e-5)
    assert torch.equal(torch.isneginf(actual),torch.isneginf(expected))
