"""Exact integer/data-movement gates; CPU interpreter != CUDA execution."""
import os
import pytest
import torch

INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1'
pytestmark=pytest.mark.skipif(not INTERPRETER and not torch.cuda.is_available(),reason='requires real CUDA or Triton interpreter')


@pytest.mark.parametrize('r',[33,257,4224])
@pytest.mark.parametrize('kind',['none','all','random','alternating'])
def test_fused_compact(r,kind):
    from sparse_attn.indexer.tli_batch_kernels import compact
    device='cpu' if INTERPRETER else 'cuda'
    torch.manual_seed(319)
    ids=torch.randint(-2**40,2**40,(2,2,r),device=device)
    keep=torch.rand(2,2,r,device=device)>.5
    if kind=='none':keep.zero_()
    if kind=='all':keep.fill_(True)
    if kind=='alternating':keep[:]=torch.arange(r,device=device)%2==0
    order=(~keep).argsort(dim=-1,stable=True)
    actual,valid=compact(ids,keep)
    assert torch.equal(actual,ids.gather(-1,order))
    assert torch.equal(valid,keep.gather(-1,order))

@pytest.mark.parametrize('capacity,sink,swa',[(1,0,1),(33,4,16),(259,0,32),(259,4,32),(259,300,32)])
def test_fused_mask(capacity,sink,swa):
    from sparse_attn.indexer.tli_batch_kernels import assemble_mask
    dev='cpu' if INTERPRETER else 'cuda'
    B,H=3,2;R=min(capacity,17)
    lengths=torch.tensor([1,max(1,capacity//2),capacity],device=dev)
    ids=torch.stack([torch.randperm(capacity,device=dev)[:R] for _ in range(B*H)]).view(B,H,R)
    valid=(ids<lengths[:,None,None]) & (torch.rand(B,H,R,device=dev)>.3)
    ids[~valid]=2**40
    actual=assemble_mask(ids,valid,lengths,capacity,sink,swa)
    expected=torch.zeros(B,H,capacity+1,device=dev,dtype=torch.bool)
    expected.scatter_(-1,torch.where(valid,ids,capacity),True)
    ar=torch.arange(capacity,device=dev)
    force=(ar[None]<sink) | (ar[None]>=(lengths-swa).clamp(min=0)[:,None])
    expected=(expected[...,:capacity] | force[:,None]) & (ar[None,None]<lengths[:,None,None])
    assert torch.equal(actual,expected)

@pytest.mark.parametrize('layout',['token','head'])
@pytest.mark.parametrize('mode',['same','coarse','full'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@torch.no_grad()
def test_fused_update_gather(layout,mode,dtype):
    if INTERPRETER and dtype==torch.bfloat16:
        pytest.skip('BF16 interpreter unsupported; retained for CUDA')
    from sparse_attn.indexer.tli_batch_kernels import gather_update
    dev='cpu' if INTERPRETER else 'cuda'
    B,S,H,D,bs=5,97,2,128,64
    k=torch.randn(B,S,H,D,device=dev,dtype=dtype)
    if layout=='head':k=k.transpose(1,2).contiguous().transpose(1,2)
    lengths=torch.tensor([1,32,64,65,97],device=dev)
    for b,l in enumerate(lengths.tolist()):k[b,l:]=float('nan')
    dims=torch.arange(D,device=dev)[::4] # deliberately strided indices
    cd=None if mode=='full' else dims
    fd=None if mode=='coarse' else dims
    page,fine=gather_update(k,lengths,cd,fd,bs)
    rows=torch.arange(B,device=dev);pos=lengths-1
    ids=(pos//bs)[:,None]*bs+torch.arange(bs,device=dev)[None]
    expected=k[rows[:,None],ids.clamp(max=S-1)]
    if cd is not None:expected=expected.index_select(-1,cd)
    expected=expected.masked_fill((ids>=lengths[:,None])[...,None,None],0)
    ef=k[rows,pos]
    if fd is not None:ef=ef.index_select(-1,fd)
    assert torch.equal(page,expected) and torch.equal(fine,ef)


@pytest.mark.parametrize('method',['avg','minmax'])
@torch.no_grad()
def test_fused_pipeline_mask_and_incremental_state(method):
    from test_tli_batch_parallel import make_indexer
    from sparse_attn.indexer.tli_batch import prepare_mask_batched
    from sparse_attn.indexer import tli_batch_kernels as kernels
    from unittest.mock import patch
    dev='cpu' if INTERPRETER else 'cuda'
    torch.manual_seed(317)
    with patch.dict(os.environ,TLI_BATCH_SCORE='torch',TLI_BATCH_FUSIONS='0'):
        reference=make_indexer(tli_far_method=method,tli_near_method=method)
        candidate=make_indexer(tli_far_method=method,tli_near_method=method)
        k=torch.randn(2,2,512,128,device=dev).transpose(1,2)
        lengths=torch.tensor([290,399],device=dev);q=torch.randn(2,1,8,128,device=dev)
        prepare_mask_batched(candidate,q,k,lengths,128**-.5)
        # This explicitly tests real kernel functions via the interpreter on
        # CPU; it is not a successful GPU route or a timing measurement.
        candidate._batch_tli.triton_ops=kernels
        candidate._batch_tli.triton_score=kernels.score
        for _ in range(3):
            actual,_=prepare_mask_batched(candidate,q,k,lengths,128**-.5)
            expected,_=prepare_mask_batched(reference,q,k,lengths,128**-.5)
            assert torch.equal(actual,expected)
            for name in ('kmin','kmax','kavg','kfine'):
                x,y=getattr(candidate._batch_tli,name),getattr(reference._batch_tli,name)
                assert (x is None and y is None) or torch.equal(x,y)
            k[torch.arange(2,device=dev),lengths]=torch.randn(2,2,128,device=dev)
            lengths.add_(1);q.normal_()
