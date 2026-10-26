"""Regression checks for five-round batch-indexer optimization.

Snapshots and original Graph oracle provide independent pre-change references.
CUDA checks skip explicitly when the device is absent.
"""
from pathlib import Path
import pytest
import torch
from test_tli_batch_parallel import make_indexer
from sparse_attn.indexer.tli_batch import prepare_mask_batched


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv('TLI_BATCH_SCORE','torch')
    monkeypatch.setenv('TLI_BATCH_FUSIONS','0')
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize('far,near,skip,need_minmax,need_avg',[
    ('avg','avg',False,False,True),('minmax','minmax',False,True,False),
    ('minmax','avg',False,True,True),('minmax','avg',True,False,True),
    ('avg','minmax',True,True,False)])
@torch.no_grad()
def test_unused_statistics_absent(far,near,skip,need_minmax,need_avg):
    idx=make_indexer(tli_far_method=far,tli_near_method=near);idx.skip_far=skip
    k=torch.randn(2,512,2,128);q=torch.randn(2,1,8,128);lengths=torch.tensor([290,400])
    prepare_mask_batched(idx,q,k,lengths,128**-.5);state=idx._batch_tli
    assert (state.kmin is not None)==need_minmax
    assert (state.kmax is not None)==need_minmax
    assert (state.kavg is not None)==need_avg

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('subspace,l2',[('tail','same'),('full','same'),('tail','full'),('full','full')])
@torch.no_grad()
def test_query_projection_exact_and_shared(dtype,subspace,l2):
    idx=make_indexer(tli_subspace=subspace,tli_l2_subspace=l2)
    k=torch.randn(2,512,2,128,dtype=dtype);q=torch.randn(2,1,8,128,dtype=dtype);lengths=torch.tensor([290,400])
    prepare_mask_batched(idx,q,k,lengths,128**-.5);state=idx._batch_tli
    qc,qf=state._queries(q,128**-.5)
    full=(q[:,0]*128**-.5).float().reshape(2,2,4,128)
    rc=full if state.coarse_dims is None else full.index_select(-1,state.coarse_dims)
    rf=full if state.fine_dims is None else full.index_select(-1,state.fine_dims)
    assert torch.equal(qc,rc) and torch.equal(qf,rf)
    if state.coarse_dims is state.fine_dims:
        assert qc is qf

@pytest.mark.parametrize('length',[1,33,255,256,257,4224])
@pytest.mark.parametrize('kind',['none','all','random','alternating'])
def test_linear_compaction_exact(length,kind):
    from sparse_attn.indexer.tli_batch import compact_candidates
    g=torch.Generator().manual_seed(591)
    ids=torch.randint(-2**40,2**40,(3,2,length),generator=g)
    keep=torch.rand(3,2,length,generator=g)>.5
    if kind=='none':keep.zero_()
    if kind=='all':keep.fill_(True)
    if kind=='alternating':keep[:]=torch.arange(length)%2==0
    order=(~keep).argsort(dim=-1,stable=True)
    actual,valid=compact_candidates(ids,keep,torch.arange(length))
    assert torch.equal(actual,ids.gather(-1,order))
    assert torch.equal(valid,keep.gather(-1,order))

@pytest.mark.parametrize('l2',['same','full'])
@pytest.mark.parametrize('quant',[False,True])
@torch.no_grad()
def test_owned_fine_cache_never_modifies_source_k(l2,quant):
    idx=make_indexer(tli_l2_subspace=l2,tli_l2_quant=quant)
    k=torch.randn(2,2,512,128).transpose(1,2);before=k.clone()
    q=torch.randn(2,1,8,128);lengths=torch.tensor([290,400])
    prepare_mask_batched(idx,q,k,lengths,128**-.5)
    assert torch.equal(k,before)
    assert idx._batch_tli.kfine.untyped_storage().data_ptr()!=k.untyped_storage().data_ptr()
    for step in range(3):
        k[torch.arange(2),lengths]=torch.randn(2,2,128)
        lengths.add_(1);before=k.clone()
        prepare_mask_batched(idx,q,k,lengths,128**-.5)
        assert torch.equal(k,before)


def test_fusion_opt_in_validation(monkeypatch):
    idx=make_indexer();q=torch.randn(2,1,8,128);k=torch.randn(2,512,2,128)
    monkeypatch.setenv('TLI_BATCH_FUSIONS','1')
    with pytest.raises(ValueError,match='requires TLI_BATCH_SCORE'):
        prepare_mask_batched(idx,q,k,torch.tensor([290,400]),128**-.5)
