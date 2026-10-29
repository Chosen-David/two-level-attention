"""Packed ordering preserves native worker/ordinal traversal and stable duplicates."""
import pytest
import torch
import triton
from sparse_attn.ops.sparse_normalization import _prepare_keys,_offsets_and_keys,candidate_probabilities

@pytest.mark.parametrize('capacity',[1,8192,16384,32768,32840,57344,65528,65535])
def test_all_tokens_signed_encoding_order(capacity):
    token=torch.arange(capacity,dtype=torch.int64);end=(capacity//8192)*8192
    worker=torch.where(token<end,(token//8)%1024,(token-end)%1024)
    ordinal=torch.where(token<end,(token//8192)*8+token%8,end//1024+(token-end)//1024)
    original=worker*capacity+ordinal
    packed=worker*64+ordinal-32768
    assert packed.min()>=-32768 and packed.max()<32767
    assert torch.equal(original.argsort(stable=True),packed.to(torch.int16).argsort(stable=True))
    raw=packed+32768
    assert torch.equal((raw//64)*capacity+raw%64,original)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='queued CUDA gate')
@pytest.mark.parametrize('capacity',[32768,32840,65528])
@pytest.mark.parametrize('r',[1,129,8448])
def test_native_packed_keys_offsets_and_unique(capacity,r):
    torch.manual_seed(7030);hk=3
    ids=torch.randint(capacity,(hk,r),device='cuda');keep=torch.rand(hk,r,device='cuda')>.2
    if r>10:ids[:,5:10]=ids[:,:5];keep[:,5:10]=keep[:,:5]
    def run(packed):
        keys=torch.empty_like(ids,dtype=torch.int16 if packed else torch.int32)
        valid=torch.empty_like(keep);long=torch.empty_like(ids);unique=torch.empty_like(keep)
        offsets=torch.empty(hk,1025,device='cuda',dtype=torch.int64)
        _prepare_keys[(hk,triton.cdiv(r,256))](ids,keep,keys,valid,r,capacity,256,PACKED=packed)
        keys,perm=keys.sort(dim=-1,stable=True)
        _offsets_and_keys[(hk,triton.cdiv(max(r,1025),256))](keys,perm,long,offsets,unique,r,capacity,256,PACKED=packed)
        return long,perm,offsets,unique,valid
    for kind in ['random','empty','last','signed_minus_one']:
        if kind=='empty':keep.zero_()
        if kind=='last':ids.fill_(capacity-1);keep.fill_(True)
        if kind=='signed_minus_one':
            # Raw packed 32767 belongs to worker511/ordinal63, valid at65528.
            ids.fill_(65023 if capacity==65528 else 0);keep.fill_(True)
        old=run(False);new=run(True)
        for a,b in zip(old,new):assert torch.equal(a,b),(capacity,r,kind)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='queued CUDA gate')
@pytest.mark.parametrize('capacity',[32768,32840,65528])
def test_native_graph_probability_and_unique(monkeypatch,capacity):
    torch.manual_seed(309);h,g,r=3,4,8448
    ids=torch.randint(capacity,(h,r),device='cuda');keep=torch.ones_like(ids,dtype=torch.bool)
    token=torch.randn(h*g,capacity,device='cuda',dtype=torch.bfloat16)
    scores=token.gather(1,ids.repeat_interleave(g,0))
    def run(flag):
        monkeypatch.setenv('TLI_SPARSE_SORT_INT16',str(flag))
        return candidate_probabilities(scores,ids,keep,capacity,g,return_unique=True)
    run(1);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run(1)
    for kind in ['random','ties','empty','inf','nan','underflow']:
        ids.random_(capacity);ids[:,-1024:]=ids[:,:1024];keep.fill_(True);token.normal_()
        if kind=='ties':token.zero_()
        if kind=='empty':keep.zero_()
        if kind=='inf':token.fill_(torch.inf)
        if kind=='nan':token.fill_(torch.nan)
        if kind=='underflow':token.mul_(100)
        scores.copy_(token.gather(1,ids.repeat_interleave(g,0)))
        graph.replay();ref=run(0)
        for a,b in zip(actual,ref):assert torch.equal(a,b),(capacity,kind)

@pytest.mark.parametrize('value',['','true','2','-1'])
def test_packed_flag_is_explicit(monkeypatch,value):
    from sparse_attn.ops.sparse_normalization import _packed_sort_enabled
    monkeypatch.setenv('TLI_SPARSE_SORT_INT16',value)
    with pytest.raises(ValueError,match='must be 0 or 1'):_packed_sort_enabled(32840)

@pytest.mark.parametrize('capacity',[0,65536,131072,True,32768.0])
def test_packed_capacity_guard(monkeypatch,capacity):
    from sparse_attn.ops.sparse_normalization import _packed_sort_enabled
    monkeypatch.setenv('TLI_SPARSE_SORT_INT16','1')
    with pytest.raises(ValueError,match='integer capacity'):_packed_sort_enabled(capacity)
    monkeypatch.setenv('TLI_SPARSE_SORT_INT16','0')
    assert not _packed_sort_enabled(capacity)
