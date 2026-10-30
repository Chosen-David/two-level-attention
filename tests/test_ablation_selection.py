import pytest
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.frozen_region_tli import region_mask, tli_args
from sparse_attn.ops.frozen_region_cache import RegionGeometry, RegionCacheLayer

@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON',False)
    monkeypatch.setenv('TLI_FINE_GATHER','0')
    monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    torch.set_num_threads(1)

@pytest.mark.parametrize('sink',[0,4,16,64])
@pytest.mark.parametrize('swa',[0,64,128,256])
def test_protected_grid_unique_budget_and_cache(sink,swa):
    torch.manual_seed(42)
    length=10001
    idx=TLIIndexer(tli_args());idx.layer_idx=0
    idx.ablation_spec=dict(sink=sink,swa=swa,near=4096,k1=128,k2=1024,
        near_pages=32,partition=True,far_score='minmax',near_score='avg',boundary=True,measure_mass=True)
    q=torch.randn(1,1,8,128);k=torch.randn(1,length,2,128)
    mask,receipt=region_mask(idx,q,k,list(range(length)),length,128**-.5)
    assert mask.sum(-1).tolist()==[[[1024,1024]]]
    assert max(receipt['candidate_pages_per_kv_head'])<=128
    assert mask[...,:sink].all()
    if swa: assert mask[...,-swa:].all()
    assert 0<=receipt['selected_full_attention_mass']<=receipt['candidate_full_attention_mass']<=1.00001
    g=RegionGeometry(sink=sink,window=swa,max_new_tokens=70)
    layer=RegionCacheLayer(g)
    x=k.transpose(1,2);layer.update(x,x);layer.freeze()
    for _ in range(65): layer.update(x[...,:1,:],x[...,:1,:])
    assert layer.gather()[2]==list(range(length+65))

@pytest.mark.parametrize('subspace',['full','tail','highfreq','random'])
def test_subspace_real_coordinates(subspace):
    args=tli_args();args.tli_subspace=subspace
    if subspace == 'full': args.tli_l2_subspace='full'
    idx=TLIIndexer(args);idx.layer_idx=0
    idx.ablation_spec=dict(sink=4,swa=128,near=4096,k1=128,k2=1024,
        near_pages=32,partition=True,far_score='minmax',near_score='avg',boundary=True)
    q=torch.randn(1,1,8,128);k=torch.randn(1,9000,2,128)
    mask,receipt=region_mask(idx,q,k,list(range(9000)),9000,128**-.5)
    assert len(receipt['dimensions'])==(128 if subspace=='full' else 32)
    assert mask.sum(-1).max()<=1024

@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
def test_retain_and_reclaim_identical_visible_kv(near,far):
    g=RegionGeometry(near=128,window=16,page=64,max_new_tokens=100)
    keep=RegionCacheLayer(g,reclaim=False);evict=RegionCacheLayer(g,reclaim=True)
    x=torch.randn(1,2,400,128)
    for layer in (keep,evict):
        layer.update(x,x);layer.freeze(skip_near=near,skip_far=far)
    for _ in range(90):
        token=torch.randn(1,2,1,128)
        for layer in (keep,evict): layer.update(token,token)
        a,b=keep.gather(),evict.gather()
        assert a[2]==b[2]
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert sum(n.resident_bytes() for n in keep.nodes.values())>=sum(n.resident_bytes() for n in evict.nodes.values())
    if far: assert len(keep.nodes)>len(evict.nodes)
