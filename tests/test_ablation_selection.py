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

@pytest.mark.parametrize('sink',[0,4,16,64])
@pytest.mark.parametrize('swa',[0,64,128,256])
@torch.inference_mode()
def test_all_protection_cells_dense_equivalence_below_budget(sink,swa):
    import copy
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache
    from sparse_attn.patches import register_patch
    from sparse_attn.info import resolve_treatment_snapshot
    from sparse_attn.ops.frozen_region_cache import FrozenRegionCache
    torch.manual_seed(27)
    config=Qwen3Config(vocab_size=51,hidden_size=256,intermediate_size=384,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        head_dim=128,attention_dropout=0.,max_position_embeddings=1024)
    model=Qwen3ForCausalLM(config).eval();model.config._attn_implementation='sdpa'
    dense=copy.deepcopy(model)
    args=tli_args();args.tli_sink_tokens=sink;args.tli_swa_size=swa
    args.tli_ablation_allow_zero_swa=True
    register_patch(model,args,resolve_treatment_snapshot(args))
    idx=model.model.layers[0].self_attn.indexer
    idx.ablation_spec=dict(sink=sink,swa=swa,near=4096,k1=128,k2=1024,
        near_pages=32,partition=True,far_score='minmax',near_score='avg',boundary=True)
    cache=FrozenRegionCache(1,RegionGeometry(sink=sink,window=swa,max_new_tokens=4));cache.tli_regions=True
    dc=DynamicCache(config=config)
    ids=torch.randint(0,51,(1,384))
    a=model(ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
    b=dense(ids,past_key_values=dc,use_cache=True,logits_to_keep=1)
    torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
    cache.freeze({0:dict(skip_near=False,skip_far=False)})
    for step in range(3):
        token=torch.tensor([[step+2]])
        a=model(token,past_key_values=cache,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
        b=dense(token,past_key_values=dc,use_cache=True,logits_to_keep=1)
        torch.testing.assert_close(a.logits,b.logits,rtol=1e-5,atol=2e-6)
        assert cache.layers[0].tli_receipt['selected_per_kv_head']==[385+step]
