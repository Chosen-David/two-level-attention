import pytest
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.frozen_region_tli import region_mask, tli_args
from sparse_attn.ops.frozen_region_cache import RegionCacheLayer, RegionGeometry

@pytest.fixture(autouse=True)
def reference_env(monkeypatch):
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON',False)
    monkeypatch.setenv('TLI_FINE_GATHER','0')
    monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

@pytest.mark.parametrize('length',[127,4231,8193,12000])
@torch.inference_mode()
def test_gate_off_matches_existing_tli_selection(length):
    torch.manual_seed(193)
    k=torch.randn(1,length,2,128);q=torch.randn(1,1,8,128)
    idx=TLIIndexer(tli_args());idx.layer_idx=0
    cu=torch.tensor([0,length],dtype=torch.int32)
    expected,_=idx.prepare_mask(q,torch.tensor([length-1]),k,cu)
    idx.clear()
    actual,receipt=region_mask(idx,q,k,list(range(length)),length,128**-.5)
    assert receipt['dimensions']==list(range(48,64))+list(range(112,128))
    assert torch.equal(actual,expected), (actual!=expected).sum().item()

@pytest.mark.parametrize('near,far',[(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_skip_never_reads_staging_and_aged_far_is_visible(near,far):
    torch.manual_seed(94)
    length=8400;g=RegionGeometry(max_new_tokens=80)
    layer=RegionCacheLayer(g)
    x=torch.randn(1,2,length,128);layer.update(x,x)
    layer.freeze(skip_near=near,skip_far=far)
    idx=TLIIndexer(tli_args());idx.layer_idx=0;idx.skip_far=far
    q=torch.randn(1,1,8,128)
    for step in range(65):
        kv=torch.randn(1,2,1,128);layer.update(kv,kv)
    if near and not far:
        # A poisoned staging node must not alter the selected live keys.
        layer.nodes['near_staging'].keys.fill_(float('nan'))
    k,_,positions=layer.gather()
    mask,receipt=region_mask(idx,q,k.transpose(1,2),positions,layer.logical_length,128**-.5)
    selected={positions[i] for i in mask.any(dim=(0,1,2)).nonzero().flatten().tolist()}
    s,f,n=g.bounds(layer.logical_length)
    if near: assert not any(f<=p<n for p in selected)
    if far: assert not any(s<=p<f for p in selected)
    assert set(range(s))<=selected
    assert set(range(n,layer.logical_length))<=selected
    assert max(receipt['selected_per_kv_head'])<=1024
    if near and not far:
        assert layer.nodes['far'].size>g.bounds(length)[1]-s

@pytest.mark.parametrize('near,far',[(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_patched_qwen_segmented_logits_match_logical_reference(near,far):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache
    from sparse_attn.patches import register_patch
    from sparse_attn.info import resolve_treatment_snapshot
    from sparse_attn.ops.frozen_region_cache import FrozenRegionCache
    import copy
    torch.manual_seed(38)
    config=Qwen3Config(vocab_size=51,hidden_size=256,intermediate_size=384,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        head_dim=128,attention_dropout=0.,max_position_embeddings=10000)
    model=Qwen3ForCausalLM(config).eval();model.config._attn_implementation='sdpa'
    args=tli_args();register_patch(model,args,resolve_treatment_snapshot(args))
    reference=copy.deepcopy(model)
    cache=FrozenRegionCache(1,RegionGeometry(max_new_tokens=8));cache.tli_regions=True
    baseline=DynamicCache(config=config)
    ids=torch.randint(0,51,(1,4300))
    a=model(ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
    b=reference(ids,past_key_values=baseline,use_cache=True,logits_to_keep=1)
    torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
    cache.freeze({0:dict(skip_near=near,skip_far=far)})
    idx=reference.model.layers[0].self_attn.indexer;idx.skip_far=far
    old=idx.compute_mask
    def logical_mask(qids,scores):
        length=scores['valid_length'];s,f,n=cache.layers[0].geometry.bounds(length)
        scores['score_fine']=scores['score_fine'].clone()
        if near:scores['score_fine'][...,f:n]=float('-inf')
        if far:scores['score_fine'][...,s:f]=float('-inf')
        mask=old(qids,scores)
        if near:mask[...,f:n]=False
        if far:mask[...,s:f]=False
        return mask
    idx.compute_mask=logical_mask
    for step in range(4):
        token=torch.tensor([[step+2]])
        a=model(token,past_key_values=cache,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
        b=reference(token,past_key_values=baseline,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
        torch.testing.assert_close(a.logits,b.logits,atol=2e-6,rtol=1e-5)
        assert cache.layers[0].tli_receipt['selector']=='TLIIndexer.compute_mask'
