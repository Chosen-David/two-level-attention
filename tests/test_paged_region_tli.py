import copy
import pytest
import torch
from sparse_attn.ops.frozen_region_cache import RegionGeometry,RegionCacheLayer,FrozenRegionCache
from sparse_attn.ops.paged_region_cache import PagedRegionLayer,PagedRegionCache
from sparse_attn.ops.frozen_region_tli import region_mask,tli_args
from sparse_attn.ops.paged_region_tli import region_selection
from sparse_attn.ops.paged_region_kernels import paged_attention
from sparse_attn.indexer.tli_indexer import TLIIndexer

@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv('TLI_FINE_GATHER','0');monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON',False)
    n=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(n)

@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
@torch.inference_mode()
def test_exact_selected_mask_and_projection_against_ring(near,far):
    torch.manual_seed(1);g=RegionGeometry(max_new_tokens=80)
    ring=RegionCacheLayer(g);paged=PagedRegionLayer(g);k=torch.randn(1,2,8400,128)
    for cache in (ring,paged):cache.update(k,k);cache.freeze(near,far)
    q=torch.randn(1,8,1,128);idx=TLIIndexer(tli_args());idx.layer_idx=0;idx.skip_far=far
    for step in range(65):
        nk=torch.randn(1,2,1,128)
        for cache in (ring,paged):cache.update(nk,nk)
    keys,_,positions=ring.gather();expected,_=region_mask(idx,q.transpose(1,2),keys.transpose(1,2),positions,ring.logical_length,128**-.5)
    actual,_=region_selection(idx,q,paged,128**-.5)
    assert torch.equal(actual[...,positions],expected)
    assert torch.equal(actual.sum(-1),expected.sum(-1))

@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
@torch.inference_mode()
def test_patched_qwen_logits_ring_vs_paged(near,far):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from sparse_attn.patches import register_patch
    from sparse_attn.info import resolve_treatment_snapshot
    torch.manual_seed(38)
    cfg=Qwen3Config(vocab_size=51,hidden_size=256,intermediate_size=384,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=128,attention_dropout=0.,max_position_embeddings=10000)
    model=Qwen3ForCausalLM(cfg).eval();model.config._attn_implementation='sdpa'
    args=tli_args();register_patch(model,args,resolve_treatment_snapshot(args))
    ring=FrozenRegionCache(1);ring.tli_regions=True;paged=PagedRegionCache(1)
    ids=torch.randint(0,51,(1,4300));a=model(ids,past_key_values=ring,use_cache=True,logits_to_keep=1)
    b=model(ids,past_key_values=paged,use_cache=True,logits_to_keep=1)
    torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
    decisions={0:dict(skip_near=near,skip_far=far)};ring.freeze(decisions);paged.freeze(decisions)
    for step in range(4):
        token=torch.tensor([[step+2]])
        a=model(token,past_key_values=ring,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
        b=model(token,past_key_values=paged,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
        torch.testing.assert_close(a.logits,b.logits,rtol=1e-5,atol=2e-6)


@torch.inference_mode()
def test_both_skip_has_zero_selector_calls_and_attention_reads_only_protected_pages(monkeypatch):
    torch.manual_seed(818);length=4300
    layer=PagedRegionLayer(RegionGeometry(max_new_tokens=8))
    k=torch.randn(1,2,length,128);v=torch.randn(1,2,length,128)
    layer.update(k,v);layer.freeze(True,True)
    q=torch.randn(1,8,1,128);idx=TLIIndexer(tli_args());idx.layer_idx=0
    def forbidden(*args,**kwargs):
        raise AssertionError('both-skip invoked projection, L1/L2, top-k or KV concatenation')
    for name in ('compute_score','compute_mask','_select_l1_blocks','_subspace_indices'):
        monkeypatch.setattr(idx,name,forbidden)
    monkeypatch.setattr(layer,'project_visible',forbidden)
    monkeypatch.setattr(torch,'topk',forbidden)
    mask,receipt=region_selection(idx,q,layer,128**-.5)
    protected=list(range(idx.sink_tokens))+list(range(length-idx.sliding_window_size,length))
    assert receipt['scorer']==receipt['selector']=='protected_only'
    assert receipt['l1_calls']==receipt['l2_calls']==receipt['selection_topk_calls']==0
    assert receipt['index_scratch_bytes']==0
    assert mask.shape==(1,1,2,length)
    assert mask.sum(-1).flatten().tolist()==[len(protected)]*2
    assert mask.any(dim=(0,1,2)).nonzero().flatten().tolist()==protected
    expected=torch.nn.functional.scaled_dot_product_attention(
        q,k[:,:,protected].repeat_interleave(4,1),v[:,:,protected].repeat_interleave(4,1),
        scale=128**-.5).transpose(1,2)
    # 非支持集即使含 NaN 也不能影响输出；禁止拼接以抓住全可见 KV gather 回退。
    for lp,(_,pk,pv) in layer.pages.items():
        positions=torch.arange(layer.geometry.page)+lp*layer.geometry.page
        hidden=(positions>=idx.sink_tokens)&(positions<length-idx.sliding_window_size)&(positions<length)
        pk[:,:,hidden]=float('nan');pv[:,:,hidden]=float('nan')
    monkeypatch.setattr(torch,'cat',forbidden)
    actual=paged_attention(layer,q,mask,128**-.5)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
