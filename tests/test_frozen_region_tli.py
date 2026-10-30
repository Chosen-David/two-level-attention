import pytest
import torch
from types import SimpleNamespace
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.frozen_region_tli import region_mask, tli_args
from sparse_attn.ops.frozen_region_selector import region_mask_physical
from sparse_attn.ops.eager_decoding import eager_decoding_attn, eager_decoding_attn_physical
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


@pytest.mark.parametrize('length',[127,4231,4301])
@pytest.mark.parametrize('near,far',[(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_physical_l1_l2_indices_match_reference_after_ring_wraps(monkeypatch,length,near,far):
    torch.manual_seed(712)
    layer=RegionCacheLayer(RegionGeometry(max_new_tokens=80))
    x=torch.randn(1,2,length,128);layer.update(x,x)
    layer.freeze(skip_near=near,skip_far=far,physical_arena=True)
    for _ in range(65):
        kv=torch.randn(1,2,1,128);layer.append_decode(kv,kv)
    q=torch.randn(1,1,8,128)
    k,_,positions=layer.gather()
    reference=TLIIndexer(tli_args());reference.layer_idx=0;reference.skip_far=far
    topk_calls=[]
    original=torch.topk
    def capture_topk(input,k,*args,**kwargs):
        output=original(input,k,*args,**kwargs)
        topk_calls.append(output.indices.detach().clone())
        return output
    monkeypatch.setattr(torch,'topk',capture_topk)
    expected,_=region_mask(reference,q,k.transpose(1,2),positions,
                           layer.logical_length,128**-.5)
    monkeypatch.setattr(torch,'topk',original)
    idx=TLIIndexer(tli_args());idx.layer_idx=0;idx.skip_far=far
    intermediates={}
    actual,receipt=region_mask_physical(idx,q,layer,128**-.5,intermediates)
    assert torch.equal(actual,expected), (actual!=expected).sum().item()
    if near and far:
        assert receipt['l1_calls']==receipt['l2_topk_calls']==0
        assert intermediates['l1'] is None
        return
    assert len(topk_calls) in (2,3)  # 空 near 池不调用 topk。
    l1=intermediates['l1']
    assert torch.equal(l1['i_f'],topk_calls[0]+l1['far_pool_lo'])
    if len(topk_calls)==3:
        assert torch.equal(l1['i_n'],topk_calls[1]+l1['near_pool_lo'])
    else:
        assert l1['i_n'].numel()==0
    expected_l2=topk_calls[-1][0,0]+l1['far_lo_blk']*l1['bs']
    # topk 排名次序不被 mask scatter 消费；紧凑 fine GEMM 的归约顺序
    # 可改变近邻名次，但最终选中集合必须逐项一致。
    assert torch.equal(intermediates['l2_indices'].sort(-1).values,
                       expected_l2.sort(-1).values)
    candidate_ids=intermediates['candidate_ids']
    candidate_valid=intermediates['candidate_valid']
    visible=layer.visible_mask(candidate_ids.clamp(0,layer.logical_length-1))
    assert not (candidate_valid & ~visible).any()


@pytest.mark.parametrize('near,far',[(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_physical_attention_consumes_only_final_support_set(monkeypatch,near,far):
    """16K 以下也必须直接消费最终物理 ID，不得回退全可见 dense attention。"""
    import sparse_attn.ops.eager_decoding as eager_ops
    torch.manual_seed(991)
    layer=RegionCacheLayer(RegionGeometry(max_new_tokens=80))
    x=torch.randn(1,2,4301,128);y=torch.randn_like(x)
    layer.update(x,y)
    layer.freeze(skip_near=near,skip_far=far,physical_arena=True)
    for _ in range(65):
        k=torch.randn(1,2,1,128);v=torch.randn_like(k)
        layer.append_decode(k,v)
    q=torch.randn(1,1,8,128)
    visible_k,visible_v,positions=layer.gather()
    reference=TLIIndexer(tli_args());reference.layer_idx=0;reference.skip_far=far
    expected_mask,_=region_mask(reference,q,visible_k.transpose(1,2),positions,
                                layer.logical_length,128**-.5)
    monkeypatch.setenv('TLI_E127_FAST','0')
    cu=torch.tensor([0,len(positions)],dtype=torch.int32)
    expected=eager_decoding_attn(q,visible_k.transpose(1,2),visible_v.transpose(1,2),
                                 expected_mask,1,cu,softmax_scale=128**-.5)
    def forbidden(*args,**kwargs):
        raise AssertionError('physical attention fell back to visible-KV/dense path')
    monkeypatch.setattr(layer,'gather',forbidden)
    monkeypatch.setattr(eager_ops,'_eager_decoding_attn_dense',forbidden)
    idx=TLIIndexer(tli_args());idx.layer_idx=0;idx.skip_far=far
    selection={}
    actual_mask,_=region_mask_physical(idx,q,layer,128**-.5,selection)
    assert torch.equal(actual_mask,expected_mask)
    actual=eager_decoding_attn_physical(
        q,layer.arena_k,layer.arena_v,selection['physical_ids'],
        selection['physical_counts'],softmax_scale=128**-.5)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=1e-5)


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
@pytest.mark.parametrize('physical',[True,False])
@torch.inference_mode()
def test_patched_qwen_segmented_logits_match_logical_reference(monkeypatch,near,far,physical):
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
    monkeypatch.setenv('TLI_FROZEN_EAGER_REFERENCE','0' if physical else '1')
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
        expected_selector = ('TLIIndexer.compute_mask' if not physical else
                             'protected_only' if near and far else
                             'physical_l1_original_free_topk')
        assert cache.layers[0].tli_receipt['selector']==expected_selector


@pytest.mark.parametrize('physical',[False,True])
@torch.inference_mode()
def test_decode_trace_is_opt_in_and_retains_every_step(monkeypatch,physical):
    torch.manual_seed(527)
    layer=RegionCacheLayer(RegionGeometry(max_new_tokens=8))
    x=torch.randn(1,2,4301,128);layer.update(x,x)
    layer.freeze(skip_near=False,skip_far=False,physical_arena=physical)
    cache=SimpleNamespace(
        layers=[layer],decode_trace_enabled=False,
        update=lambda k,v,layer_idx,kwargs: layer.update(k,v,kwargs))
    idx=TLIIndexer(tli_args());idx.layer_idx=0
    module=SimpleNamespace(layer_idx=0,indexer=idx,scaling=128**-.5)
    monkeypatch.setenv('TLI_FROZEN_EAGER_REFERENCE','0' if physical else '1')
    monkeypatch.setattr('sparse_attn.ops.frozen_region_tli.eager_decoding_attn',
                        lambda *args,**kwargs: None)
    monkeypatch.setattr('sparse_attn.ops.frozen_region_tli.eager_decoding_attn_physical',
                        lambda *args,**kwargs: None)
    from sparse_attn.ops.frozen_region_tli import decode

    def one_step(token):
        q=torch.randn(1,8,1,128);kv=torch.randn(1,2,1,128)
        decode(module,q,kv,kv,cache,torch.tensor([layer.logical_length]))

    one_step(0)
    assert not hasattr(layer,'tli_decode_trace')
    cache.decode_trace_enabled=True
    for token in range(1,4):
        one_step(token)
    trace=layer.tli_decode_trace
    assert [entry['decode_step'] for entry in trace]==[1,2,3]
    assert len(trace)==3
    for entry in trace:
        logical=entry['selected_logical_ids']
        counts=torch.as_tensor(entry['selected_per_kv_head'])
        assert torch.equal((logical>=0).sum(-1,dtype=torch.int32),counts)
        for row,count in zip(logical,counts):
            selected_count=int(count)
            assert torch.equal(row[:selected_count].sort().values,row[:selected_count])
            assert (row[selected_count:]==-1).all()
        if physical:
            assert entry['physical_mapping_matches'].item()
            assert entry['selected_physical_ids'].shape==logical.shape
        else:
            assert 'selected_physical_ids' not in entry
    assert torch.equal(torch.as_tensor(trace[-1]['selected_per_kv_head']),
                       torch.as_tensor(layer.tli_receipt['selected_per_kv_head']))
    assert trace[-1]['logical_tokens']==layer.logical_length
