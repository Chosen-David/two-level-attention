import gc
import weakref
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from sparse_attn.ops.frozen_region_cache import FrozenRegionCache, RegionCacheLayer, RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig, decide_regions, profile_regions


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def expected_positions(length, prompt, g, near, far):
    sink, initial_far, _ = g.bounds(prompt)
    if near:
        far_hi = min(g.sink,length) if far else g.bounds(length)[1]
        return list(range(far_hi)) + list(range(max(g.sink, length-g.window), length))
    current_far = g.bounds(length)[1]
    return list(range(length)) if not far else list(range(min(g.sink, length)))+list(range(current_far, length))


@pytest.mark.parametrize('near,far', [(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_physical_node_deletion_and_multiple_ring_wraps(near, far):
    g = RegionGeometry(sink=2, near=6, window=3, page=4, max_new_tokens=40)
    layer = RegionCacheLayer(g)
    initial = torch.arange(17.).reshape(1,1,17,1)
    k,v = layer.update(initial, initial*2)
    assert k.data_ptr() == initial.data_ptr()
    assert all(n.keys.untyped_storage().data_ptr()!=initial.untyped_storage().data_ptr() for n in layer.nodes.values())
    refs = {name: weakref.ref(layer.nodes[name].keys) for name in ['near','far']}
    before = sum(n.resident_bytes() for n in layer.nodes.values())
    deleted = sum(layer.nodes[name].resident_bytes() for name, drop in [('near',near and far),('far',far)] if drop)
    layer.freeze(skip_near=near,skip_far=far);gc.collect()
    assert ('near' not in layer.nodes)==near and ('far' not in layer.nodes)==far
    if near and far: assert refs['near']() is None
    elif near:
        assert refs['near']() is layer.nodes['near_staging'].keys
        assert not layer.nodes['near_staging'].attention_visible
    if far: assert refs['far']() is None
    assert sum(n.resident_bytes() for n in layer.nodes.values())==before-deleted
    for t in range(17, 49):
        new=torch.tensor([[[[float(t)]]]])
        layer.update(new,new*2,dict(cache_position=torch.tensor([t])))
        k,v,positions=layer.gather()
        expected=expected_positions(t+1,17,g,near,far)
        assert positions==expected
        assert k.flatten().tolist()==[float(p) for p in expected]
        assert v.flatten().tolist()==[2.*p for p in expected]
        assert layer.get_seq_length()==t+1
        assert layer.nodes['swa'].keys.shape[-2]==3
        if near and not far:
            assert layer.permanently_dropped==0
            assert layer.report()['stored_tokens']==t+1
            assert layer.report()['attention_tokens']<t+1


@torch.inference_mode()
def test_near_staging_reappears_in_far_after_aging_without_kv_savings():
    g=RegionGeometry(sink=2,near=6,window=3,page=4,max_new_tokens=40)
    layer=RegionCacheLayer(g);initial=torch.arange(17.).reshape(1,1,17,1)
    layer.update(initial,initial)
    before=sum(n.resident_bytes() for n in layer.nodes.values())
    layer.freeze(skip_near=True,skip_far=False)
    assert sum(n.resident_bytes() for n in layer.nodes.values())==before
    assert layer.report()['linked_order']==['sink','far','near_staging','swa']
    assert layer.report()['attention_order']==['sink','far','swa']
    assert 8 not in layer.gather()[2]  # Token8 starts in near and is not attended.
    for t in range(17,21):
        new=torch.tensor([[[[float(t)]]]]);layer.update(new,new)
    assert 8 in layer.gather()[2]  # It is now far and must become visible again.
    assert layer.nodes['far'].read()[2]==list(range(2,g.bounds(21)[1]))


@torch.inference_mode()
def test_short_prefill_and_lifecycle_guards():
    g=RegionGeometry(sink=4,near=6,window=3,page=4,max_new_tokens=12)
    layer=RegionCacheLayer(g);x=torch.zeros(1,1,1,2)
    layer.update(x,x)
    with pytest.raises(ValueError):layer.update(x,x)
    layer.freeze()
    for _ in range(10):layer.update(x,x)
    assert layer.gather()[2]==list(range(11))
    with pytest.raises(ValueError):layer.freeze(skip_near=True)
    with pytest.raises(ValueError):layer.update(x,x,dict(cache_position=torch.tensor([0])))


def test_independent_gates_and_absent_regions():
    profile={'near':{'mean':.035},'far':{'mean':.04}}
    assert GateConfig().epsilon == .05
    assert decide_regions(profile,GateConfig())==dict(skip_near=True,skip_far=True)
    profile={'near':{'mean':.03},'far':{'mean':.04}}
    assert decide_regions(profile,GateConfig())==dict(skip_near=True,skip_far=True)
    assert decide_regions(profile,GateConfig(),False,False)==dict(skip_near=False,skip_far=False)
    assert decide_regions(profile,GateConfig(mode='full'))==dict(skip_near=False,skip_far=False)


def test_profile_matches_each_query_own_regions():
    torch.manual_seed(7)
    g=RegionGeometry(sink=2,near=6,window=3,page=4)
    q=torch.randn(1,4,20,8);k=torch.randn(1,2,20,8)
    actual=profile_regions(q,k,g,rows=8,chunk=3)
    values={'near':[],'far':[]}
    for pos in range(12,20):
        scores=q[:,:,pos:pos+1]@k[:,:,:pos+1].repeat_interleave(2,1).transpose(-1,-2)/8**.5
        p=scores.softmax(-1);sink,hi,nearhi=g.bounds(pos+1)
        values['near'].append(float(p[...,hi:nearhi].sum(-1).mean()))
        values['far'].append(float(p[...,sink:hi].sum(-1).mean()))
    for region in values:
        assert actual[region]['mean']==pytest.approx(sum(values[region])/8,abs=1e-6)


@pytest.mark.parametrize('near,far', [(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_tiny_qwen_logits_match_explicit_full_cache_gather_reference(near,far):
    torch.manual_seed(11)
    config=Qwen3Config(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,
        attention_dropout=0.,attn_implementation='sdpa')
    model=Qwen3ForCausalLM(config).eval()
    g=RegionGeometry(sink=2,near=6,window=3,page=4,max_new_tokens=40)
    prompt=torch.randint(0,37,(1,17));tokens=torch.randint(0,37,(1,25))
    node_cache=FrozenRegionCache(2,g)
    out=model(prompt,past_key_values=node_cache,use_cache=True)
    prefill=out.logits.clone();node_cache.freeze({i:dict(skip_near=near,skip_far=far) for i in range(2)})
    logits=[]
    for i in range(25):
        out=model(tokens[:,i:i+1],past_key_values=node_cache,attention_mask={'full_attention':None},use_cache=True)
        logits.append(out.logits)
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    def reference(module,q,k,v,attention_mask,**kwargs):
        if q.shape[2]==1:
            positions=expected_positions(k.shape[2],17,g,near,far)
            k,v=k[:,:,positions],v[:,:,positions]
            attention_mask=None;kwargs['is_causal']=False
        return original(module,q,k,v,attention_mask,**kwargs)
    ALL_ATTENTION_FUNCTIONS['sdpa']=reference
    try:
        cache=DynamicCache(config=config);out=model(prompt,past_key_values=cache,use_cache=True)
        torch.testing.assert_close(out.logits,prefill,atol=1e-6,rtol=1e-5)
        for i in range(25):
            out=model(tokens[:,i:i+1],past_key_values=cache,attention_mask={'full_attention':None},use_cache=True)
            torch.testing.assert_close(out.logits,logits[i],atol=1e-6,rtol=1e-5)
    finally:ALL_ATTENTION_FUNCTIONS['sdpa']=original


@pytest.mark.parametrize('n,f,expected',[(.03,.04,(True,True)),(.03,.06,(True,False)),(.06,.04,(False,True)),(.06,.07,(False,False)),(.05,.05,(False,False))])
def test_independent_mass_gates_four_states_and_threshold(n,f,expected):
    profile={'near':{'mean':n},'far':{'mean':f}}
    config=GateConfig(epsilon=.05)
    decision=decide_regions(profile,config)
    assert (decision['skip_near'],decision['skip_far'])==expected
    assert decision['skip_far']==decide_regions(profile,GateConfig(epsilon=.05,mode='far'))['skip_far']
    assert decision['skip_near']==decide_regions(profile,GateConfig(epsilon=.05,mode='near'))['skip_near']


def test_gate_other_region_mass_does_not_override_decision():
    for other in [0.,.03,.049,.05,.9]:
        assert decide_regions({'near':{'p95':.04},'far':{'p95':other}},GateConfig(epsilon=.05,statistic='p95'))['skip_near']
        assert decide_regions({'near':{'p95':other},'far':{'p95':.04}},GateConfig(epsilon=.05,statistic='p95'))['skip_far']


@pytest.mark.parametrize('near,far', [(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_tli_physical_arena_mapping_survives_ring_wraps(near, far):
    g = RegionGeometry(sink=2, near=6, window=3, page=4, max_new_tokens=40)
    cache = FrozenRegionCache(1, g); cache.tli_regions = True
    layer = cache.layers[0]
    initial = torch.arange(17.).reshape(1,1,17,1)
    cache.update(initial, initial*2, 0)
    old = [weakref.ref(node.keys) for node in layer.nodes.values()]
    cache.freeze({0:dict(skip_near=near,skip_far=far)}); gc.collect()
    assert all(ref() is None for ref in old)
    assert layer.arena_k.shape[0] == sum(node.capacity for node in layer.nodes.values())
    for t in range(17,49):
        new = torch.tensor([[[[float(t)]]]])
        layer.append_decode(new,new*2)
        k,v,positions = layer.gather()
        assert positions == expected_positions(t+1,17,g,near,far)
        logical = torch.tensor(positions,dtype=torch.int64)
        physical = layer.physical_positions(logical)
        gathered_k = layer.arena_k[physical].permute(1,0,2).unsqueeze(0)
        gathered_v = layer.arena_v[physical].permute(1,0,2).unsqueeze(0)
        assert torch.equal(gathered_k,k) and torch.equal(gathered_v,v)
        assert torch.equal(layer.logical_positions(),logical)


@torch.inference_mode()
def test_append_decode_never_gathers_visible_kv(monkeypatch):
    g = RegionGeometry(sink=2,near=6,window=3,page=4,max_new_tokens=40)
    cache = FrozenRegionCache(1,g); cache.tli_regions = True
    initial = torch.arange(17.).reshape(1,1,17,1)
    cache.update(initial,initial,0)
    cache.freeze({0:dict(skip_near=False,skip_far=False)})
    layer = cache.layers[0]
    monkeypatch.setattr(layer,'gather',lambda:(_ for _ in ()).throw(AssertionError('gather called')))
    for t in range(17,49):
        new = torch.tensor([[[[float(t)]]]])
        layer.append_decode(new,new)
    assert layer.logical_length == 49
    assert layer.nodes['far'].size > 0
