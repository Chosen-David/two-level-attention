"""Real CPU request-lifecycle regressions; no GPU/quality/performance claim."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                          )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp/vendor_baselines/asynctls_vendor'))
from llama3_asynctls_patch import enable_asynctls as enable_llama
from qwen3_asynctls_patch import enable_asynctls as enable_qwen
try:
    from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
except ImportError:
    Glm4MoeLiteConfig = None
if Glm4MoeLiteConfig is not None:
    from glm4_moe_lite_asynctls_patch import enable_asynctls as enable_glm


def model_and_enable(family, layers=1):
    if family == 'glm':
        if Glm4MoeLiteConfig is None:
            pytest.skip("GLM requires Transformers 5")
        cfg = Glm4MoeLiteConfig(vocab_size=32, hidden_size=32,
            intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=layers,
            num_attention_heads=2, num_key_value_heads=2, kv_lora_rank=8,
            q_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4,
            v_head_dim=8, n_routed_experts=2, num_experts_per_tok=1,
            max_position_embeddings=128)
        cfg._attn_implementation = 'eager'
        torch.manual_seed(21)
        return Glm4MoeLiteForCausalLM(cfg).eval(), enable_glm
    config_cls, model_cls, enable = ((LlamaConfig, LlamaForCausalLM, enable_llama)
        if family == 'llama' else (Qwen3Config, Qwen3ForCausalLM, enable_qwen))
    cfg = config_cls(vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128)
    cfg._attn_implementation = 'eager'
    torch.manual_seed(21)
    return model_cls(cfg).eval(), enable


def save_channels(tmp_path, values):
    path = tmp_path / 'channels.json'
    path.write_text(json.dumps({'channels': [[list(v), list(v)] for v in values]}))
    return str(path)



@pytest.fixture(params=['qwen', 'llama', 'glm'])
def setup(request, tmp_path):
    model, enable = model_and_enable(request.param)
    reference = copy.deepcopy(model)
    enable(model, save_channels(tmp_path, [range(4)]), token_budget=4, block_size=8, top_kb=1)
    return model, reference, model.model.layers[0].self_attn


def run(model, length=17, cache=None, offset=0):
    with torch.no_grad():
        return model(input_ids=(torch.arange(length).reshape(1,-1)+offset)%32,
                     past_key_values=cache, use_cache=True)


def snapshot(cache):
    return [(l.keys.clone(), l.values.clone()) for l in cache.layers]


def unchanged(cache, before):
    for layer, (k, v) in zip(cache.layers, before):
        torch.testing.assert_close(layer.keys,k,rtol=0,atol=0)
        torch.testing.assert_close(layer.values,v,rtol=0,atol=0)


def test_interleaved_same_length_cache_rejected_before_update(setup):
    model, _, attn = setup
    a = run(model).past_key_values
    b = run(model, offset=11).past_key_values
    assert a is not b and a.get_seq_length() == b.get_seq_length()
    before = snapshot(a)
    with pytest.raises(RuntimeError, match='AsyncTLS.*(owner|request|cache)'):
        run(model, 1, a)
    unchanged(a, before)


def test_cropped_cache_rejected_before_update(setup):
    model, _, _ = setup
    cache = run(model).past_key_values
    cache.crop(-8)
    before = snapshot(cache)
    with pytest.raises(RuntimeError, match='AsyncTLS.*(length|cache)'):
        run(model, 1, cache)
    unchanged(cache, before)


def test_uninitialized_nonempty_cache_rejected_before_update(setup):
    model, reference, _ = setup
    cache = run(reference).past_key_values
    before = snapshot(cache)
    with pytest.raises(RuntimeError, match='AsyncTLS'):
        run(model, 1, cache)
    unchanged(cache, before)


def test_serial_requests_and_multi_step_sparse_are_repeatable(setup):
    model, _, _ = setup
    isolated = copy.deepcopy(model)
    for length in [1, 17, 9]:
        expected = run(isolated, length).past_key_values
        actual = run(model, length).past_key_values
        for token in [3, 4, 5]:
            a, e = run(model,1,actual,token), run(isolated,1,expected,token)
            torch.testing.assert_close(a.logits,e.logits,rtol=0,atol=0)
            unchanged(actual,snapshot(expected))


def test_full_budget_matches_native_dense_multi_step(setup):
    model, reference, attn = setup
    attn.asynctls_token_budget=1024
    attn.asynctls_top_kb=128
    actual, expected = run(model).past_key_values, run(reference).past_key_values
    for token in [2, 3, 4]:
        a,e=run(model,1,actual,token),run(reference,1,expected,token)
        torch.testing.assert_close(a.logits,e.logits,rtol=3e-5,atol=3e-5)


def test_blocks_do_not_keep_cache_alive(setup):
    import weakref, gc
    model, _, _ = setup
    cache=run(model).past_key_values
    ref=weakref.ref(cache)
    del cache
    gc.collect()
    assert ref() is None


def test_empty_cache_restart_rebinds(setup):
    model, _, _ = setup
    cache=run(model).past_key_values
    cache.crop(-17)
    actual=run(model,1,cache,7)
    assert actual.past_key_values.get_seq_length()==1
    run(model,1,cache,8)
    assert cache.get_seq_length()==2


def test_failed_sparse_step_cannot_reuse_stale_blocks(setup, monkeypatch):
    model, _, attn=setup
    cache=run(model).past_key_values
    forward=attn.o_proj.forward
    def fail(*args, **kwargs):
        raise RuntimeError('injected projection failure')
    monkeypatch.setattr(attn.o_proj,'forward',fail)
    with pytest.raises(RuntimeError,match='injected'):
        run(model,1,cache)
    monkeypatch.setattr(attn.o_proj,'forward',forward)
    before=snapshot(cache)
    with pytest.raises(RuntimeError,match='AsyncTLS'):
        run(model,1,cache)
    unchanged(cache,before)
    # Explicit fresh prefill recovers, not a retry on half-written cache.
    fresh=run(model).past_key_values
    run(model,1,fresh)


def test_non_weakref_cache_is_rejected_without_state_mutation(setup):
    from qwen3_asynctls_patch import _begin_asynctls_step
    model, _, attn=setup
    cache=run(model).past_key_values
    blocks=attn._asynctls_prev_blocks
    owner=attn._asynctls_cache_owner
    length=attn._asynctls_cache_length
    with pytest.raises(TypeError,match='AsyncTLS.*weak-referenceable'):
        _begin_asynctls_step(attn, object(),0,decode=False)
    assert attn._asynctls_prev_blocks is blocks
    assert attn._asynctls_cache_owner is owner
    assert attn._asynctls_cache_length == length
    run(model,1,cache)


@pytest.mark.parametrize('family',['qwen','llama','glm'])
def test_late_layer_failure_requires_new_request(family,tmp_path,monkeypatch):
    model,enable=model_and_enable(family,layers=2)
    enable(model,save_channels(tmp_path,[range(4),range(4)]),
           token_budget=4,block_size=8,top_kb=1)
    cache=run(model).past_key_values
    late=model.model.layers[1].self_attn
    forward=late.o_proj.forward
    def fail(*args,**kwargs):
        raise RuntimeError('injected late-layer failure')
    monkeypatch.setattr(late.o_proj,'forward',fail)
    with pytest.raises(RuntimeError,match='injected'):
        run(model,1,cache)
    monkeypatch.setattr(late.o_proj,'forward',forward)
    before=(cache.layers[1].keys.clone(),cache.layers[1].values.clone())
    with pytest.raises(RuntimeError,match='AsyncTLS'):
        run(model,1,cache)
    # Guard is per-layer, not a model-wide cache transaction: earlier layers may
    # have advanced on this invalid retry; the failed layer cannot reuse M_t.
    torch.testing.assert_close(cache.layers[1].keys,before[0],rtol=0,atol=0)
    torch.testing.assert_close(cache.layers[1].values,before[1],rtol=0,atol=0)
    fresh=run(model).past_key_values
    run(model,1,fresh)
