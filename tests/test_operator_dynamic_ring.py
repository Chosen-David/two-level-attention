"""Real DynamicCache append/ring lifecycle with independent retained oracle."""
import copy
from functools import partial
import pytest
import torch
from test_far_skip_hybrid_cache import make_model, prefill, isolated_runtime


@pytest.mark.parametrize('cache_mode',['evict','retain'])
def test_dynamic_operator_matches_retained_reference(monkeypatch,cache_mode):
    import sparse_attn.ops.partial_attention as ops
    import sparse_attn.ops.operator_pipeline as pipeline
    # CPU math oracle exercises routing/append semantics, not GPU admission.
    monkeypatch.setattr(pipeline,'validate_operator_inputs',lambda *args: None)
    native_partial=ops.partial_attention
    def torch_partial(*args,**kwargs):
        kwargs['backend']='torch'
        return native_partial(*args,**kwargs)
    monkeypatch.setattr(ops,'partial_attention',torch_partial)
    monkeypatch.setattr(ops,'merge_partials',partial(ops.merge_partials,backend='torch'))
    real_decode=pipeline.decode_selected
    calls=[]
    def counted(*args,**kwargs):
        calls.append(kwargs['layout'] is not None)
        return real_decode(*args,**kwargs)
    monkeypatch.setattr(pipeline,'decode_selected',counted)
    model=make_model(cache_mode)
    reference=copy.deepcopy(model)
    reference.model.layers[0].self_attn.indexer.far_skip_cache='retain'
    monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    actual_cache=prefill(model,23);reference_cache=prefill(reference,23)
    for pos in range(23,63):
        args=dict(input_ids=torch.tensor([[pos%32]]),position_ids=torch.tensor([[pos]]),
                  cache_position=torch.tensor([pos]),use_cache=True)
        monkeypatch.setenv('TLI_OPERATOR_MODE','serial')
        actual=model(**args,past_key_values=actual_cache).logits
        monkeypatch.setenv('TLI_OPERATOR_MODE','off')
        expected=reference(**args,past_key_values=reference_cache).logits
        assert actual_cache.get_seq_length(0)==pos+1,'duplicate or missing append'
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    assert len(calls)==40
    assert all(calls) if cache_mode=='evict' else not any(calls)


@pytest.mark.parametrize('backend,error', [('tc','TC backend requires'),('simt','requires CUDA')])
def test_unsupported_operator_preserves_cache(monkeypatch,backend,error):
    monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    model=make_model('evict');cache=prefill(model,23)
    layers=list(cache.layers)
    snapshots=[(layer.keys.data_ptr(),layer.values.data_ptr(),layer.keys.clone(),layer.values.clone())
               for layer in layers]
    monkeypatch.setenv('TLI_OPERATOR_MODE','serial')
    monkeypatch.setenv('TLI_OPERATOR_BACKEND',backend)
    with pytest.raises((RuntimeError,ValueError),match=error):
        model(input_ids=torch.tensor([[3]]),position_ids=torch.tensor([[23]]),
              cache_position=torch.tensor([23]),past_key_values=cache,use_cache=True)
    for i,(layer,saved) in enumerate(zip(layers,snapshots)):
        assert cache.layers[i] is layer
        assert cache.get_seq_length(i)==23
        assert (layer.keys.data_ptr(),layer.values.data_ptr())==saved[:2]
        torch.testing.assert_close(layer.keys,saved[2],rtol=0,atol=0)
        torch.testing.assert_close(layer.values,saved[3],rtol=0,atol=0)
