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
