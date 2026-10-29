"""Short live-width requests must be rejected before any CUDA allocation."""
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops import quality_graph as quality


def fake_model():
    parameter=SimpleNamespace(device=torch.device('cuda'),dtype=torch.bfloat16)
    return SimpleNamespace(parameters=lambda:iter([parameter]))


def test_strict_short_logical_width_rejected_before_allocation(monkeypatch):
    monkeypatch.setenv('TLI_CUDA_GRAPH','1')
    monkeypatch.setenv('TLI_LIVE_NORMALIZATION','1')
    monkeypatch.setenv('TLI_QUALITY_TIMING','0')
    def forbidden(*args,**kwargs):
        raise AssertionError('CUDA allocation before live-width admission')
    monkeypatch.setattr(torch,'full',forbidden)
    cache=SimpleNamespace(get_max_cache_shape=lambda:20000)
    ctx=SimpleNamespace(alive=[True])
    with pytest.raises(RuntimeError,match='live_normalization_length_below_9217'):
        quality.static_decode(fake_model(),cache,[[1]],[9215],set(),10,ctx)


@pytest.mark.parametrize('lens,reason',[([9215],'live_normalization_length_below_9217'),([9216],None),([16000,9000],'live_normalization_length_below_9217'),([16000,9216],None)])
def test_live_width_admission_uses_first_decode_length(monkeypatch,lens,reason):
    monkeypatch.setenv('TLI_LIVE_NORMALIZATION','1')
    assert quality._live_width_fallback(lens)==reason


def test_disabled_live_width_does_not_restrict_old_route(monkeypatch):
    monkeypatch.setenv('TLI_LIVE_NORMALIZATION','0')
    assert quality._live_width_fallback([1]) is None


@pytest.mark.parametrize('policy',['auto','1','0'])
def test_dynamic_cache_admission_precedes_conversion(monkeypatch,policy):
    monkeypatch.setenv('TLI_CUDA_GRAPH',policy)
    monkeypatch.setenv('TLI_LIVE_NORMALIZATION','1')
    model=fake_model()
    indexer=type('TLIIndexer',(),{})()
    model.modules=lambda:[SimpleNamespace(indexer=indexer)]
    model.config=SimpleNamespace(model_type='qwen3')
    past=SimpleNamespace(get_seq_length=lambda:9215)
    if policy=='1':
        with pytest.raises(RuntimeError,match='live_normalization_length_below_9217'):
            quality.from_dynamic(model,past,1,10000,set())
    else:
        assert quality.from_dynamic(model,past,1,10000,set()) is None
        expected=None if policy=='0' else 'live_normalization_length_below_9217'
        assert model._quality_graph_execution['actual']=='eager'
        assert model._quality_graph_execution['fallback_reason']==expected
