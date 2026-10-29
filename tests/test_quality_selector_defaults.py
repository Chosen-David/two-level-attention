"""Quality-only defaults must match host admission and reset across requests."""
import pytest
import torch
from sparse_attn.ops.cuda_graph_state import enable_graph, disable_graph, graph_state
from sparse_attn.ops.sparse_policy import live_normalization_enabled, stats_policy, selection_fallback
from sparse_attn.ops.quality_graph import _live_width_fallback


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for key in ('TLI_LIVE_NORMALIZATION', 'TLI_SPARSE_SELECTION', 'TLI_SPARSE_STATS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch, '__version__', '2.8.0+cu128')
    disable_graph()
    yield
    disable_graph()


def route():
    return selection_fallback(torch.bfloat16, torch.device('cuda'), 16401, 2, 8,
                              l2_bypass=False, protected_end=128, topk=1024)


def test_quality_defaults_are_request_scoped():
    assert not live_normalization_enabled()
    assert route() == 'disabled'
    enable_graph(1, quality_defaults=True)
    assert live_normalization_enabled()
    assert stats_policy() == 'owner'
    assert route() is None
    disable_graph()
    assert not graph_state().quality_defaults
    enable_graph(1)
    assert not live_normalization_enabled()
    assert route() == 'disabled'


def test_host_admission_uses_quality_defaults_before_enable():
    assert not graph_state().enabled
    assert _live_width_fallback([9215], torch.bfloat16) == 'live_normalization_length_below_9217'
    assert _live_width_fallback([16384], torch.float16) == 'live_normalization_requires_bfloat16'
    assert _live_width_fallback([9216], torch.bfloat16) is None


def test_explicit_policies_override_quality_defaults(monkeypatch):
    enable_graph(1, quality_defaults=True)
    monkeypatch.setenv('TLI_LIVE_NORMALIZATION', '0')
    monkeypatch.setenv('TLI_SPARSE_SELECTION', '0')
    monkeypatch.setenv('TLI_SPARSE_STATS', 'sorted')
    assert not live_normalization_enabled()
    assert _live_width_fallback([1], torch.float32) is None
    assert route() == 'disabled'
    monkeypatch.setenv('TLI_SPARSE_SELECTION', '1')
    assert route() == 'unaligned_requires_owner'
