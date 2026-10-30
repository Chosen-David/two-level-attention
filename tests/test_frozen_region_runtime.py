"""Shared frozen prefill/freeze/receipt lifecycle contract tests."""
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_runtime import (
    complete_frozen_request, prefill_frozen_request)


def _model():
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=19, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    return Qwen3ForCausalLM(config).eval()


@torch.inference_mode()
def test_each_prefill_owns_a_complete_fresh_cache_and_json_receipt():
    model = _model()
    ids = torch.tensor([[2, 3, 4, 5]])
    caches = []
    for _ in range(2):
        output, audit = prefill_frozen_request(
            model, ids, GateConfig(mode='full'), RegionGeometry(
                sink=1, near=4, window=2, page=2, max_new_tokens=3))
        cache = output.past_key_values
        caches.append(cache)
        assert set(audit['profiles']) == {0, 1}
        assert set(audit['decisions']) == {0, 1}
        assert not any(d['skip_near'] or d['skip_far']
                       for d in audit['decisions'].values())
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        model(token, past_key_values=cache, use_cache=True,
              attention_mask={'full_attention': None}, logits_to_keep=1)
        receipt = complete_frozen_request(model, cache, 2, audit)
        assert receipt['gate_counts'] == {'keep': 2, 'near': 0, 'far': 0, 'both': 0}
        json.dumps(receipt)
    assert caches[0] is not caches[1]
    assert caches[0].layers[0] is not caches[1].layers[0]


@torch.inference_mode()
def test_missing_real_tli_receipt_is_rejected_before_commit():
    model = _model()
    output, audit = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'),
        RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3))
    model._frozen_tli = True
    with pytest.raises(ValueError, match='missing real TLI decode receipts'):
        complete_frozen_request(model, output.past_key_values, 2, audit)


@torch.inference_mode()
def test_device_receipt_values_are_converted_once_at_request_boundary():
    model = _model()
    output, audit = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'),
        RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3))
    model._frozen_tli = True
    for layer in output.past_key_values.layers:
        layer.tli_receipt = {'selected': torch.tensor([1, 2]), 'l1_calls': 1}
    receipt = complete_frozen_request(model, output.past_key_values, 2, audit)
    assert receipt['tli_receipts'][0]['selected'] == [1, 2]
    json.dumps(receipt)


@torch.inference_mode()
def test_prefill_failure_restores_the_native_sdpa_hook():
    model = _model()
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    hook = model.register_forward_pre_hook(
        lambda module, args: (_ for _ in ()).throw(ValueError('synthetic prefill failure')))
    try:
        with pytest.raises(ValueError, match='synthetic prefill failure'):
            prefill_frozen_request(
                model, torch.tensor([[2, 3, 4, 5]]), GateConfig(), RegionGeometry())
    finally:
        hook.remove()
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
    output, _ = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'), RegionGeometry())
    assert output.past_key_values is not None
