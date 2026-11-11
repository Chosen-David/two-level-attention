"""Real CPU model regressions for transactional AsyncTLS installation."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                          Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp/vendor_baselines/asynctls_vendor'))
from llama3_asynctls_patch import enable_asynctls as enable_llama
from qwen3_asynctls_patch import enable_asynctls as enable_qwen
from glm4_moe_lite_asynctls_patch import enable_asynctls as enable_glm


def model_and_enable(family):
    if family == 'glm':
        cfg = Glm4MoeLiteConfig(vocab_size=32, hidden_size=32,
            intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=3,
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
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128)
    cfg._attn_implementation = 'eager'
    torch.manual_seed(21)
    return model_cls(cfg).eval(), enable


def save_channels(tmp_path, values):
    path = tmp_path / 'channels.json'
    path.write_text(json.dumps({'channels': [[list(v), list(v)] for v in values]}))
    return str(path)


@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
@pytest.mark.parametrize('invalid', ['missing', 'extra', 'negative', 'duplicate', 'out_of_range', 'empty', 'boolean'])
def test_failed_install_does_not_mutate(family, invalid, tmp_path):
    model, enable = model_and_enable(family)
    channels = [[0, 1], [0, 1], [0, 1]]
    if invalid == 'missing': channels.pop()
    elif invalid == 'extra': channels.append([0, 1])
    else: channels[-1] = {'negative': [-1, 0], 'duplicate': [0, 0],
        'out_of_range': [0, 8], 'empty': [], 'boolean': [False, 1]}[invalid]
    old = [dict(layer.self_attn.__dict__) for layer in model.model.layers]
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad(): expected = model(tokens).logits
    with pytest.raises(ValueError):
        enable(model, save_channels(tmp_path, channels), block_size=8)
    for layer, attrs in zip(model.model.layers, old):
        assert set(layer.self_attn.__dict__) == set(attrs)
        assert layer.self_attn.forward == attrs.get('forward', type(layer.self_attn).forward.__get__(layer.self_attn))
    with torch.no_grad(): actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
def test_reinstall_rejected_without_losing_original_forward(family, tmp_path):
    model, enable = model_and_enable(family)
    path = save_channels(tmp_path, [[0, 1]] * 3)
    assert enable(model, path, block_size=8) == 3
    forwards = [(x.self_attn.forward, x.self_attn.flash_forward) for x in model.model.layers]
    with pytest.raises(ValueError): enable(model, path, block_size=8)
    assert forwards == [(x.self_attn.forward, x.self_attn.flash_forward) for x in model.model.layers]
    with torch.no_grad(): assert torch.isfinite(model(torch.tensor([[1, 2, 3]])).logits).all()


@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
def test_native_layer_id_binds_channels(family, tmp_path):
    model, enable = model_and_enable(family)
    model.model.layers = torch.nn.ModuleList(reversed(model.model.layers))
    channels = [[0, 1], [2, 3], [4, 5]]
    assert enable(model, save_channels(tmp_path, channels), block_size=8) == 3
    for layer in model.model.layers:
        assert layer.self_attn.asynctls_channels == [channels[layer.self_attn.layer_idx]] * 2


@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
def test_duplicate_layer_id_rejected(family, tmp_path):
    model, enable = model_and_enable(family)
    model.model.layers[-1].self_attn.layer_idx = 0
    with pytest.raises(ValueError): enable(model, save_channels(tmp_path, [[0, 1]] * 3), block_size=8)
    assert not any(hasattr(x.self_attn, 'asynctls_channels') for x in model.model.layers)

@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
def test_valid_full_budget_decode_matches_dense(family, tmp_path):
    model, enable = model_and_enable(family)
    reference = copy.deepcopy(model)
    path = save_channels(tmp_path, [list(range(8))] * 3)
    assert enable(model, path, block_size=8) == 3
    with torch.no_grad():
        def run(m):
            cache = m(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), use_cache=True).past_key_values
            return m(torch.tensor([[8]]), past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(run(model), run(reference), rtol=3e-5, atol=3e-5)

@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
def test_layer_dictionary_does_not_silently_relabel_missing_ids(family, tmp_path):
    model, enable = model_and_enable(family)
    path = tmp_path / 'channels.json'
    path.write_text(json.dumps({'layer_0': [[0, 1]] * 2, 'layer_2': [[2, 3]] * 2, 'layer_3': [[4, 5]] * 2}))
    with pytest.raises(ValueError): enable(model, str(path), block_size=8)
    assert not any(hasattr(x.self_attn, 'asynctls_channels') for x in model.model.layers)

@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
@pytest.mark.parametrize('format', ['list', 'dictionary', 'legacy_dictionary'])
def test_other_supported_channel_formats(family, format, tmp_path):
    model, enable = model_and_enable(family)
    channels = [[0, 1], [2, 3], [4, 5]]
    prefix = 'layer' if format == 'dictionary' else 'legacy_head'
    nested = [[list(v), list(v)] for v in channels]
    raw = nested if format == 'list' else {f'{prefix}_{i}': nested[i] for i in [2, 0, 1]}
    path = tmp_path / 'channels.json'
    path.write_text(json.dumps(raw))
    assert enable(model, str(path), block_size=8) == 3
    for layer in model.model.layers:
        assert layer.self_attn.asynctls_channels == [channels[layer.self_attn.layer_idx]] * 2

@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
def test_duplicate_dictionary_ids_rejected(family, tmp_path):
    model, enable = model_and_enable(family)
    path = tmp_path / 'channels.json'
    path.write_text(json.dumps({'layer_0': [[0, 1]] * 2, 'other_0': [[2, 3]] * 2, 'layer_1': [[4, 5]] * 2}))
    with pytest.raises(ValueError): enable(model, str(path), block_size=8)
    assert not any(hasattr(x.self_attn, 'asynctls_channels') for x in model.model.layers)

@pytest.mark.parametrize('family', ['qwen', 'llama', 'glm'])
@pytest.mark.parametrize('bad', ['group_count', 'group_width', 'legacy_global'])
def test_group_calibration_contract(family, bad, tmp_path):
    model, enable = model_and_enable(family)
    nested = [[[0, 1], [2, 3]] for _ in range(3)]
    if bad == 'group_count': nested[-1] = [[0, 1]]
    elif bad == 'group_width': nested[-1] = [[0, 1], [2]]
    else: nested = [[0, 1] for _ in range(3)]
    path = tmp_path / 'channels.json'
    path.write_text(json.dumps({'channels': nested}))
    with pytest.raises(ValueError): enable(model, str(path), block_size=8)
    assert not hasattr(model, '_asynctls_full_prompt_prefill')
    assert not any(hasattr(x.self_attn, 'asynctls_channels') for x in model.model.layers)
