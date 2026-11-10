"""Real CPU request-lifecycle regressions; no GPU/quality/performance claim."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                          )

import os
ROOT = Path(os.environ.get('ASYNCTLS_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'exp/vendor_baselines/asynctls_vendor'))
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


@pytest.mark.parametrize("family", ["qwen", "llama", "glm"])
@pytest.mark.parametrize("length", [7, 8, 9, 15, 16, 17])
def test_full_budget_dense_parity_and_shared_padding(tmp_path, family, length, monkeypatch):
    import qwen3_asynctls_patch as shared
    module_name = {"qwen": "qwen3", "llama": "llama3", "glm": "glm4_moe_lite"}[family]
    adapter = sys.modules[module_name + "_asynctls_patch"]
    model, enable = model_and_enable(family, layers=2)
    dense = copy.deepcopy(model)
    enable(model, save_channels(tmp_path, [range(4), range(4)]),
           token_budget=128, block_size=8, top_kb=128)
    calls = []
    real_pad = shared._padded_key
    def checked_pad(keys, block_size):
        before = keys.clone()
        padded = real_pad(keys, block_size)
        torch.testing.assert_close(keys, before, rtol=0, atol=0)
        n = keys.shape[2]
        for start in range(0, n, block_size):
            real = keys[:, :, start:min(start + block_size, n)]
            padded_block = padded[:, :, start:start + block_size]
            torch.testing.assert_close(padded_block.amin(2), real.amin(2), rtol=0, atol=0)
            torch.testing.assert_close(padded_block.amax(2), real.amax(2), rtol=0, atol=0)
        calls.append(n)
        return padded
    monkeypatch.setattr(adapter, "_padded_key", checked_pad)
    with torch.no_grad():
        prefix = torch.arange(length).reshape(1, -1) % 32
        actual = model(input_ids=prefix, use_cache=True)
        expected = dense(input_ids=prefix, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-5)
        for step in range(3):
            token = torch.tensor([[step + 18]])
            actual = model(input_ids=token, past_key_values=actual.past_key_values, use_cache=True)
            expected = dense(input_ids=token, past_key_values=expected.past_key_values, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-5)
            for layer in actual.past_key_values.layers:
                assert layer.keys.shape[2] == length + step + 1
                assert layer.values.shape[2] == length + step + 1
    assert calls == [n for n in range(length, length + 4) for _ in range(2)]
