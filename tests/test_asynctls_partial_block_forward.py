"""CPU native-forward regression with a deterministic append-only KV cache."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(os.environ.get("ASYNCTLS_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
VENDOR = ROOT / "exp/vendor_baselines/asynctls_vendor"
sys.path.insert(0, str(VENDOR))
from qwen3_asynctls_patch import _qwen3_asynctls_forward
from llama3_asynctls_patch import _llama_asynctls_forward


class AppendCache:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values

    def get_seq_length(self, layer_idx):
        return self.keys.shape[2]

    def update(self, keys, values, layer_idx, cache_kwargs):
        self.keys = torch.cat([self.keys, keys], dim=2)
        self.values = torch.cat([self.values, values], dim=2)
        return self.keys, self.values


@pytest.mark.parametrize("forward", [_qwen3_asynctls_forward, _llama_asynctls_forward])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("tail_key,expected_value", [(2., 9.), (-2., 3.)])
def test_full_decode_selects_only_real_block_extrema(forward, sign, tail_key, expected_value):
    key = torch.tensor([-sign, 0.]).reshape(1, 1, 1, 2).expand(1, 1, 8, 2).clone()
    value = torch.full((1, 1, 8, 2), 3.)
    cache = AppendCache(key, value)
    config = SimpleNamespace(
        head_dim=2, layer_idx=0, scaling=1.,
        q_proj=lambda x: x.new_tensor([[[sign, 0.]]]),
        k_proj=lambda x: x.new_tensor([[[tail_key * sign, 0.]]]),
        v_proj=lambda x: x.new_full((1, 1, 2), 9.),
        q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(),
        o_proj=torch.nn.Identity(), asynctls_block_size=8,
        asynctls_top_kb=1, asynctls_token_budget=1, asynctls_channels=[0, 1])
    hidden = torch.zeros(1, 1, 2)
    position = (torch.ones(1, 1, 2), torch.zeros(1, 1, 2))
    output, weights = forward(config, hidden, position, past_key_values=cache)
    torch.testing.assert_close(output, torch.full_like(output, expected_value), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)
    assert cache.keys.shape[2] == 9  # Padding must never enter persistent KV.
    assert cache.values.shape[2] == 9


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_full_budget_tiny_model_matches_dense_across_page_boundary(tmp_path, model_type):
    import copy
    import json
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from qwen3_asynctls_patch import enable_asynctls as patch_qwen
    from llama3_asynctls_patch import enable_asynctls as patch_llama

    config_type, model_class, patch = (
        (Qwen3Config, Qwen3ForCausalLM, patch_qwen) if model_type == "qwen3"
        else (LlamaConfig, LlamaForCausalLM, patch_llama))
    config = config_type(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = model_class(config).eval()
    dense = copy.deepcopy(model)
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [list(range(8))] * 2}))
    assert patch(model, str(channels), token_budget=1024, block_size=8, top_kb=128) == 2
    prefix = torch.arange(15).unsqueeze(0)
    with torch.no_grad():
        actual = model(input_ids=prefix, use_cache=True)
        expected = dense(input_ids=prefix, use_cache=True)
        for token in (17, 18, 19):
            suffix = torch.tensor([[token]])
            actual = model(input_ids=suffix, past_key_values=actual.past_key_values, use_cache=True)
            expected = dense(input_ids=suffix, past_key_values=expected.past_key_values, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-5)
