import copy
import json
import sys
from pathlib import Path
from types import MethodType

import pytest
import torch

from transformers import LlamaConfig, LlamaForCausalLM

REPO = Path(__file__).resolve().parents[1]
for vendor in ("quest_vendor", "jssa_vendor", "asynctls_vendor"):
    sys.path.insert(0, str(REPO / "exp" / "vendor_baselines" / vendor))

from llama3_quest_patch import enable_quest
from llama3_jssa_patch import enable_jssa
from llama3_asynctls_patch import enable_asynctls
from qwen3_asynctls_patch import _int4_quant_dequant


def _tiny_model():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return LlamaForCausalLM(config).eval()


def _dense_then_decode(model, prefix, suffix, output_attentions=False):
    with torch.no_grad():
        prefill = model(input_ids=prefix, use_cache=True)
        decode = model(
            input_ids=suffix,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
        )
    return decode


def _run_pair(patch, prefix, suffix, output_attentions=False):
    model = _tiny_model()
    reference = copy.deepcopy(model)
    patch(model)
    actual = _dense_then_decode(model, prefix, suffix, output_attentions)
    expected = _dense_then_decode(reference, prefix, suffix, output_attentions)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-5)
    return model, actual


def test_llama_quest_full_budget_matches_dense_and_uses_official_decode():
    prefix = torch.arange(40).remainder(100).unsqueeze(0)
    suffix = torch.tensor([[17]])
    holder = {}

    def patch(model):
        assert enable_quest(model, token_budget=1024, chunk_size=16) == 3
        layer = model.model.layers[2].self_attn
        official = layer._official_forward
        holder["calls"] = 0

        def counted(self, *args, **kwargs):
            holder["calls"] += 1
            return official(*args, **kwargs)

        layer._official_forward = MethodType(counted, layer)

    _run_pair(patch, prefix, suffix)
    assert holder["calls"] == 1


def test_llama_jssa_full_budget_matches_dense_and_sparse_shape(tmp_path):
    prefix = torch.arange(40).remainder(100).unsqueeze(0)
    suffix = torch.tensor([[17]])
    basis = {
        "M": {
            layer: torch.eye(16)[:, :8].transpose(0, 1).repeat(2, 1, 1)
            for layer in range(3)
        }
    }
    basis_path = tmp_path / "basis.pt"
    torch.save(basis, basis_path)

    _run_pair(
        lambda model: enable_jssa(
            model, str(basis_path), budget=1024, rank=8) == 3,
        prefix, suffix)

    model = _tiny_model()
    assert enable_jssa(model, str(basis_path), budget=4, rank=8) == 3
    output = _dense_then_decode(model, prefix, suffix, output_attentions=True)
    assert output.attentions[-1].shape[-1] == 4


def test_asynctls_int4_covers_all_16_levels():
    values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 16)
    torch.testing.assert_close(
        _int4_quant_dequant(values), values, rtol=0, atol=0)
    constant = torch.full((1, 1, 1, 16), 3.25)
    torch.testing.assert_close(
        _int4_quant_dequant(constant), constant, rtol=0, atol=0)


def test_llama_asynctls_full_budget_matches_dense_and_sparse_shape(tmp_path):
    prefix = torch.arange(40).remainder(100).unsqueeze(0)
    suffix = torch.tensor([[17]])
    channel_path = tmp_path / "channels.json"
    channel_path.write_text(json.dumps({
        "channels": [[list(range(8)) for _ in range(2)] for _ in range(3)]
    }))

    _run_pair(
        lambda model: enable_asynctls(
            model, str(channel_path), token_budget=1024,
            block_size=8, top_kb=128) == 3,
        prefix, suffix)

    model = _tiny_model()
    assert enable_asynctls(
        model, str(channel_path), token_budget=4,
        block_size=8, top_kb=1) == 3
    output = _dense_then_decode(model, prefix, suffix, output_attentions=True)
    assert output.attentions[-1].shape[-1] == 4


def test_llama_base_prompt_is_raw():
    from benchmark.LongBench.pred import build_chat
    assert build_chat(None, "raw LongBench prompt", "Llama-3.1-8B") == (
        "raw LongBench prompt")
