import copy
import json
import sys
from pathlib import Path

import pytest
import torch


glm_modeling = pytest.importorskip(
    "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
    reason="GLM-4.7-Flash 需要支持 glm4_moe_lite 的 transformers",
)
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
    Glm4MoeLiteConfig,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exp" / "vendor_baselines" / "asynctls_vendor"))

from glm4_moe_lite_asynctls_patch import enable_asynctls


N_LAYERS = 3
N_KV_HEADS = 2
QK_HEAD_DIM = 24


def _tiny_model():
    config = Glm4MoeLiteConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_KV_HEADS,
        num_key_value_heads=N_KV_HEADS,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=QK_HEAD_DIM,
        max_position_embeddings=256,
        mlp_layer_types=["dense"] * N_LAYERS,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    return glm_modeling.Glm4MoeLiteForCausalLM(config).eval()


def _write_channels(tmp_path):
    path = tmp_path / "channels.json"
    path.write_text(json.dumps({
        "schema": "asynctls-per-kv-group-channels-v1",
        "channels": [
            [list(range(8)) for _ in range(N_KV_HEADS)]
            for _ in range(N_LAYERS)
        ],
    }))
    return str(path)


def _patch(model, channel_file, token_budget=4, top_kb=1):
    assert enable_asynctls(
        model,
        channel_file,
        token_budget=token_budget,
        block_size=8,
        top_kb=top_kb,
    ) == N_LAYERS


def _layers(model):
    return [layer.self_attn for layer in model.model.layers]


def _prefill(model, prefix):
    with torch.no_grad():
        return model(input_ids=prefix, use_cache=True)


def _decode(model, token, cache):
    with torch.no_grad():
        return model(
            input_ids=torch.tensor([[token]]),
            past_key_values=cache,
            use_cache=True,
        )


def test_glm_asynctls_full_budget_matches_native_dense(tmp_path):
    channels = _write_channels(tmp_path)
    model = _tiny_model()
    reference = copy.deepcopy(model)
    _patch(model, channels, token_budget=1024, top_kb=128)
    prefix = torch.arange(20).remainder(100).unsqueeze(0)

    actual_prefill = _prefill(model, prefix)
    expected_prefill = _prefill(reference, prefix)
    torch.testing.assert_close(
        actual_prefill.logits, expected_prefill.logits, rtol=1e-5, atol=1e-5)
    actual = _decode(model, 17, actual_prefill.past_key_values)
    expected = _decode(reference, 17, expected_prefill.past_key_values)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-5)


def test_glm_asynctls_decode_uses_previous_blocks_and_advances(
        tmp_path, monkeypatch):
    import qwen3_asynctls_patch as shared

    channels = _write_channels(tmp_path)
    model = _tiny_model()
    _patch(model, channels)
    prefill = _prefill(model, torch.arange(20).remainder(100).unsqueeze(0))
    initial = [layer._asynctls_prev_blocks.clone() for layer in _layers(model)]
    assert all(blocks.shape == (N_KV_HEADS, 1) for blocks in initial)

    seen = []
    original = shared._candidate_indices

    def record_candidates(top_blocks, length, block_size):
        seen.append(top_blocks.detach().clone())
        return original(top_blocks, length, block_size)

    monkeypatch.setattr(shared, "_candidate_indices", record_candidates)
    first = _decode(model, 17, prefill.past_key_values)
    after_first = [layer._asynctls_prev_blocks.clone() for layer in _layers(model)]
    _decode(model, 18, first.past_key_values)

    assert len(seen) == 2 * N_LAYERS
    for actual, expected in zip(seen[:N_LAYERS], initial):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(seen[N_LAYERS:], after_first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_glm_asynctls_second_prefill_replaces_previous_request_state(tmp_path):
    channels = _write_channels(tmp_path)
    model = _tiny_model()
    _patch(model, channels)
    fresh = copy.deepcopy(model)
    first_prefix = torch.arange(20).remainder(100).unsqueeze(0)
    second_prefix = torch.arange(24, 44).remainder(100).unsqueeze(0)

    first = _prefill(model, first_prefix)
    _decode(model, 17, first.past_key_values)
    _prefill(model, second_prefix)
    _prefill(fresh, second_prefix)

    for actual, expected in zip(_layers(model), _layers(fresh)):
        torch.testing.assert_close(
            actual._asynctls_prev_blocks,
            expected._asynctls_prev_blocks,
            rtol=0,
            atol=0,
        )


def test_glm_asynctls_sparse_attention_shape_and_current_token(tmp_path):
    channels = _write_channels(tmp_path)
    model = _tiny_model()
    _patch(model, channels, token_budget=4, top_kb=1)
    prefill = _prefill(model, torch.arange(20).remainder(100).unsqueeze(0))
    output = model(
        input_ids=torch.tensor([[17]]),
        past_key_values=prefill.past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    assert output.attentions[-1].shape == (1, N_KV_HEADS, 1, 4)


def test_glm_asynctls_rejects_invalid_channel_geometry(tmp_path):
    model = _tiny_model()
    path = tmp_path / "bad_channels.json"
    path.write_text(json.dumps({
        "channels": [[list(range(8))] for _ in range(N_LAYERS)]
    }))
    with pytest.raises(ValueError, match="KV groups"):
        _patch(model, str(path))

    model = _tiny_model()
    path.write_text(json.dumps({
        "channels": [
            [[0, 0, 1], list(range(8))]
            for _ in range(N_LAYERS)
        ],
    }))
    with pytest.raises(ValueError, match="channel 索引非法"):
        _patch(model, str(path))


def test_glm_asynctls_manual_generate_honors_eos_list(tmp_path):
    from benchmark.LongBench.pred import _manual_generate

    channels = _write_channels(tmp_path)
    model = _tiny_model()
    _patch(model, channels)
    eos_token = 117
    model.lm_head = torch.nn.Linear(32, 128, bias=True)
    torch.nn.init.zeros_(model.lm_head.weight)
    torch.nn.init.zeros_(model.lm_head.bias)
    with torch.no_grad():
        model.lm_head.bias[eos_token] = 1.0

    generated = _manual_generate(
        model,
        torch.arange(20).remainder(100).unsqueeze(0),
        torch.empty(1, 0, dtype=torch.long),
        max_gen=4,
        eos_token_id=[1, eos_token],
        stop_on_eos=True,
    )
    assert generated == [eos_token]
