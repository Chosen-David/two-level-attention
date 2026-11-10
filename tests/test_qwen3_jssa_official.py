from types import SimpleNamespace

import pytest
import torch

from exp.vendor_baselines.jssa_vendor.qwen3_jssa_official import (
    JSSAQwen3Runtime,
)


@pytest.mark.parametrize("budget", [64, 256])
def test_official_runtime_matches_reference_and_graph(tmp_path, budget):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(123)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch, num_qo_heads, num_kv_heads = 2, 8, 2
    head_dim, rank, length, page_size = 128, 32, 300, 16
    basis = torch.randn(num_kv_heads, rank, head_dim, dtype=torch.float32)
    basis_path = tmp_path / "basis.pt"
    torch.save({"M": {0: basis}}, basis_path)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=num_qo_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=num_qo_heads * head_dim,
    )
    model = SimpleNamespace(
        config=config,
        dtype=dtype,
        model=SimpleNamespace(layers=[SimpleNamespace(
            self_attn=SimpleNamespace(sliding_window=None))]),
    )
    runtime = JSSAQwen3Runtime(
        model, batch, length, 8, device, str(basis_path),
        budget=budget, rank=rank, page_size=page_size)
    runtime._layer_scalings = [head_dim ** -0.5]
    for request_id in range(batch):
        key = torch.randn(
            1, num_kv_heads, length, head_dim, device=device, dtype=dtype)
        value = torch.randn_like(key)
        runtime.begin_prefill_request(request_id)
        runtime.append_prefill(0, key, value)

    def check(query, key, value, sequence_length):
        logical_cache = runtime.kv_cache[0].view(
            batch, -1, num_kv_heads, 2 * head_dim)
        all_keys = logical_cache[:, :sequence_length, :, :head_dim]
        all_values = logical_cache[:, :sequence_length, :, head_dim:]
        for request_id in range(batch):
            torch.testing.assert_close(
                all_keys[request_id, -1], key[request_id, :, 0], rtol=0, atol=0)
            torch.testing.assert_close(
                all_values[request_id, -1], value[request_id, :, 0], rtol=0, atol=0)
        expected_rows = []
        for request_id in range(batch):
            head_rows = []
            for kv_head in range(num_kv_heads):
                indices = runtime.top_idx[request_id, kv_head].long()
                selected_k = all_keys[request_id, indices, kv_head].float()
                selected_v = all_values[request_id, indices, kv_head].float()
                grouped_q = query[request_id, :, 0].view(
                    num_kv_heads, num_qo_heads // num_kv_heads, head_dim
                )[kv_head].float()
                probabilities = torch.softmax(
                    grouped_q @ selected_k.T * (head_dim ** -0.5), dim=-1)
                head_rows.append(probabilities @ selected_v)
            expected_rows.append(torch.cat(head_rows, dim=0))
        expected = torch.stack(expected_rows)
        actual = runtime.decode_output[0].float()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        # Independently dequantize the official FP8 cache and check its score.
        projected_keys = runtime.projk_cache[0].view(
            batch, -1, num_kv_heads, rank
        )[0, :sequence_length, 0].float()
        expected_scores = projected_keys @ runtime.projq_hrc[0, 0].float()
        actual_scores = runtime.scores[0, 0, :sequence_length]
        torch.testing.assert_close(
            actual_scores, expected_scores, rtol=2e-3, atol=2e-3)

    runtime.prepare_decode(None, 0)
    query = torch.randn(
        batch, num_qo_heads, 1, head_dim, device=device, dtype=dtype)
    key = torch.randn(
        batch, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    runtime.prepare_decode_step(0)
    runtime.decode_attention(0, query, key, value)
    check(query, key, value, length + 1)

    runtime.prepare_decode_step(0)
    runtime.decode_attention(0, query, key, value)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    runtime.prepare_decode_step(0)
    with torch.cuda.graph(graph):
        runtime.decode_attention(0, query, key, value)
    query.copy_(torch.randn_like(query))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    runtime.prepare_decode_step(1)
    graph.replay()
    check(query, key, value, length + 2)

    receipt = runtime.receipt()
    assert receipt["budget"] == budget
    assert receipt["page_size"] == page_size
    assert receipt["decode_attention_calls_warmup_and_capture"] == 3
