import json
from types import SimpleNamespace

import pytest
import torch

from exp.vendor_baselines.asynctls_vendor.qwen3_asynctls_graph import (
    AsyncTLSQwen3GraphRuntime,
)
from exp.vendor_baselines.asynctls_vendor.qwen3_asynctls_patch import (
    _asynctls_sparse_attention,
)


def _reference_blocks(query, key, block_size, top_kb):
    """独立 naive reference：最后一个部分块只统计有效 token。"""
    kv_heads = key.shape[1]
    groups = query.shape[1] // kv_heads
    length = key.shape[2]
    scores = []
    for head in range(kv_heads):
        query_group = query[0, head * groups:(head + 1) * groups, 0]
        per_head = []
        for start in range(0, length, block_size):
            block = key[0, head, start:min(start + block_size, length)]
            score = (
                query_group.clamp(min=0) * block.amax(dim=0)
            ).sum() + (
                query_group.clamp(max=0) * block.amin(dim=0)
            ).sum()
            per_head.append(score)
        scores.append(torch.stack(per_head))
    return torch.topk(torch.stack(scores), k=top_kb, dim=-1).indices


def _reference_attention(runtime, layer_idx, query, key, value, previous):
    module = SimpleNamespace(
        asynctls_block_size=runtime.block_size,
        asynctls_channels=runtime.channels[layer_idx].cpu().tolist(),
        asynctls_token_budget=runtime.token_budget,
        scaling=runtime._layer_scalings[layer_idx],
    )
    return _asynctls_sparse_attention(module, query, key, value, previous)[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asynctls_graph_matches_reference_and_replays_pipeline(tmp_path):
    channel_path = tmp_path / "channels.json"
    channel_path.write_text(json.dumps({
        "schema": "asynctls-per-kv-group-channels-v1",
        "channels": [
            [list(range(32)), list(range(32))] for _ in range(2)
        ],
    }), encoding="utf-8")
    torch.manual_seed(20261110)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    layers, batch, length = 2, 2, 190
    q_heads, kv_heads, head_dim = 4, 2, 32
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=layers,
            num_attention_heads=q_heads,
            num_key_value_heads=kv_heads,
            hidden_size=q_heads * head_dim,
            head_dim=head_dim,
        ),
        model=SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace(sliding_window=None))
            for _ in range(layers)
        ]),
        dtype=dtype,
    )
    runtime = AsyncTLSQwen3GraphRuntime(
        model, batch, length, 3, device,
        channel_file=str(channel_path),
        token_budget=16, block_size=64, top_kb=2)
    runtime._layer_scalings = [0.173, 0.173]
    prompts = [[None for _ in range(batch)] for _ in range(layers)]
    for request_id in range(batch):
        for layer_idx in range(layers):
            query = torch.randn(
                1, q_heads, length, head_dim, device=device, dtype=dtype)
            key = torch.randn(
                1, kv_heads, length, head_dim, device=device, dtype=dtype)
            value = torch.randn(
                1, kv_heads, length, head_dim, device=device, dtype=dtype)
            runtime.append_prefill(layer_idx, query, key, value)
            prompts[layer_idx][request_id] = (query, key, value)
    runtime.prepare_decode("graph", 2)

    for layer_idx in range(layers):
        for request_id in range(batch):
            query, key, _ = prompts[layer_idx][request_id]
            expected = _reference_blocks(
                query[:, :, -1:], key, runtime.block_size, runtime.top_kb)
            torch.testing.assert_close(
                runtime.initial_blocks[layer_idx, request_id], expected, rtol=0, atol=0)

    queries = [[torch.randn(
        1, q_heads, 1, head_dim, device=device, dtype=dtype)
        for _ in range(batch)] for _ in range(layers)]
    keys = [[torch.randn(
        1, kv_heads, 1, head_dim, device=device, dtype=dtype)
        for _ in range(batch)] for _ in range(layers)]
    values = [[torch.randn(
        1, kv_heads, 1, head_dim, device=device, dtype=dtype)
        for _ in range(batch)] for _ in range(layers)]

    # 每个 layer 独立 capture，便于精确比较同一固定 buffer 的两次 replay。
    for layer_idx in range(layers):
        query = torch.cat(queries[layer_idx], dim=0)
        key = torch.cat(keys[layer_idx], dim=0)
        value = torch.cat(values[layer_idx], dim=0)
        runtime.prepare_decode_step(0, reset_pipeline=True)
        runtime.decode_attention(layer_idx, query, key, value)  # warmup
        runtime.prepare_decode_step(0, reset_pipeline=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runtime.decode_attention(layer_idx, query, key, value)
        graph.replay()
        torch.cuda.synchronize()

        expected_blocks = []
        for request_id in range(batch):
            prompt_q, prompt_k, prompt_v = prompts[layer_idx][request_id]
            full_k = torch.cat((prompt_k, keys[layer_idx][request_id]), dim=2)
            full_v = torch.cat((prompt_v, values[layer_idx][request_id]), dim=2)
            expected = _reference_attention(
                runtime, layer_idx, queries[layer_idx][request_id], full_k,
                full_v, runtime.initial_blocks[layer_idx, request_id])
            torch.testing.assert_close(
                runtime.decode_output[layer_idx, request_id],
                expected[0, :, 0], atol=2e-4, rtol=2e-4)
            expected_blocks.append(_reference_blocks(
                queries[layer_idx][request_id], full_k,
                runtime.block_size, runtime.top_kb))
        torch.testing.assert_close(
            runtime.previous_blocks[layer_idx], torch.stack(expected_blocks), rtol=0, atol=0)

        next_queries = [torch.randn_like(row) for row in queries[layer_idx]]
        next_keys = [torch.randn_like(row) for row in keys[layer_idx]]
        next_values = [torch.randn_like(row) for row in values[layer_idx]]
        query.copy_(torch.cat(next_queries, dim=0))
        key.copy_(torch.cat(next_keys, dim=0))
        value.copy_(torch.cat(next_values, dim=0))
        replay_blocks = runtime.previous_blocks[layer_idx].clone()
        runtime.prepare_decode_step(1)
        graph.replay()
        torch.cuda.synchronize()
        for request_id in range(batch):
            prompt_q, prompt_k, prompt_v = prompts[layer_idx][request_id]
            full_k = torch.cat((
                prompt_k, keys[layer_idx][request_id],
                next_keys[request_id]), dim=2)
            full_v = torch.cat((
                prompt_v, values[layer_idx][request_id],
                next_values[request_id]), dim=2)
            expected = _reference_attention(
                runtime, layer_idx, next_queries[request_id], full_k, full_v,
                replay_blocks[request_id])
            torch.testing.assert_close(
                runtime.decode_output[layer_idx, request_id],
                expected[0, :, 0], atol=2e-4, rtol=2e-4)
