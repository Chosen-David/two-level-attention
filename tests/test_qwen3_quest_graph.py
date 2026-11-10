from types import SimpleNamespace

import pytest
import torch

from exp.vendor_baselines.quest_vendor.qwen3_quest_graph import (
    QuestQwen3GraphRuntime,
)


def _reference_attention(runtime, layer_idx, query, key, value):
    """独立逐 query-head naive reference；不调用 runtime 的打分或选页代码。"""
    q = query[0, :, 0]
    k = key[0]
    v = value[0]
    length = k.shape[1]
    position = length - 1
    groups = runtime.num_kv_groups
    outputs = []
    selected_pages = []
    for head in range(runtime.num_qo_heads):
        kv_head = head // groups
        if layer_idx < runtime.dense_layers:
            page_ids = None
            token_ids = torch.arange(length, device=key.device)
        else:
            current_page = position // runtime.page_size
            scores = []
            for page in range(current_page):
                start = page * runtime.page_size
                stop = min(start + runtime.page_size, position)
                block = k[kv_head, start:stop]
                query_head = q[head]
                positive = torch.matmul(
                    query_head.clamp(min=0).float(),
                    block.amax(dim=0).float()).to(key.dtype)
                negative = torch.matmul(
                    query_head.clamp(max=0).float(),
                    block.amin(dim=0).float()).to(key.dtype)
                scores.append(positive + negative)
            history = torch.topk(
                torch.stack(scores), k=runtime.history_topk).indices
            page_ids = torch.cat((
                history,
                torch.tensor([current_page], device=key.device)))
            token_ids = (
                page_ids[:, None] * runtime.page_size
                + torch.arange(runtime.page_size, device=key.device)
            ).reshape(-1)
            token_ids = torch.sort(token_ids).values
            token_ids = token_ids[token_ids <= position]
        selected_k = k[kv_head].index_select(0, token_ids)
        selected_v = v[kv_head].index_select(0, token_ids)
        weights = torch.matmul(q[head], selected_k.transpose(0, 1))
        weights = weights * runtime._layer_scalings[layer_idx]
        weights = torch.softmax(weights.float(), dim=-1).to(key.dtype)
        outputs.append(torch.matmul(weights, selected_v))
        selected_pages.append(page_ids)
    return torch.stack(outputs), selected_pages


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quest_graph_matches_naive_reference_and_replays():
    torch.manual_seed(20261110)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    layers, batch, length = 3, 2, 190
    q_heads, kv_heads, head_dim = 8, 2, 32
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
    runtime = QuestQwen3GraphRuntime(
        model, batch, length, 3, device,
        token_budget=64, page_size=16, dense_layers=2,
        record_selection=True)
    runtime._layer_scalings = [0.173] * layers
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

    # 非对齐尾页只统计有效 token，不能用 padding 改变 min/max。
    for layer_idx in range(layers):
        for request_id in range(batch):
            key = prompts[layer_idx][request_id][1][0].transpose(0, 1)
            tail = key[-(length % 16):].transpose(0, 1)
            torch.testing.assert_close(
                runtime.k_min[layer_idx, request_id, :, runtime.prefill_pages - 1],
                tail.amin(dim=1), rtol=0, atol=0)
            torch.testing.assert_close(
                runtime.k_max[layer_idx, request_id, :, runtime.prefill_pages - 1],
                tail.amax(dim=1), rtol=0, atol=0)

    queries = [[torch.randn(
        1, q_heads, 1, head_dim, device=device, dtype=dtype)
        for _ in range(batch)] for _ in range(layers)]
    keys = [[torch.randn(
        1, kv_heads, 1, head_dim, device=device, dtype=dtype)
        for _ in range(batch)] for _ in range(layers)]
    values = [[torch.randn(
        1, kv_heads, 1, head_dim, device=device, dtype=dtype)
        for _ in range(batch)] for _ in range(layers)]

    # 每层独立 capture，精确检查同一固定 buffer 的 step0/step1 replay。
    for layer_idx in range(layers):
        query = torch.cat(queries[layer_idx], dim=0)
        key = torch.cat(keys[layer_idx], dim=0)
        value = torch.cat(values[layer_idx], dim=0)
        runtime.prepare_decode_step(0)
        runtime.decode_attention(layer_idx, query, key, value)  # warmup
        runtime.prepare_decode_step(0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runtime.decode_attention(layer_idx, query, key, value)
        graph.replay()
        torch.cuda.synchronize()

        for request_id in range(batch):
            prompt_q, prompt_k, prompt_v = prompts[layer_idx][request_id]
            full_k = torch.cat((prompt_k, keys[layer_idx][request_id]), dim=2)
            full_v = torch.cat((prompt_v, values[layer_idx][request_id]), dim=2)
            expected, expected_pages = _reference_attention(
                runtime, layer_idx, queries[layer_idx][request_id], full_k, full_v)
            torch.testing.assert_close(
                runtime.decode_output[layer_idx, request_id],
                expected, atol=3e-3, rtol=3e-3)
            if layer_idx >= runtime.dense_layers:
                for head in range(q_heads):
                    pages = expected_pages[head]
                    current = (length) // runtime.page_size
                    assert current in pages.tolist()
                    assert pages.numel() == runtime.page_budget
                    torch.testing.assert_close(
                        runtime.selected_pages[layer_idx, request_id, head],
                        pages, rtol=0, atol=0)

        next_queries = [torch.randn_like(row) for row in queries[layer_idx]]
        next_keys = [torch.randn_like(row) for row in keys[layer_idx]]
        next_values = [torch.randn_like(row) for row in values[layer_idx]]
        query.copy_(torch.cat(next_queries, dim=0))
        key.copy_(torch.cat(next_keys, dim=0))
        value.copy_(torch.cat(next_values, dim=0))
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
            expected, expected_pages = _reference_attention(
                runtime, layer_idx, next_queries[request_id], full_k, full_v)
            torch.testing.assert_close(
                runtime.decode_output[layer_idx, request_id],
                expected, atol=3e-3, rtol=3e-3)
            if layer_idx >= runtime.dense_layers:
                for head in range(q_heads):
                    assert (length + 1) // runtime.page_size in (
                        expected_pages[head].tolist())
                    torch.testing.assert_close(
                        runtime.selected_pages[layer_idx, request_id, head],
                        expected_pages[head], rtol=0, atol=0)
