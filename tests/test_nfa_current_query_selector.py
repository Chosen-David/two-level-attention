"""单层 Req-Batched selector 的当前 query、区域与 Graph 回归门禁。"""
from types import SimpleNamespace

import pytest
import torch

from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.indexer.nfa_indexer import NFAIndexer
from sparse_attn.indexer.nfa_layer_batch_selector import (
    mark_first_decode_seeded, select_topk_for_layer_batched)
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.policies.region_geometry import RegionGeometry


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

PAGE = 64
HEAD_DIM = 128
GEOMETRY = RegionGeometry(
    sink=4, near=128, window=64, page=PAGE, max_new_tokens=8)


def _indexer(k1=4, near_pages=2):
    args = nfa_args()
    args.tia_level1_topk = k1
    args.tia_level2_topk = 128
    args.nfa_near_pages = near_pages
    args.nfa_swa_size = GEOMETRY.window
    args.nfa_near_len = GEOMETRY.near
    return NFAIndexer(args)


def _make_cache(req_ids, lengths, skip_states, *, seeds=None, k1=4,
                near_pages=2):
    cache = GraphReadyPagedRegionCache(
        1, GEOMETRY, max_requests=8, max_physical_pages=64, max_blocks=16,
        max_visible_len=640, max_query_tokens=8, num_kv_heads=1,
        num_query_heads=2, head_dim_k=HEAD_DIM, head_dim_v=HEAD_DIM,
        dtype=torch.bfloat16, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(20261102)
    for request_id, length, (skip_far, skip_near) in zip(
            req_ids, lengths, skip_states):
        cache.add_request(request_id)
        keys = torch.randn(
            1, 1, length, HEAD_DIM, generator=generator,
            dtype=torch.bfloat16, device='cuda')
        values = torch.randn_like(keys)
        cache.write_prefill(0, request_id, keys, values)
        layer_seeds = None if seeds is None else {0: seeds[request_id]}
        cache.freeze(
            request_id, [skip_far], [skip_near], layer_seeds)
    attention = SimpleNamespace(indexer=_indexer(k1, near_pages), scaling=1.0)
    return cache, attention, generator


def _plan_and_write(cache, req_ids, generator):
    cache.new_step(req_ids)
    cache.plan_common(req_ids)
    batch = len(req_ids)
    keys = torch.randn(
        batch, 1, 1, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    values = torch.randn_like(keys)
    cache.write_decode(
        0, keys, values, cache.slot_mapping_for_layer(0)[:batch])


def _logical_ids(cache, request_id, row):
    count = int(cache.visible_lens_for_layer(0)[row])
    slots = cache.gather_indices_for_layer(0)[row, :count].cpu().tolist()
    inverse = {
        physical: logical
        for logical, physical in cache.states[request_id].block_tables[0].items()}
    return sorted(inverse[slot // PAGE] * PAGE + slot % PAGE for slot in slots)


def _counterfactual_queries():
    current = torch.zeros(
        1, 2, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    previous = torch.zeros_like(current)
    current[..., 48] = 1
    previous[..., 49] = 1
    return current, previous


def _make_counterfactual_cache():
    cache, attention, generator = _make_cache(
        [2], [448], [(False, False)], k1=1, near_pages=0)
    # 用两组独立尾维构造确定性的远端块偏好：当前 Q 选块 1，上一帧 Q 选块 3。
    request_id = 2
    state = cache.states[request_id]
    keys = torch.zeros(
        1, 1, 448, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    keys[:, :, 64:128, 48] = 8
    keys[:, :, 192:256, 49] = 8
    values = torch.zeros_like(keys)
    # 重新建立请求，避免依赖内部页地址来覆写 prefill。
    cache.remove_request(request_id)
    cache.add_request(request_id)
    cache.write_prefill(0, request_id, keys, values)
    cache.freeze(request_id, [False], [False])
    _plan_and_write(cache, [request_id], generator)
    return cache, attention, state


def test_current_query_not_previous_frame_query():
    cache, attention, _ = _make_counterfactual_cache()
    current, previous = _counterfactual_queries()

    select_topk_for_layer_batched(
        cache, 0, [2], current, attention=attention)
    current_ids = _logical_ids(cache, 2, 0)
    assert sum(64 <= token < 128 for token in current_ids) == 60
    assert not any(192 <= token < 256 for token in current_ids)

    select_topk_for_layer_batched(
        cache, 0, [2], previous, attention=attention)
    previous_ids = _logical_ids(cache, 2, 0)
    assert sum(192 <= token < 256 for token in previous_ids) == 60
    assert not any(64 <= token < 128 for token in previous_ids)
    assert current_ids != previous_ids


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_batched_rows_match_isolated_noncontiguous_requests_and_skip_states(batch):
    req_ids = [1, 3, 6, 7][:batch]
    lengths = [448, 449, 510, 513][:batch]
    skip_states = [
        (False, False), (True, False), (False, True), (True, True)][:batch]
    cache, attention, generator = _make_cache(req_ids, lengths, skip_states)
    _plan_and_write(cache, req_ids, generator)
    query = torch.randn(
        batch, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')

    expected = {}
    for row, request_id in enumerate(req_ids):
        cache.new_step([request_id])
        cache.plan_common([request_id])
        select_topk_for_layer_batched(
            cache, 0, [request_id], query[row:row+1],
            attention=attention)
        expected[request_id] = _logical_ids(cache, request_id, 0)

    cache.new_step(req_ids)
    cache.plan_common(req_ids)
    select_topk_for_layer_batched(
        cache, 0, req_ids, query, attention=attention)
    for row, (request_id, length, (skip_far, skip_near)) in enumerate(
            zip(req_ids, lengths, skip_states)):
        actual = _logical_ids(cache, request_id, row)
        assert actual == expected[request_id]
        sink_hi, far_hi, near_hi = GEOMETRY.bounds(length+1)
        if skip_far:
            assert not any(sink_hi <= token < far_hi for token in actual)
        if skip_near:
            assert not any(far_hi <= token < near_hi for token in actual)
        if skip_far and skip_near:
            assert len(actual) == GEOMETRY.sink + GEOMETRY.window


def test_first_decode_seed_union_is_deduplicated_and_consumed_once():
    req_ids = [1, 3]
    seeds = {
        1: [2, 64, 65, 191],       # 2 与 sink 重叠，其余位于被跳过的 far。
        3: [3, 192, 300, 449],    # 3/449 与保护集重叠，300 位于 near。
    }
    cache, attention, generator = _make_cache(
        req_ids, [448, 449], [(True, True), (True, True)], seeds=seeds)
    _plan_and_write(cache, req_ids, generator)
    query = torch.randn(
        2, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    select_topk_for_layer_batched(
        cache, 0, req_ids, query, attention=attention)

    first = {rid: _logical_ids(cache, rid, row)
             for row, rid in enumerate(req_ids)}
    assert set(seeds[1]) <= set(first[1])
    assert set(seeds[3]) <= set(first[3])
    assert len(first[1]) == GEOMETRY.sink + GEOMETRY.window + 3
    assert len(first[3]) == GEOMETRY.sink + GEOMETRY.window + 2

    # 与 GraphRunner 一致：整步所有层完成后才统一消费 seed。
    global_free_before = len(cache.layers[0].free_ids)
    seeded_totals = {
        rid: (len(cache.states[rid].block_tables[0])
              + len(cache.states[rid].free_phys_slots_by_layer[0]))
        for rid in req_ids
    }
    mark_first_decode_seeded(cache, req_ids)
    cache.commit_step()
    torch.cuda.synchronize()
    cache._reclaim_pending_free()
    # 首步结束后 seed-only 页及重复计算的临时余量必须回到全局池。
    expected_release = sum(
        seeded_totals[rid]-cache.states[rid].max_resident_pages_by_layer[0]
        for rid in req_ids)
    assert expected_release >= 4
    assert len(cache.layers[0].free_ids) == global_free_before + expected_release
    for rid in req_ids:
        state = cache.states[rid]
        total = (len(state.block_tables[0])
                 + len(state.free_phys_slots_by_layer[0]))
        assert total == state.max_resident_pages_by_layer[0]
        assert total < seeded_totals[rid]
    _plan_and_write(cache, req_ids, generator)
    select_topk_for_layer_batched(
        cache, 0, req_ids, query, attention=attention)
    for row, request_id in enumerate(req_ids):
        steady = _logical_ids(cache, request_id, row)
        assert len(steady) == GEOMETRY.sink + GEOMETRY.window
        assert not (set(seeds[request_id])-set(range(4))) & set(steady) & {
            64, 65, 191, 192, 300}
        assert cache.states[request_id].first_decode_seeded
    assert cache.first_decode_seeded_buf[req_ids].tolist() == [1, 1]


def test_selector_capture_replay_uses_live_current_query():
    cache, attention, _ = _make_counterfactual_cache()
    query, previous = _counterfactual_queries()

    def run():
        select_topk_for_layer_batched(
            cache, 0, [2], query, attention=attention)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    current_ids = _logical_ids(cache, 2, 0)
    assert sum(64 <= token < 128 for token in current_ids) == 60

    query.copy_(previous)
    graph.replay()
    replay_ids = _logical_ids(cache, 2, 0)
    run()
    eager_ids = _logical_ids(cache, 2, 0)
    assert replay_ids == eager_ids
    assert sum(192 <= token < 256 for token in replay_ids) == 60
    assert not any(64 <= token < 128 for token in replay_ids)
