"""单层 Req-Batched selector 的当前 query、区域与 Graph 回归门禁。"""
from types import SimpleNamespace

import pytest
import torch
import triton

import sparse_attn.indexer.nfa_layer_batch_selector as selector_ops
from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.indexer.nfa_indexer import NFAIndexer
from sparse_attn.indexer.nfa_layer_batch_selector import select_topk_for_layer_batched
from sparse_attn.ops.paged_region_kernels import sparse_paged_attention_gather
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


def _make_cache(req_ids, lengths, skip_states, *, k1=4,
                near_pages=2, num_kv_heads=1, max_physical_pages=64,
                max_blocks=16, max_visible_len=640):
    cache = GraphReadyPagedRegionCache(
        1, GEOMETRY, max_requests=8, max_physical_pages=max_physical_pages,
        max_blocks=max_blocks,
        max_visible_len=max_visible_len, max_query_tokens=8, num_kv_heads=num_kv_heads,
        num_query_heads=2 * num_kv_heads, head_dim_k=HEAD_DIM, head_dim_v=HEAD_DIM,
        dtype=torch.bfloat16, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(20261102)
    for request_id, length, (skip_far, skip_near) in zip(
            req_ids, lengths, skip_states):
        cache.add_request(request_id)
        keys = torch.randn(
            1, num_kv_heads, length, HEAD_DIM, generator=generator,
            dtype=torch.bfloat16, device='cuda')
        values = torch.randn_like(keys)
        cache.write_prefill(0, request_id, keys, values)
        cache.freeze(request_id, [skip_far], [skip_near])
    attention = SimpleNamespace(indexer=_indexer(k1, near_pages), scaling=1.0)
    return cache, attention, generator


def _plan_and_write(cache, req_ids, generator):
    cache.new_step(req_ids)
    cache.plan_common(req_ids)
    batch = len(req_ids)
    keys = torch.randn(
        batch, cache.num_kv_heads, 1, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    values = torch.randn_like(keys)
    cache.write_decode(
        0, keys, values, cache.slot_mapping_for_layer(0)[:batch])


def _logical_ids(cache, request_id, row, kv_head=0):
    count = int(cache.visible_lens_for_layer(0)[row, kv_head])
    slots = cache.gather_indices_for_layer(0)[row, kv_head, :count].cpu().tolist()
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


def test_owner_generation_wrap_rebuilds_int32_dedup_table():
    wrapped, attention, _ = _make_counterfactual_cache()
    reference, reference_attention, _ = _make_counterfactual_cache()
    query, _ = _counterfactual_queries()
    select_topk_for_layer_batched(
        wrapped, 0, [2], query, attention=attention)
    select_topk_for_layer_batched(
        reference, 0, [2], query, attention=reference_attention)

    workspace = wrapped._nfa_selector_workspace

    def buffer(name):
        return next(value for key, value in workspace.items()
                    if isinstance(key, tuple) and key[0] == name)

    candidate = buffer('candidate_ids')
    owners = buffer('fine_owners')
    generations = buffer('fine_owner_generations')
    assert owners.dtype == torch.int32
    assert generations.dtype == torch.int32
    owners.fill_(2147483647)
    generations.fill_(2147483647 // candidate.shape[-1])

    select_topk_for_layer_batched(
        wrapped, 0, [2], query, attention=attention)
    select_topk_for_layer_batched(
        reference, 0, [2], query, attention=reference_attention)
    assert _logical_ids(wrapped, 2, 0) == _logical_ids(reference, 2, 0)
    assert generations.tolist() == [[1]]


@pytest.mark.parametrize('batch', [1, 2, 4])
@pytest.mark.parametrize('num_kv_heads', [1, 2])
def test_batched_rows_match_isolated_noncontiguous_requests_and_skip_states(
        batch, num_kv_heads):
    req_ids = [1, 3, 6, 7][:batch]
    lengths = [448, 449, 510, 513][:batch]
    skip_states = [
        (False, False), (True, False), (False, True), (True, True)][:batch]
    cache, attention, generator = _make_cache(
        req_ids, lengths, skip_states, num_kv_heads=num_kv_heads)
    _plan_and_write(cache, req_ids, generator)
    query = torch.randn(
        batch, 2 * num_kv_heads, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')

    expected = {}
    for row, request_id in enumerate(req_ids):
        cache.new_step([request_id])
        cache.plan_common([request_id])
        select_topk_for_layer_batched(
            cache, 0, [request_id], query[row:row+1],
            attention=attention)
        expected[request_id] = [
            _logical_ids(cache, request_id, 0, kv_head=kv)
            for kv in range(num_kv_heads)]

    cache.new_step(req_ids)
    cache.plan_common(req_ids)
    select_topk_for_layer_batched(
        cache, 0, req_ids, query, attention=attention)
    for row, (request_id, length, (skip_far, skip_near)) in enumerate(
            zip(req_ids, lengths, skip_states)):
        actual = [
            _logical_ids(cache, request_id, row, kv_head=kv)
            for kv in range(num_kv_heads)]
        assert actual == expected[request_id]
        protected, selectable, excluded = _strict_region_oracle(
            length + 1, skip_far, skip_near)
        for group in actual:
            attended = set(group)
            assert len(group) == len(attended)
            assert protected <= attended
            assert attended <= protected | selectable
            assert not attended & excluded
            assert all(0 <= token < length + 1 for token in attended)
        if skip_far and skip_near:
            assert all(set(group) == protected for group in actual)


def _strict_region_oracle(length, skip_far, skip_near):
    """独立计算 P/R/X；不复用 cache 的区间或 selector 的 union 实现。"""
    sink_hi = min(GEOMETRY.sink, length)
    near_hi = max(sink_hi, length - GEOMETRY.window)
    far_hi = min(near_hi, max(
        sink_hi,
        ((length - GEOMETRY.window - GEOMETRY.near) // PAGE) * PAGE))
    protected = set(range(sink_hi)) | set(range(near_hi, length))
    selectable = set(range(sink_hi, far_hi)) | set(range(far_hi, near_hi))
    excluded = set()
    if skip_far:
        excluded.update(range(sink_hi, far_hi))
    if skip_near:
        excluded.update(range(far_hi, near_hi))
    return protected, selectable, excluded


def _strict_both_skip_oracle(length):
    protected, _, excluded = _strict_region_oracle(length, True, True)
    return protected, excluded


def test_first_sparse_forward_both_skip_matches_independent_set_and_addresses():
    req_ids = [1, 3]
    cache, attention, generator = _make_cache(
        req_ids, [448, 449], [(True, True), (True, True)])
    query = torch.randn(
        2, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')

    # 第一稀疏 forward 与后续稳态必须使用同一 strict-skip 集合合同。
    for step in range(2):
        _plan_and_write(cache, req_ids, generator)
        select_topk_for_layer_batched(
            cache, 0, req_ids, query, attention=attention)
        actual_attention = sparse_paged_attention_gather(
            cache.layers[0], query, cache.gather_indices_for_layer(0),
            cache.visible_lens_for_layer(0),
            torch.arange(len(req_ids), dtype=torch.int32, device='cuda'),
            scale=1.0, split_kv=False)
        for row, request_id in enumerate(req_ids):
            length = cache.states[request_id].logical_length + 1
            protected, excluded = _strict_both_skip_oracle(length)
            actual = _logical_ids(cache, request_id, row)
            assert len(actual) == len(set(actual))
            assert set(actual) == protected
            assert not set(actual) & excluded

            # 每个最终 gather 物理槽必须反解到同一请求的有效逻辑 token。
            count = int(cache.visible_lens_for_layer(0)[row, 0])
            slots = cache.gather_indices_for_layer(0)[row, 0, :count].cpu().tolist()
            inverse = {
                physical: logical
                for logical, physical in cache.states[request_id].block_tables[0].items()}
            assert len(slots) == len(protected)
            for slot in slots:
                assert slot // PAGE in inverse
                logical = inverse[slot // PAGE] * PAGE + slot % PAGE
                assert logical in protected
                assert 0 <= logical < length

            # attention reference 直接使用独立 P 集合构造物理地址，不读 gather。
            table = cache.states[request_id].block_tables[0]
            reference_slots = torch.tensor(
                [table[token // PAGE] * PAGE + token % PAGE
                 for token in sorted(protected)], device='cuda')
            phys, offset = reference_slots // PAGE, reference_slots % PAGE
            k = cache.layers[0].kv_pool_k[phys, 0, 0, offset].float()
            v = cache.layers[0].kv_pool_v[phys, 0, 0, offset].float()
            probabilities = torch.softmax(k @ query[row, :2].float().T, dim=0)
            expected_attention = probabilities.T @ v
            torch.testing.assert_close(
                actual_attention[row], expected_attention.to(actual_attention.dtype),
                atol=.03, rtol=.03)
        cache.commit_step()
        torch.cuda.synchronize()


def test_near_skip_reentry_incremental_stats_match_rebuild_and_slot_reuse_invalidates():
    incremental, attention, incremental_generator = _make_cache(
        [2], [504], [(False, True)], k1=2, near_pages=0)
    reference, reference_attention, reference_generator = _make_cache(
        [2], [504], [(False, True)], k1=2, near_pages=0)
    generator = torch.Generator(device='cuda').manual_seed(20261103)
    queries = torch.randn(
        8, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')

    for step in range(8):
        _plan_and_write(incremental, [2], incremental_generator)
        _plan_and_write(reference, [2], reference_generator)
        if step == 7:
            # 长度 512 时逻辑页 4 从 near 进入 far。统计内容不依赖区域归属，
            # 应继续走增量路径，并由下方全量重建 reference 验证数值等价。
            assert int(incremental._nfa_page_stats_lengths[0, 2]) == 511
        if hasattr(reference, '_nfa_page_stats_lengths'):
            reference._nfa_page_stats_lengths.zero_()
        select_topk_for_layer_batched(
            incremental, 0, [2], queries[step:step+1], attention=attention)
        select_topk_for_layer_batched(
            reference, 0, [2], queries[step:step+1],
            attention=reference_attention)
        assert _logical_ids(incremental, 2, 0) == _logical_ids(reference, 2, 0)
        incremental.commit_step()
        reference.commit_step()

    torch.testing.assert_close(
        incremental.block_table_buf, reference.block_table_buf, rtol=0, atol=0)
    logical_length = incremental.states[2].logical_length
    for logical_page, physical_page in incremental.states[2].block_tables[0].items():
        valid = min(PAGE, logical_length - logical_page * PAGE)
        torch.testing.assert_close(
            incremental.layers[0].kv_pool_k[physical_page, 0, :, :valid],
            reference.layers[0].kv_pool_k[physical_page, 0, :, :valid],
            rtol=0, atol=0)
    incremental_buffers = next(iter(
        incremental._nfa_page_stats_persistent.values()))
    reference_buffers = next(iter(
        reference._nfa_page_stats_persistent.values()))
    # 全量 KV 驻留下，对请求的全部逻辑页比较增量与全量重建统计。
    for logical_page in incremental.states[2].block_tables[0]:
        for actual, expected in zip(incremental_buffers, reference_buffers):
            torch.testing.assert_close(
                actual[:, 2:3, logical_page], expected[:, 2:3, logical_page],
                rtol=0, atol=0)

    incremental.remove_request(2)
    torch.cuda.synchronize()
    incremental.add_request(2)
    keys = torch.randn(
        1, 1, 448, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    incremental.write_prefill(0, 2, keys, torch.randn_like(keys))
    incremental.freeze(2, [False], [True])
    assert int(incremental._nfa_page_stats_lengths[0, 2]) == 0


def test_fixed_stats_grid_owner_writes_and_refreshes_latest_block():
    # 80 个逻辑块对应 5 个统计块组，覆盖固定 4 路 grid 不启动绝对
    # owner=4 的情形；第二步直接从当前 KV 计算最新块 min/max 参考。
    cache, attention, generator = _make_cache(
        [2], [4500], [(False, False)], k1=2, near_pages=0,
        max_blocks=80, max_physical_pages=100)
    query = torch.randn(
        1, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')

    for step in range(2):
        cache.new_step([2])
        cache.plan_common([2])
        keys = torch.randn(
            1, 1, 1, HEAD_DIM, generator=generator,
            dtype=torch.bfloat16, device='cuda')
        values = torch.randn_like(keys)
        decode_write = (keys, values, cache.slot_mapping_for_layer(0)[:1])

        select_topk_for_layer_batched(
            cache, 0, [2], query, attention=attention,
            decode_write=decode_write)
        logical_length = int(cache.planned_lengths_buf[0])
        last_blk = (logical_length - 1) // PAGE
        valid = logical_length - last_blk * PAGE
        physical = cache.states[2].block_tables[0][last_blk]
        dims = attention.indexer._subspace_indices(
            query.device).long()
        current_k = cache.layers[0].kv_pool_k[
            physical, 0, 0, :valid, :][:, dims].float()
        # 生产 kernel 对末块无效 lane 以 0 参与 min/max/sum，而非只归约
        # valid 前缀；参考必须复现这一既有数值契约。
        padded_k = current_k.new_zeros((PAGE, dims.numel()))
        padded_k[:valid] = current_k
        buffers = next(iter(cache._nfa_page_stats_persistent.values()))
        torch.testing.assert_close(
            buffers[0][0, 2, last_blk, 0],
            padded_k.min(dim=0).values.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(
            buffers[1][0, 2, last_blk, 0],
            padded_k.max(dim=0).values.to(torch.bfloat16), rtol=0, atol=0)
        if decode_write is not None:
            slot = int(decode_write[2][0])
            physical, offset = slot // PAGE, slot % PAGE
            torch.testing.assert_close(
                cache.layers[0].kv_pool_k[physical, 0, 0, offset],
                decode_write[0][0, 0, 0], rtol=0, atol=0)
            torch.testing.assert_close(
                cache.layers[0].kv_pool_v[physical, 0, 0, offset],
                decode_write[1][0, 0, 0], rtol=0, atol=0)
        if step == 0:
            cache.commit_step()


def test_both_skip_light_path_replay_writes_decode_kv():
    cache, attention, generator = _make_cache(
        [2], [448], [(True, True)])
    _plan_and_write(cache, [2], generator)
    query = torch.randn(
        1, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    first_keys = torch.randn(
        1, 1, 1, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    first_values = torch.randn_like(first_keys)
    select_topk_for_layer_batched(
        cache, 0, [2], query, attention=attention,
        decode_write=(first_keys, first_values,
                      cache.slot_mapping_for_layer(0)[:1]))
    assert set(_logical_ids(cache, 2, 0)) == _strict_both_skip_oracle(449)[0]

    call_plan = cache._nfa_selector_workspace[
        ('eager_selector_call_plan', 0)]
    assert [item[0][0] for item in call_plan[1]] == ['final_union']
    cache.commit_step()

    # 第二次调用走固定 eager launch 计划；轻路径没有 fused stats/Q kernel，
    # 因此必须在 replay 前显式写入本步 K/V。
    cache.new_step([2])
    cache.plan_common([2])
    keys = torch.randn(
        1, 1, 1, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    values = torch.randn_like(keys)
    slots = cache.slot_mapping_for_layer(0)[:1]
    select_topk_for_layer_batched(
        cache, 0, [2], query, attention=attention,
        decode_write=(keys, values, slots))
    slot = int(slots[0])
    physical, offset = slot // PAGE, slot % PAGE
    torch.testing.assert_close(
        cache.layers[0].kv_pool_k[physical, 0, 0, offset], keys[0, 0, 0])
    torch.testing.assert_close(
        cache.layers[0].kv_pool_v[physical, 0, 0, offset], values[0, 0, 0])
    steady = _logical_ids(cache, 2, 0)
    assert set(steady) == _strict_both_skip_oracle(450)[0]


def test_both_skip_light_path_real_gqa_geometry():
    # 精确模拟 Qwen3-8B Graph capture 的 padding 行：36 层、8 个 KV group、
    # 32K 页表容量与 8K visible bucket；大 N_PAD 尾部 lane 不得负索引 gather。
    cache = GraphReadyPagedRegionCache(
        36, GEOMETRY, max_requests=1, max_physical_pages=8, max_blocks=521,
        max_visible_len=8192, max_query_tokens=1, num_kv_heads=8,
        num_query_heads=32, head_dim_k=HEAD_DIM, head_dim_v=HEAD_DIM,
        dtype=torch.bfloat16, device='cuda')
    attention = SimpleNamespace(indexer=_indexer(), scaling=1.0)
    cache.new_step([-1])
    cache.plan_common([-1])
    query = torch.randn(
        1, 32, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    select_topk_for_layer_batched(
        cache, 0, [-1], query, attention=attention)
    torch.cuda.synchronize()
    assert int(cache.visible_lens_for_layer(0)[0, 0]) >= 0


def test_both_skip_light_path_capture_replay_writes_live_decode_kv():
    cache, attention, generator = _make_cache(
        [2], [448], [(True, True)])
    _plan_and_write(cache, [2], generator)
    query = torch.randn(
        1, 2, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    keys = torch.randn(
        1, 1, 1, HEAD_DIM, generator=generator,
        dtype=torch.bfloat16, device='cuda')
    values = torch.randn_like(keys)
    slots = cache.slot_mapping_for_layer(0)[:1].clone()

    def run():
        select_topk_for_layer_batched(
            cache, 0, [2], query, attention=attention,
            decode_write=(keys, values, slots))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    assert set(_logical_ids(cache, 2, 0)) == _strict_both_skip_oracle(449)[0]
    cache.commit_step()

    cache.new_step([2])
    cache.plan_common([2])
    keys.normal_(generator=generator)
    values.normal_(generator=generator)
    slots.copy_(cache.slot_mapping_for_layer(0)[:1])
    graph.replay()
    slot = int(slots[0])
    physical, offset = slot // PAGE, slot % PAGE
    torch.testing.assert_close(
        cache.layers[0].kv_pool_k[physical, 0, 0, offset], keys[0, 0, 0])
    torch.testing.assert_close(
        cache.layers[0].kv_pool_v[physical, 0, 0, offset], values[0, 0, 0])
    replay_ids = _logical_ids(cache, 2, 0)
    run()
    assert replay_ids == _logical_ids(cache, 2, 0)
    assert set(replay_ids) == _strict_both_skip_oracle(450)[0]


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


def test_padded_capture_does_not_specialize_non_skip_replay():
    cache, attention, _ = _make_counterfactual_cache()
    query, _ = _counterfactual_queries()

    # GraphRunner 用全 padding 行 capture；这不代表 replay 请求会 both-skip。
    cache.new_step([-1])
    cache.plan_common([-1])
    select_topk_for_layer_batched(
        cache, 0, [-1], query, attention=attention)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_topk_for_layer_batched(
            cache, 0, [-1], query, attention=attention)

    cache.new_step([2])
    cache.plan_common([2])
    graph.replay()
    replay_ids = _logical_ids(cache, 2, 0)
    select_topk_for_layer_batched(
        cache, 0, [2], query, attention=attention)
    eager_ids = _logical_ids(cache, 2, 0)
    assert replay_ids == eager_ids
    assert sum(64 <= token < 128 for token in replay_ids) == 60


@pytest.mark.parametrize('split_kv', [False, True])
def test_per_kv_group_selection_and_attention_consumer(split_kv):
    cache, attention, generator = _make_cache(
        [2], [448], [(False, False)], k1=1, near_pages=0, num_kv_heads=2)
    request_id = 2
    cache.remove_request(request_id)
    cache.add_request(request_id)
    keys = torch.zeros(
        1, 2, 448, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    # 两个 KV group 的重要远端块不同，head-0 共享选集必然漏掉其中一组。
    keys[0, 0, 64:128, 48] = 8
    keys[0, 1, 192:256, 49] = 8
    values = torch.randn(
        keys.shape, generator=generator, dtype=torch.bfloat16, device='cuda')
    cache.write_prefill(0, request_id, keys, values)
    cache.freeze(request_id, [False], [False])
    _plan_and_write(cache, [request_id], generator)

    query = torch.zeros(
        1, 4, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    query[0, :2, 48] = 1
    query[0, 2:, 49] = 1
    select_topk_for_layer_batched(
        cache, 0, [request_id], query, attention=attention)

    group0 = _logical_ids(cache, request_id, 0, kv_head=0)
    group1 = _logical_ids(cache, request_id, 0, kv_head=1)
    assert sum(64 <= token < 128 for token in group0) == 60
    assert not any(192 <= token < 256 for token in group0)
    assert sum(192 <= token < 256 for token in group1) == 60
    assert not any(64 <= token < 128 for token in group1)

    gather = cache.gather_indices_for_layer(0)
    lens = cache.visible_lens_for_layer(0)
    actual = sparse_paged_attention_gather(
        cache.layers[0], query, gather, lens,
        torch.zeros(1, dtype=torch.int32, device='cuda'), scale=1.0,
        split_kv=split_kv)
    expected = torch.empty_like(actual)
    for kv in range(2):
        count = int(lens[0, kv])
        slots = gather[0, kv, :count].long()
        phys, offset = slots // PAGE, slots % PAGE
        k = cache.layers[0].kv_pool_k[phys, 0, kv, offset].float()
        v = cache.layers[0].kv_pool_v[phys, 0, kv, offset].float()
        q_group = query[0, kv * 2:(kv + 1) * 2].float()
        probabilities = torch.softmax(k @ q_group.T, dim=0)
        expected[0, kv * 2:(kv + 1) * 2] = probabilities.T @ v
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.parametrize('capacity', [16384, 65728, 131520])
def test_tiled_filtered_topk_bit_exact_with_quota_and_ties(capacity):
    m, hkv, r, width = 3, 2, 8452, 1024
    token_bits = 32 if capacity >= 65536 else 16
    generator = torch.Generator(device='cuda').manual_seed(20261103)
    cand = torch.stack([
        torch.randperm(capacity, generator=generator, device='cuda')[:r]
        for _ in range(m * hkv)
    ]).view(m, hkv, r).to(torch.int64).contiguous()
    logits = torch.randn(
        m, hkv, r, generator=generator, dtype=torch.bfloat16,
        device='cuda').contiguous()
    # 同分、同 token、NaN/Inf、区间外 token 和零 quota 同时覆盖阈值与补洞。
    logits[:, :, 128:512] = 0.25
    logits[:, :, 512:640] = float('-inf')
    logits[:, :, 640:768] = 0.5
    logits[:, :, 768:800] = float('nan')
    logits[:, :, 800:808] = float('inf')
    cand[:, :, 640:768] = capacity - 1
    lower = torch.tensor([128, 0, 0], dtype=torch.int64, device='cuda')
    upper = torch.tensor([8000, 64, 4096], dtype=torch.int64, device='cuda')
    quota = torch.tensor([1024, 300, 0], dtype=torch.int64, device='cuda')

    reference_out = torch.empty(
        m, hkv, width, dtype=torch.int32, device='cuda')
    reference_count = torch.empty(m, hkv, dtype=torch.int32, device='cuda')
    selector_ops._nfa_filtered_topk_kernel[(m * hkv,)](
        logits, cand, lower, upper, quota, reference_out, reference_count, r,
        R_PAD=triton.next_power_of_2(max(r, width)), S=capacity, K=width,
        SINK=4, META_GROUP=hkv, TOKEN_BITS=token_bits, num_warps=8)

    # 模拟 fine-score kernel 的 packed-key 输出，与 mono reference 逐位同构。
    bits = logits.view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    ordered = torch.where(negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
    score = logits.float()
    good = (
        (cand >= lower[:, None, None]) & (cand < upper[:, None, None])
        & (quota[:, None, None] > 0) & (score == score)
        & (score != float('-inf')))
    dtype = torch.int64 if token_bits == 32 else torch.int32
    delta = capacity - cand.to(dtype)
    invalid = -32768 << token_bits
    packed = torch.where(
        good, (ordered.to(dtype) << token_bits) + delta,
        torch.full((), invalid, dtype=dtype, device='cuda')).contiguous()
    actual = selector_ops._nfa_filtered_topk(
        packed, cand, lower, upper, quota, capacity, width, sink=4)
    torch.testing.assert_close(actual.indices, reference_out, rtol=0, atol=0)
    torch.testing.assert_close(actual.counts, reference_count, rtol=0, atol=0)
