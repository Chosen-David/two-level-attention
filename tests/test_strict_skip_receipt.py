import pytest
import torch

from benchmark.efficiency.benchmark_nfa_end_to_end import _strict_skip_receipt
from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.policies.region_geometry import RegionGeometry


GEOMETRY = RegionGeometry(sink=2, near=8, window=3, page=4, max_new_tokens=2)
PROTECTED = {0, 1, 18, 19, 20}


def _cache(skip_far, skip_near):
    cache = GraphReadyPagedRegionCache(
        1, GEOMETRY, max_requests=1, max_physical_pages=16, max_blocks=8,
        max_visible_len=64, max_query_tokens=1, num_kv_heads=1,
        num_query_heads=1, head_dim_k=4, head_dim_v=4,
        dtype=torch.float32, device='cpu')
    cache.add_request(0)
    keys = torch.randn(1, 1, 20, 4)
    cache.write_prefill(0, 0, keys, keys + 1)
    cache.freeze(0, [skip_far], [skip_near])
    cache.new_step([0])
    cache.plan_common([0])
    return cache


def _receipt(cache, logical_ids):
    table = cache.block_table_buf[0, 0].tolist()
    slots = [table[token // 4] * 4 + token % 4 for token in logical_ids]
    cache.gather_indices_buf[0, 0, 0, :len(slots)] = torch.tensor(
        slots, dtype=torch.int32)
    cache.visible_lens_buf[0, 0, 0] = len(slots)
    return _strict_skip_receipt(
        cache, GEOMETRY, [0], cache.gather_indices_buf,
        cache.visible_lens_buf, planned_length=21)


@pytest.mark.parametrize(
    'skip_far,skip_near,optional',
    [(False, False, {3, 9}), (True, False, {9}),
     (False, True, {3}), (True, True, set())])
def test_first_step_receipt_accepts_all_four_frozen_gate_states(
        skip_far, skip_near, optional):
    expected = sorted(PROTECTED | optional)
    receipt = _receipt(_cache(skip_far, skip_near), expected)
    assert receipt['valid'], receipt['violations']
    assert receipt['planned_length'] == 21
    assert receipt['layers'][0]['logical_ids_by_kv_head'] == [expected]
    assert receipt['layers'][0]['physical_slots_by_kv_head'][0] == [
        token // 4 * 4 + token % 4 for token in expected]


def test_first_step_receipt_rejects_frozen_seed_like_token():
    receipt = _receipt(_cache(True, True), sorted(PROTECTED | {4}))
    assert not receipt['valid']
    assert any('frozen tokens re-entered' in item['message']
               for item in receipt['violations'])


def test_first_step_receipt_rejects_duplicate_and_missing_protected_tokens():
    receipt = _receipt(_cache(True, True), [1, 18, 19, 20, 20])
    messages = [item['message'] for item in receipt['violations']]
    assert not receipt['valid']
    assert 'duplicate physical gather slots' in messages
    assert 'duplicate logical token IDs' in messages
    assert 'protected tokens are missing' in messages


def test_first_step_receipt_rejects_foreign_physical_page():
    cache = _cache(True, True)
    receipt = _receipt(cache, sorted(PROTECTED))
    cache.gather_indices_buf[0, 0, 0, 0] = 15 * 4
    receipt = _strict_skip_receipt(
        cache, GEOMETRY, [0], cache.gather_indices_buf,
        cache.visible_lens_buf, planned_length=21)
    assert not receipt['valid']
    assert any('not owned by request' in item['message']
               for item in receipt['violations'])
