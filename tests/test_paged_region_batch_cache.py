import torch
from sparse_attn.cache.paged_region_cache import (
    PagedRegionLayer, PagedRegionCache, SINK, FAR, NEAR, SWA)
from sparse_attn.ops.region_geometry import RegionGeometry


@torch.inference_mode()
def test_global_physical_pool_and_per_request_block_tables():
    torch.manual_seed(7)
    g = RegionGeometry(sink=2, near=16, window=4, page=4)
    cache = PagedRegionCache(2, g)
    r0 = cache.add_request()
    r1 = cache.add_request()
    assert r0 == 0 and r1 == 1

    for request_id, length in ((r0, 10), (r1, 13)):
        cache.set_active_request(request_id)
        k = torch.randn(1, 2, length, 8)
        v = torch.randn(1, 2, length, 6)
        cache.write_request(0, k, v)
        cache.write_request(1, k, v)

    for layer_idx in (0, 1):
        layer = cache.layers[layer_idx]
        assert set(layer.block_tables[r0].values()).isdisjoint(
            set(layer.block_tables[r1].values()))
        assert len(layer.kv_pool) == sum(
            len(layer.block_tables[r]) for r in (r0, r1))
        assert layer.seq_lens == [10, 13]
        assert layer.block_tables[r0] == {0: 0, 1: 1, 2: 2}
        assert layer.block_tables[r1] == {0: 3, 1: 4, 2: 5, 3: 6}

    # Same logical page maps to a different physical page in each request.
    assert cache.layers[0].block_tables[r0][0] != cache.layers[0].block_tables[r1][0]


@torch.inference_mode()
def test_per_request_freeze_reclaim_uses_monotonic_cursor():
    torch.manual_seed(19)
    g = RegionGeometry(sink=2, near=8, window=4, page=4, max_new_tokens=40)
    layer = PagedRegionLayer(g)
    r = layer.add_request()
    k = torch.randn(1, 2, 40, 8)
    v = torch.randn(1, 2, 40, 8)
    layer.write_request(r, k, v)
    layer.freeze_request(r, skip_near=False, skip_far=True)

    assert layer.reclaim_cursors[r] >= (g.sink + g.page - 1) // g.page
    assert layer.reclaimed_pages > 0
    assert layer.report(r)['reclaimed_pages'] == layer.reclaimed_pages
    cursor = layer.reclaim_cursors[r]

    for step in range(20):
        old_cursor = layer.reclaim_cursors[r]
        nk = torch.randn(1, 2, 1, 8)
        nv = torch.randn(1, 2, 1, 8)
        layer.write_request(r, nk, nv)
        assert layer.reclaim_cursors[r] >= old_cursor
        assert layer.report(r)['aging_copy_bytes'] == 0

    assert layer.reclaim_cursors[r] >= cursor


@torch.inference_mode()
def test_decode_slot_mapping_and_batch_write():
    torch.manual_seed(23)
    g = RegionGeometry(sink=2, near=16, window=4, page=4, max_new_tokens=20)
    layer = PagedRegionLayer(g)
    r0 = layer.add_request()
    r1 = layer.add_request()
    for request_id, length in ((r0, 5), (r1, 9)):
        layer.write_request(request_id, torch.randn(1, 2, length, 8), torch.randn(1, 2, length, 6))
        layer.freeze_request(request_id, skip_near=False, skip_far=False)

    request_ids = [r0, r1]
    for _ in range(6):
        slot_mapping = layer.decode_slot_mapping(request_ids)
        expected = []
        for request_id in request_ids:
            length = layer.seq_lens[request_id]
            lp, offset = divmod(length, g.page)
            expected.append(layer.block_tables[request_id][lp] * g.page + offset)
        assert slot_mapping.tolist() == expected

        keys = torch.randn(2, 2, 1, 8)
        values = torch.randn(2, 2, 1, 6)
        layer.write_decode(request_ids, keys, values, slot_mapping)
        for batch, request_id in enumerate(request_ids):
            length = layer.seq_lens[request_id]
            lp, offset = divmod(length - 1, g.page)
            physical = layer.block_tables[request_id][lp]
            k, v = layer.kv_pool[physical]
            torch.testing.assert_close(k[..., offset:offset + 1, :], keys[batch:batch + 1])
            torch.testing.assert_close(v[..., offset:offset + 1, :], values[batch:batch + 1])

    assert layer.seq_lens == [11, 15]


@torch.inference_mode()
def test_inactive_slot_is_skipped_and_request_exit_recycles_pool():
    torch.manual_seed(29)
    g = RegionGeometry(sink=2, near=16, window=4, page=4, max_new_tokens=20)
    layer = PagedRegionLayer(g)
    r0 = layer.add_request()
    r1 = layer.add_request()
    for request_id, length in ((r0, 5), (r1, 9)):
        layer.write_request(request_id, torch.randn(1, 2, length, 8), torch.randn(1, 2, length, 6))
        layer.freeze_request(request_id, skip_near=False, skip_far=False)

    old_physical = dict(layer.block_tables[r0])
    layer.remove_request(r0)

    sm = layer.decode_slot_mapping([r0, r1])
    length = layer.seq_lens[r1]
    lp, offset = divmod(length, g.page)
    assert sm.tolist() == [-1, layer.block_tables[r1][lp] * g.page + offset]
    keys = torch.randn(2, 2, 1, 8)
    values = torch.randn(2, 2, 1, 6)
    layer.write_decode([r0, r1], keys, values, sm)
    assert layer.seq_lens == [0, 10]

    r2 = layer.add_request()
    assert r2 == 0
    assert layer.active_flags == [True, True]

    layer.write_request(r2, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 6))
    assert set(layer.block_tables[r2].values()).issubset(set(old_physical.values()))


@torch.inference_mode()
def test_stale_slot_mapping_rejected_without_partial_write():
    torch.manual_seed(31)
    g = RegionGeometry(sink=2, near=16, window=4, page=4, max_new_tokens=20)
    layer = PagedRegionLayer(g)
    r = layer.add_request()
    layer.write_request(r, torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 6))
    layer.freeze_request(r, skip_near=False, skip_far=False)
    before = layer.report(r)

    sm = layer.decode_slot_mapping([r])
    bad = sm.clone()
    bad[0] += 1
    try:
        layer.write_decode([r], torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 6), bad)
    except ValueError as exc:
        assert 'stale slot mapping' in str(exc)
    else:
        raise AssertionError('stale slot mapping was accepted')
    assert layer.report(r) == before


@torch.inference_mode()
def test_page_label_visible_mask_and_resident_mask_are_separate():
    torch.manual_seed(37)
    g = RegionGeometry(sink=2, near=8, window=4, page=4, max_new_tokens=20)
    layer = PagedRegionLayer(g)
    r = layer.add_request()
    layer.write_request(r, torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8))

    expected_labels = {0: SINK | FAR, 1: FAR, 2: NEAR, 3: NEAR, 4: SWA}
    assert layer.page_labels(r) == expected_labels

    # Far-skip: mixed SINK|FAR page remains resident for its sink half but
    # far-only pages become reclaimable.
    layer.freeze_request(r, skip_near=False, skip_far=True)
    assert layer.visible_mask(0, r) == SINK
    assert layer.visible_mask(1, r) == 0
    assert layer.visible_mask(2, r) == NEAR
    assert layer.visible_mask(4, r) == SWA
    assert not layer.page_reclaimable(0, r)
    assert layer.page_reclaimable(1, r)
    assert not layer.page_reclaimable(2, r)
    assert not layer.page_reclaimable(4, r)

    # Near-only skip must not claim resident reduction while far remains
    # resident: visible near pages are zero but still resident.
    r_near = layer.add_request()
    layer.write_request(r_near, torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8))
    layer.freeze_request(r_near, skip_near=True, skip_far=False)
    assert layer.visible_mask(2, r_near) == 0
    assert not layer.page_reclaimable(2, r_near)
    assert layer.resident_mask(r_near) & NEAR
