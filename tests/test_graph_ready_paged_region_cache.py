import unittest
import torch

from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.policies.region_geometry import RegionGeometry


class GraphReadyPagedRegionCacheTest(unittest.TestCase):
    def _make_cache(self, n_layers=2, max_new_tokens=20):
        geom = RegionGeometry(sink=2, near=8, window=3, page=4,
                              max_new_tokens=max_new_tokens)
        return GraphReadyPagedRegionCache(
            n_layers, geom, max_requests=4, max_physical_pages=64,
            max_blocks=64, max_visible_len=64, max_query_tokens=16,
            num_kv_heads=2, head_dim_k=4, head_dim_v=4,
            dtype=torch.float32, device='cpu')

    def test_freeze_keeps_prefill_pages_and_decode_appends_physical_pages(self):
        cache = self._make_cache(n_layers=1)
        req = cache.add_request()
        length = 20
        k = torch.randn(1, 2, length, 4)
        v = torch.randn(1, 2, length, 4)
        cache.write_prefill(0, req, k, v)

        cache.freeze(req, skip_far_list=[True], skip_near_list=[True])
        state = cache.states[req]
        initial_table = dict(state.block_tables[0])
        self.assertEqual(list(initial_table), list(range(5)))
        global_free = len(cache.layers[0].free_ids)
        fixed_addresses = (
            cache.layers[0].kv_pool_k.data_ptr(),
            cache.block_table_buf.data_ptr(), cache.slot_mapping_buf.data_ptr())
        written_slots = []

        for step in range(8):
            cache.new_step([req, -1])
            cache.plan_common([req, -1])
            cache.prepare_metadata([req, -1], 0)
            pos = length + step
            logical_page, offset = divmod(pos, cache.geometry.page)
            physical = state.block_tables[0][logical_page]
            self.assertEqual(
                cache.slot_mapping_for_layer(0)[0].item(),
                physical * cache.geometry.page + offset)
            self.assertEqual(
                list(state.block_tables[0]),
                list(range((pos + cache.geometry.page) // cache.geometry.page)))
            for logical, physical_id in initial_table.items():
                self.assertEqual(state.block_tables[0][logical], physical_id)
            dk = torch.full((1, 2, 1, 4), float(step + 1))
            dv = torch.full((1, 2, 1, 4), float(step + 11))
            cache.write_decode(
                0, dk, dv, cache.slot_mapping_for_layer(0)[:1])
            written_slots.append((physical, offset, step + 1))
            cache.commit_step()

        self.assertEqual(state.logical_length, length + 8)
        self.assertEqual(list(state.block_tables[0]), list(range(7)))
        self.assertEqual(len(cache.layers[0].free_ids), global_free - 2)
        self.assertEqual(fixed_addresses, (
            cache.layers[0].kv_pool_k.data_ptr(),
            cache.block_table_buf.data_ptr(), cache.slot_mapping_buf.data_ptr()))
        for physical, offset, value in written_slots:
            self.assertTrue(torch.equal(
                cache.layers[0].kv_pool_k[physical, 0, :, offset],
                torch.full((2, 4), float(value))))

    def test_block_table_h2d_only_when_sequence_enters_new_page(self):
        from sparse_attn.indexer.nfa_layer_batch_selector import (
            mark_first_decode_seeded)

        cache = self._make_cache(n_layers=1)
        req = cache.add_request()
        cache.write_prefill(0, req, torch.randn(1, 2, 20, 4),
                            torch.randn(1, 2, 20, 4))
        cache.freeze(req, [True], [True], {0: [8]})
        cache.new_step([req])
        cache.plan_common([req])
        mark_first_decode_seeded(cache, [req])
        cache.commit_step()

        original_prepare = cache._prepare_common
        full_prepare_calls = []

        def counted_prepare(req_ids, query_lengths):
            full_prepare_calls.append(cache.states[req].logical_length)
            return original_prepare(req_ids, query_lengths)

        cache._prepare_common = counted_prepare
        # 21->22、22->23、23->24 复用页表；24->25 跨页时才做完整 H2D。
        for expected_calls in (0, 0, 0, 1):
            cache.new_step([req])
            cache.plan_common([req])
            self.assertEqual(len(full_prepare_calls), expected_calls)
            self.assertNotEqual(
                cache.block_table_buf[0, req, 2].item(),
                cache.layers[0].PAD_PAGE_ID)
            cache.commit_step()
        self.assertEqual(full_prepare_calls, [24])

    def test_far_seed_consumption_does_not_change_physical_pages(self):
        from sparse_attn.indexer.nfa_layer_batch_selector import (
            mark_first_decode_seeded)

        cache = self._make_cache(n_layers=1)
        req = cache.add_request()
        cache.write_prefill(0, req, torch.randn(1, 2, 40, 4),
                            torch.randn(1, 2, 40, 4))
        cache.freeze(req, [True], [True], {0: [16, 20, 24]})
        state = cache.states[req]
        before = dict(state.block_tables[0])
        self.assertTrue({4, 5, 6}.issubset(before))

        cache.new_step([req])
        cache.plan_common([req])
        # 长度 40 位于页边界，本步正常追加一页；此后的 seed 消费不再改页表。
        before = dict(state.block_tables[0])
        global_free = len(cache.layers[0].free_ids)
        block_table_before = cache.block_table_buf.clone()
        mark_first_decode_seeded(cache, [req])
        cache.commit_step()
        self.assertEqual(state.block_tables[0], before)
        self.assertTrue(torch.equal(cache.block_table_buf, block_table_before))
        self.assertEqual(len(cache.layers[0].free_ids), global_free)

    def test_visible_bucket_uses_unique_skip_policy_groups(self):
        cache = self._make_cache(n_layers=4)
        req = cache.add_request()
        for layer_idx in range(4):
            cache.write_prefill(
                layer_idx, req, torch.randn(1, 2, 20, 4),
                torch.randn(1, 2, 20, 4))
        cache.freeze(
            req, [True, False, True, False], [True, True, True, True])
        original = cache._intervals_for_layer
        calls = []

        def counted(state, layer_idx, length):
            calls.append(layer_idx)
            return original(state, layer_idx, length)

        cache._intervals_for_layer = counted
        cache.choose_visible_bucket([req], (16, 32, 64))
        self.assertEqual(calls, [0, 1])

    def test_metadata_shapes_and_padding(self):
        cache = self._make_cache(n_layers=2)
        req0 = cache.add_request()
        req1 = cache.add_request()
        for req, length in ((req0, 9), (req1, 5)):
            k = torch.randn(1, 2, length, 4)
            v = torch.randn(1, 2, length, 4)
            for layer_idx in range(2):
                cache.write_prefill(layer_idx, req, k, v)
            cache.freeze(req, [False, False], [False, False])

        req_ids = [req0, req1, -1, -1]
        cache.new_step(req_ids)
        for layer_idx in range(2):
            gather, vis = cache.prepare_metadata(req_ids, layer_idx)
            self.assertEqual(gather.shape, (4, 2, 64))
            self.assertEqual(vis.shape, (4, 2))
            self.assertEqual(vis.tolist()[2:], [[0, 0], [0, 0]])

        self.assertEqual(cache.block_table_buf[:, 2].unique().tolist(),
                         [cache.layers[0].PAD_PAGE_ID])
        self.assertEqual(cache.slot_mapping_buf[:, 2].unique().tolist(), [-1])

    def test_prepare_metadata_does_not_advance_logical_length(self):
        cache = self._make_cache(n_layers=1)
        req = cache.add_request()
        cache.write_prefill(0, req, torch.randn(1, 2, 5, 4),
                            torch.randn(1, 2, 5, 4))
        cache.freeze(req, [False], [False])
        before = cache.states[req].logical_length

        cache.new_step([req])
        cache.prepare_metadata([req], 0)
        self.assertEqual(cache.states[req].logical_length, before)

        cache.commit_step()
        self.assertEqual(cache.states[req].logical_length, before + 1)

    def test_decode_capacity_is_prompt_plus_max_new_for_all_skip_policies(self):
        for skip in (False, True):
            with self.subTest(skip=skip):
                cache = self._make_cache(n_layers=1, max_new_tokens=4)
                req = cache.add_request()
                cache.write_prefill(0, req, torch.randn(1, 2, 5, 4),
                                    torch.randn(1, 2, 5, 4))
                cache.freeze(req, [skip], [skip])
                for step in range(4):
                    cache.new_step([req])
                    cache.plan_common([req])
                    cache.write_decode(
                        0, torch.full((1, 2, 1, 4), float(step)),
                        torch.full((1, 2, 1, 4), float(step)),
                        cache.slot_mapping_for_layer(0)[:1])
                    cache.commit_step()
                self.assertEqual(cache.states[req].logical_length, 9)
                self.assertEqual(
                    list(cache.states[req].block_tables[0]), [0, 1, 2])
                cache.new_step([req])
                with self.assertRaisesRegex(RuntimeError, 'request capacity'):
                    cache.plan_common([req])
                cache.remove_request(req)
                self.assertEqual(len(cache.layers[0].free_ids), 64)

    def test_batched_append_only_slot_mapping_for_b1_b2_b4(self):
        lengths = {0: 5, 1: 9, 2: 13, 3: 17}
        request_order = [3, 1, 2, 0]
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                cache = self._make_cache(n_layers=2)
                req_ids = request_order[:batch]
                for request_id in req_ids:
                    cache.add_request(request_id)
                    length = lengths[request_id]
                    for layer_idx in range(2):
                        cache.write_prefill(
                            layer_idx, request_id, torch.randn(1, 2, length, 4),
                            torch.randn(1, 2, length, 4))
                    cache.freeze(
                        request_id,
                        [request_id % 2 == 0, request_id % 2 != 0],
                        [request_id % 3 == 0, True])

                for step in range(3):
                    cache.new_step(req_ids)
                    cache.plan_common(req_ids)
                    for row, request_id in enumerate(req_ids):
                        state = cache.states[request_id]
                        pos = lengths[request_id] + step
                        logical_page, offset = divmod(pos, cache.geometry.page)
                        for layer_idx in range(2):
                            physical = state.block_tables[layer_idx][logical_page]
                            self.assertEqual(
                                cache.slot_mapping_for_layer(layer_idx)[row].item(),
                                physical * cache.geometry.page + offset)
                    for layer_idx in range(2):
                        cache.prepare_metadata(req_ids, layer_idx)
                        cache.write_decode(
                            layer_idx, torch.randn(batch, 2, 1, 4),
                            torch.randn(batch, 2, 1, 4),
                            cache.slot_mapping_for_layer(layer_idx)[:batch])
                    cache.commit_step()

                for request_id in req_ids:
                    state = cache.states[request_id]
                    final_length = lengths[request_id] + 3
                    self.assertEqual(state.logical_length, final_length)
                    expected_pages = (
                        final_length + cache.geometry.page - 1) // cache.geometry.page
                    for layer_idx in range(2):
                        self.assertEqual(
                            list(state.block_tables[layer_idx]),
                            list(range(expected_pages)))

    def test_skip_changes_visible_intervals_not_physical_pages(self):
        cache = self._make_cache(n_layers=2)
        req = cache.add_request()
        length = 24
        for layer_idx in range(2):
            k = torch.full((1, 2, length, 4), float(layer_idx + 1))
            v = torch.full((1, 2, length, 4), float(layer_idx + 3))
            cache.write_prefill(layer_idx, req, k, v)
        cache.freeze(req, [True, False], [True, False])
        state = cache.states[req]
        initial_tables = [dict(table) for table in state.block_tables]

        self.assertEqual(initial_tables[0], initial_tables[1])
        self.assertEqual(list(initial_tables[0]), list(range(6)))
        self.assertEqual(len(cache.layers[0].free_ids),
                         len(cache.layers[1].free_ids))
        skip_intervals = cache._intervals_for_layer(state, 0, length)
        keep_intervals = cache._intervals_for_layer(state, 1, length)
        self.assertNotEqual(skip_intervals, keep_intervals)
        self.assertLess(sum(b - a for a, b in skip_intervals),
                        sum(b - a for a, b in keep_intervals))

        for step in range(11):
            cache.new_step([req])
            cache.plan_common([req])
            for layer_idx in range(2):
                cache.prepare_metadata([req], layer_idx)
                pos = length + step
                logical_page, offset = divmod(pos, cache.geometry.page)
                physical = state.block_tables[layer_idx][logical_page]
                self.assertEqual(
                    cache.slot_mapping_for_layer(layer_idx)[0].item(),
                    physical * cache.geometry.page + offset)
                cache.write_decode(
                    layer_idx,
                    torch.full((1, 2, 1, 4), float(step + 1)),
                    torch.full((1, 2, 1, 4), float(step + 3)),
                    cache.slot_mapping_for_layer(layer_idx)[:1])
            self.assertLess(cache.visible_lens_buf[0, req, 0].item(),
                            cache.visible_lens_buf[1, req, 0].item())
            cache.commit_step()
            self.assertEqual(state.block_tables[0], state.block_tables[1])
            for layer_idx in range(2):
                for logical_page, physical in initial_tables[layer_idx].items():
                    self.assertEqual(
                        state.block_tables[layer_idx][logical_page], physical)

        self.assertEqual(list(state.block_tables[0]), list(range(9)))
        self.assertTrue(torch.equal(
            cache.block_table_buf[0, req], cache.block_table_buf[1, req]))
        self.assertEqual(len(cache.layers[0].free_ids),
                         len(cache.layers[1].free_ids))

        cache.remove_request(req)
        self.assertEqual([len(layer.free_ids) for layer in cache.layers], [64, 64])

        # 同一 request slot 和物理页立即可安全复用，且仍按逻辑页连续分配。
        reused_req = cache.add_request(req)
        for layer_idx in range(2):
            cache.write_prefill(
                layer_idx, reused_req, torch.randn(1, 2, length, 4),
                torch.randn(1, 2, length, 4))
        reused_state = cache.states[reused_req]
        self.assertEqual(reused_state.block_tables[0], initial_tables[0])
        self.assertEqual(reused_state.block_tables[1], initial_tables[1])
        cache.remove_request(reused_req)
        self.assertEqual([len(layer.free_ids) for layer in cache.layers], [64, 64])

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_prefill_batched_scatter_matches_cpu_reference(self):
        def make_cache(device):
            geometry = RegionGeometry(
                sink=2, near=8, window=3, page=4, max_new_tokens=20)
            cache = GraphReadyPagedRegionCache(
                2, geometry, max_requests=8, max_physical_pages=64,
                max_blocks=8, max_visible_len=64, max_query_tokens=16,
                num_kv_heads=2, head_dim_k=4, head_dim_v=4,
                dtype=torch.float32, device=device)
            for layer in cache.layers:
                layer.kv_pool_k.fill_(-7)
                layer.kv_pool_v.fill_(-11)
            return cache

        cpu_cache = make_cache('cpu')
        gpu_cache = make_cache('cuda')
        generator = torch.Generator().manual_seed(20261103)
        lengths = {0: 1, 1: 3, 2: 4, 4: 5, 5: 9, 6: 17}
        for request_id, length in lengths.items():
            cpu_cache.add_request(request_id)
            gpu_cache.add_request(request_id)
            for layer_idx in range(2):
                keys = torch.randn(
                    1, 2, length, 4, generator=generator) + layer_idx
                values = torch.randn(
                    1, 2, length, 4, generator=generator) - layer_idx
                cpu_cache.write_prefill(layer_idx, request_id, keys, values)
                gpu_cache.write_prefill(
                    layer_idx, request_id, keys.cuda(), values.cuda())

        torch.testing.assert_close(
            gpu_cache.block_table_buf.cpu(), cpu_cache.block_table_buf,
            rtol=0, atol=0)
        for layer_idx in range(2):
            # 比较整个 arena，同时覆盖非整页尾部和越界写入。
            torch.testing.assert_close(
                gpu_cache.layers[layer_idx].kv_pool_k.cpu(),
                cpu_cache.layers[layer_idx].kv_pool_k, rtol=0, atol=0)
            torch.testing.assert_close(
                gpu_cache.layers[layer_idx].kv_pool_v.cpu(),
                cpu_cache.layers[layer_idx].kv_pool_v, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
