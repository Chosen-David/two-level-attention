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

    def test_freeze_preallocates_resident_ring_and_decode_has_no_global_alloc(self):
        cache = self._make_cache(n_layers=1)
        req = cache.add_request()
        length = 20
        k = torch.randn(1, 2, length, 4)
        v = torch.randn(1, 2, length, 4)
        cache.write_prefill(0, req, k, v)

        # only sink + SWA remain resident after freeze
        cache.freeze(req, skip_far_list=[True], skip_near_list=[True])
        state = cache.states[req]
        self.assertGreaterEqual(
            state.max_resident_pages_by_layer[0], len(state.block_tables[0]))
        global_free = len(cache.layers[0].free_ids)

        for _ in range(8):
            cache.new_step([req, -1])
            cache.prepare_metadata([req, -1], 0)
            dk = torch.randn(1, 2, 1, 4)
            dv = torch.randn(1, 2, 1, 4)
            cache.write_decode(
                0, dk, dv, cache.slot_mapping_for_layer(0)[:1])
            cache.commit_step()

        self.assertEqual(len(cache.layers[0].free_ids), global_free)
        self.assertEqual(state.logical_length, length + 8)

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
            self.assertEqual(gather.shape, (4, 64))
            self.assertEqual(vis.shape, (4,))
            self.assertEqual(vis.tolist()[2:], [0, 0])

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

    def test_skip_layer_reclaims_and_reuses_pages_independently(self):
        cache = self._make_cache(n_layers=2)
        req = cache.add_request()
        length = 24
        for layer_idx in range(2):
            k = torch.full((1, 2, length, 4), float(layer_idx + 1))
            v = torch.full((1, 2, length, 4), float(layer_idx + 3))
            cache.write_prefill(layer_idx, req, k, v)
        cache.freeze(req, [True, False], [True, False])
        state = cache.states[req]

        self.assertNotIn(2, state.block_tables[0])
        self.assertIn(2, state.block_tables[1])
        self.assertLess(len(state.block_tables[0]), len(state.block_tables[1]))
        self.assertGreater(len(cache.layers[0].free_ids),
                           len(cache.layers[1].free_ids))
        report = cache.report()
        self.assertGreater(report['reclaimed_pages_by_layer'][0], 0)
        self.assertEqual(report['reclaimed_pages_by_layer'][1], 0)

        cache.new_step([req])
        for layer_idx in range(2):
            cache.prepare_metadata([req], layer_idx)
        self.assertEqual(cache.block_table_buf[0, req, 2].item(),
                         cache.layers[0].PAD_PAGE_ID)
        self.assertNotEqual(cache.block_table_buf[1, req, 2].item(),
                            cache.layers[1].PAD_PAGE_ID)
        cache.commit_step()

        keep_before = cache.layers[1].kv_pool_k[:6].clone()
        global_free = [len(layer.free_ids) for layer in cache.layers]
        for _ in range(11):
            cache.new_step([req])
            for layer_idx in range(2):
                cache.prepare_metadata([req], layer_idx)
                cache.write_decode(
                    layer_idx,
                    torch.full((1, 2, 1, 4), float(layer_idx + 1)),
                    torch.full((1, 2, 1, 4), float(layer_idx + 3)),
                    cache.slot_mapping_for_layer(layer_idx)[:1])
            cache.commit_step()

        self.assertEqual([len(layer.free_ids) for layer in cache.layers],
                         global_free)
        common_pages = set(state.block_tables[0]) & set(state.block_tables[1])
        self.assertTrue(any(
            state.block_tables[0][lp] != state.block_tables[1][lp]
            for lp in common_pages))
        self.assertTrue(torch.equal(cache.layers[1].kv_pool_k[:6], keep_before))

        cache.remove_request(req)
        self.assertEqual([len(layer.free_ids) for layer in cache.layers], [64, 64])


if __name__ == '__main__':
    unittest.main()
