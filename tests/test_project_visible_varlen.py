import unittest

import torch

from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.policies.region_geometry import RegionGeometry


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProjectVisibleVarlenTest(unittest.TestCase):
    def _make_cache(self):
        geometry = RegionGeometry(
            sink=2, near=8, window=3, page=4, max_new_tokens=20)
        cache = GraphReadyPagedRegionCache(
            1, geometry,
            max_requests=4,
            max_physical_pages=64,
            max_blocks=64,
            max_visible_len=64,
            max_query_tokens=16,
            num_kv_heads=2,
            head_dim_k=8,
            head_dim_v=4,
            dtype=torch.float32,
            device='cuda',
        )
        return cache, geometry

    def test_batch_matches_eager_projection(self):
        cache, geometry = self._make_cache()
        reqs = []
        configs = [
            (5, False, False),
            (9, True, False),
            (12, True, True),
        ]
        for length, skip_far, skip_near in configs:
            rid = cache.add_request()
            cache.write_prefill(
                0, rid,
                torch.randn(1, 2, length, 8, device='cuda'),
                torch.randn(1, 2, length, 4, device='cuda'),
            )
            cache.freeze(rid, [skip_far], [skip_near])
            reqs.append(rid)

        cache.new_step(reqs)
        cache.plan_common(reqs)

        dims = torch.tensor([0, 3, 7], device='cuda')
        block_size = 4
        lengths = [cache.states[rid].logical_length for rid in reqs]
        batched = cache.project_visible_varlen(
            0, reqs, dims, block_size, lengths=lengths)

        for batch_idx, rid in enumerate(reqs):
            reference = cache.project_visible_for_request(
                0, rid, dims, block_size, length=lengths[batch_idx])
            width = min(batched.shape[1], reference.shape[1])
            torch.testing.assert_close(
                batched[batch_idx, :width], reference[0, :width],
                atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
