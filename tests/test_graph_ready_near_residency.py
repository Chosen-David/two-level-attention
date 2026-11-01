"""Host lifecycle oracle plus optional native PyTorch/CUDA regression.

BASELINE_SOURCE=/path/to/main/paged_region_cache.py runs the same host
checks on an unmodified source. Host storage doubles are not GPU evidence.
"""
import ast
from collections import deque
from itertools import product
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('BASELINE_SOURCE', str(ROOT / 'sparse_attn/cache/paged_region_cache.py')))


def source_class():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'GraphReadyPagedRegionCache')
    names = {'freeze', '_intervals_for_layer', '_build_skip_policy_groups',
             '_resident_page_ranges_for_length', '_merge_page_ranges',
             '_pages_in_ranges', '_max_resident_pages_for_layer',
             '_sync_block_table', '_range_diff'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = []
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(SOURCE), 'exec'), ns)
    return ns['GraphReadyPagedRegionCache']


class Geometry:
    sink = 2
    near = 32
    window = 7
    max_new_tokens = 192

    def __init__(self, page):
        self.page = page

    def bounds(self, length):
        sink = min(self.sink, length)
        near = max(sink, length - self.window)
        far = min(near, max(sink, ((length-self.window-self.near)//self.page)*self.page))
        return sink, far, near


class NearResidencyHostTest(unittest.TestCase):
    def run_lifecycle(self, flags, page):
        c = source_class()()
        c.geometry, c.n_layers = Geometry(page), len(flags)
        prompt = 96
        count = (prompt + page-1)//page
        state = NS(logical_length=prompt, prompt_length=prompt,
                   block_table={i: i for i in range(count)},
                   free_phys_slots=deque(), prefill_topk_ids_by_layer={})
        c.states = {0: state}
        # Tag each token independently of its logical/physical mapping.
        storage = {i: [[layer*100000 + i*page+j for j in range(page)]
                       for layer in range(c.n_layers)] for i in range(count)}
        free = deque()
        next_phys = count

        def release(ids):
            for phys in ids:
                storage[phys] = [[None]*page for _ in flags]
                free.append(phys)

        def allocate(n):
            nonlocal next_phys
            result = []
            for _ in range(n):
                if free:
                    phys = free.popleft()
                else:
                    phys = next_phys
                    next_phys += 1
                    storage[phys] = [[None]*page for _ in flags]
                result.append(phys)
            return result

        c._free_pages, c._alloc_pages = release, allocate
        c._write_block_table_row = lambda *args: None
        c.freeze(0, [f for f, n in flags], [n for f, n in flags])
        reserved = next_phys
        for length in range(prompt, prompt + c.geometry.max_new_tokens + 1):
            if length > prompt:
                c._sync_block_table(state, length)
                pos = length-1
                phys = state.block_table[pos//page]
                for layer in range(c.n_layers):
                    storage[phys][layer][pos % page] = layer*100000+pos
                state.logical_length = length
            # Independent visibility formula; check actual stored token identity.
            sink, far, near = c.geometry.bounds(length)
            for layer, (skip_far, skip_near) in enumerate(flags):
                for pos in range(length):
                    visible = (pos < sink or pos >= near or
                               (sink <= pos < far and not skip_far) or
                               (far <= pos < near and not skip_near))
                    if visible:
                        self.assertIn(pos//page, state.block_table)
                        self.assertEqual(storage[state.block_table[pos//page]][layer][pos % page],
                                         layer*100000+pos, (flags, page, length, layer, pos))
            physical = list(state.block_table.values()) + list(state.free_phys_slots)
            self.assertEqual(len(physical), len(set(physical)))
            self.assertEqual(next_phys, reserved)  # no decode global allocation

    def test_four_states_and_mixed_layers_over_repeated_page_reuse(self):
        for flags in ([(False, False)], [(False, True)], [(True, False)],
                      [(True, True)], [(False, True), (True, True)],
                      [(True, False), (True, True)]):
            for page in (4, 8, 16):
                with self.subTest(flags=flags, page=page):
                    self.run_lifecycle(flags, page)


class ReservationHostTest(unittest.TestCase):
    def test_both_skip_does_not_reserve_unused_near_capacity(self):
        c = source_class()()
        c.geometry = NS(sink=4, near=4096, window=128, page=64,
                        max_new_tokens=512)
        state = NS(prompt_length=8192, skip_far_by_layer=[True, True, False],
                   skip_near_by_layer=[True, False, True])
        self.assertEqual(c._max_resident_pages_for_layer(state, 0), 7)
        self.assertEqual(c._max_resident_pages_for_layer(state, 1), 69)
        self.assertEqual(c._max_resident_pages_for_layer(state, 2), 137)

    def test_bounded_pool_survives_page_edges_and_mixed_layers(self):
        # Independent page-set oracle also accounts for the transient union
        # while missing pages are allocated before stale pages are recycled.
        configs = product((1, 4, 16, 64), (0, 1, 4, 17), (1, 7, 32, 65),
                          (1, 32, 4096), (1, 63), (False, True))
        cache_class = source_class()
        for page, sink, window, near, prompt, mixed in configs:
            c = cache_class()
            c.geometry = g = Geometry(page)
            g.sink, g.window, g.near = sink, window, near
            c.n_layers = 2 if mixed else 1
            state = NS(logical_length=prompt, prompt_length=prompt,
                       block_table={i: i for i in range((prompt+page-1)//page)},
                       free_phys_slots=deque(), prefill_topk_ids_by_layer={})
            c.states = {0: state}
            next_phys = len(state.block_table)
            def allocate(n):
                nonlocal next_phys
                ids = list(range(next_phys, next_phys+n))
                next_phys += n
                return ids
            c._alloc_pages = allocate
            c._free_pages = lambda ids: None
            c._write_block_table_row = lambda *args: None
            c.freeze(0, [True]*c.n_layers, [True, False] if mixed else [True])
            reserved = next_phys
            def expected(length):
                # All far skipped; a near-retaining layer dominates the union.
                start = max(min(sink, length), length-window)
                if mixed:
                    start = min(start, max(min(sink, length),
                                           ((length-window-near)//page)*page))
                return {p//page for p in range(min(sink, length))} | {
                    p//page for p in range(start, length)}
            old = expected(prompt)
            self.assertEqual(set(state.block_table), old)
            for length in range(prompt+1, prompt+193):
                new = expected(length)
                self.assertLessEqual(len(old | new), state.max_resident_pages)
                c._sync_block_table(state, length)
                self.assertEqual(set(state.block_table), new)
                phys = list(state.block_table.values()) + list(state.free_phys_slots)
                self.assertEqual(len(phys), state.max_resident_pages)
                self.assertEqual(len(phys), len(set(phys)))
                self.assertEqual(next_phys, reserved)
                old = new
            if not mixed:
                # The tighter bound must not introduce failures on larger
                # length jumps or rollback that fit the original bound.
                original = (sink+near+window+page-1)//page+2
                for length in (prompt, prompt+191, prompt+page, prompt+1):
                    new = expected(length)
                    if len(old | new) <= original:
                        self.assertLessEqual(len(old | new), state.max_resident_pages)
                        c._sync_block_table(state, length)
                        self.assertEqual(set(state.block_table), new)
                        old = new

    def test_fixed_global_pool_admits_two_both_skip_requests(self):
        c = source_class()()
        c.geometry = g = Geometry(64)
        g.sink, g.near, g.window = 4, 4096, 128
        c.n_layers = 1
        free = deque(range(16))
        def allocate(n):
            if n > len(free):
                raise RuntimeError('host physical page pool exhausted')
            return [free.popleft() for _ in range(n)]
        c._alloc_pages = allocate
        c._free_pages = free.extend
        c._write_block_table_row = lambda *args: None
        c.states = {}
        for rid in range(2):
            c.states[rid] = NS(logical_length=64, prompt_length=64,
                block_table={0: allocate(1)[0]}, free_phys_slots=deque(),
                prefill_topk_ids_by_layer={})
            c.freeze(rid, [True], [True])
        self.assertEqual(len(free), 2)
        for length in range(65, 257):
            for state in c.states.values():
                c._sync_block_table(state, length)
            owned = [p for s in c.states.values()
                     for p in list(s.block_table.values())+list(s.free_phys_slots)]
            self.assertEqual(len(owned), 14)
            self.assertEqual(len(set(owned)), 14)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class NearResidencyNativeTest(unittest.TestCase):
    def check_both_skip_pool(self, device):
        from sparse_attn.cache import GraphReadyPagedRegionCache
        from sparse_attn.policies.region_geometry import RegionGeometry
        geometry = RegionGeometry(sink=4, near=4096, window=128, page=64,
                                  max_new_tokens=192)
        c = GraphReadyPagedRegionCache(1, geometry, max_requests=2,
            max_physical_pages=16, max_blocks=8, max_visible_len=256,
            max_query_tokens=2, num_kv_heads=1, head_dim_k=16, head_dim_v=16,
            dtype=torch.float32, device=device)
        for rid in range(2):
            c.add_request(rid)
            k = torch.full((1,1,64,16), float(rid), device=device)
            c.write_prefill(0, rid, k, k)
            c.freeze(rid, [True], [True])
        self.assertEqual(len(c.layers[0].free_ids), 2)
        for length in range(65, 257):
            c.new_step([0,1])
            c.plan_common([0,1])
            token = torch.arange(2, dtype=torch.float32, device=device).reshape(2,1,1,1).expand(-1,-1,-1,16).contiguous()
            c.write_decode(0, token, token, c.slot_mapping_buf[:2])
            c.commit_step()
        for rid, state in c.states.items():
            for pos in list(range(4)) + list(range(128,256)):
                phys = state.block_table[pos//64]
                torch.testing.assert_close(c.layers[0].kv_pool_k[phys,0,0,pos%64],
                    torch.full((16,), float(rid), device=device))
        for rid in range(2):
            c.remove_request(rid)
        if device == 'cuda':
            torch.cuda.synchronize()
            c._reclaim_pending_free()
        self.assertEqual(len(c.layers[0].free_ids), 16)
        self.assertEqual(c.add_request(), 0)

    def test_cpu_both_skip_pool_and_eos(self):
        self.check_both_skip_pool('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_both_skip_pool_and_eos(self):
        self.check_both_skip_pool('cuda')

    def check_device(self, device):
        from sparse_attn.cache import GraphReadyPagedRegionCache
        from sparse_attn.policies.region_geometry import RegionGeometry
        geometry = RegionGeometry(sink=2, near=32, window=7, page=4, max_new_tokens=96)
        c = GraphReadyPagedRegionCache(1, geometry, max_requests=1,
            max_physical_pages=64, max_blocks=64, max_visible_len=256,
            max_query_tokens=1, num_kv_heads=1, head_dim_k=16, head_dim_v=16,
            dtype=torch.float32, device=device)
        rid = c.add_request()
        k = torch.arange(96, dtype=torch.float32, device=device).reshape(1,1,96,1).expand(-1,-1,-1,16).contiguous()
        c.write_prefill(0, rid, k, k)
        c.freeze(rid, [False], [True])
        for length in range(97, 193):
            c.new_step([rid])
            c.plan_common([rid])
            token = torch.full((1,1,1,16), float(length-1), device=device)
            c.write_decode(0, token, token, c.slot_mapping_buf[:1])
            c.commit_step()
        state = c.states[rid]
        for pos in range(192):
            phys = state.block_table[pos//4]
            torch.testing.assert_close(c.layers[0].kv_pool_k[phys,0,0,pos%4],
                                       torch.full((16,), float(pos), device=device))

    def test_cpu_original_kv_survives_near_to_far(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_original_kv_survives_near_to_far(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
