"""Source-bound host regression; optional native CPU/CUDA checks.

BASELINE_SOURCE=/path/to/old/paged_region_cache.py runs the same oracle on
the pre-fix helpers. NumPy doubles validate address mapping, not GPU execution.
"""
import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

SOURCE = Path(os.environ.get('BASELINE_SOURCE', str(
    Path(__file__).resolve().parents[1] / 'sparse_attn/cache/paged_region_cache.py')))


class Tensor(np.ndarray):
    device = 'cpu'

    def numel(self):
        return self.size

    def is_contiguous(self):
        return self.flags.c_contiguous

    def contiguous(self):
        return tensor(np.ascontiguousarray(self))

    def to(self, *args, **kwargs):
        dtype = args[0] if args and args[0] != 'cpu' else self.dtype
        return tensor(np.asarray(self, dtype=dtype))

    def copy_(self, other, **kwargs):
        self[...] = other
        return self

    def zero_(self):
        self.fill(0)
        return self

    def gather(self, dim, indices):
        return tensor(np.take_along_axis(self, indices, axis=dim))

    def unsqueeze(self, dim):
        return tensor(np.expand_dims(self, dim))

    def clamp(self, min=None, max=None):
        return tensor(np.clip(self, min, max))


def tensor(data, dtype=None, **kwargs):
    return np.asarray(data, dtype=dtype).view(Tensor)


HOST = SimpleNamespace(
    Tensor=Tensor, int32=np.int32, int64=np.int64,
    tensor=tensor, as_tensor=tensor,
    clamp=lambda x, min=None, max=None: x.clamp(min, max),
    div=lambda x, y, **kw: tensor(np.floor_divide(x, y)),
    remainder=lambda x, y: tensor(np.remainder(x, y)),
    where=lambda cond, a, b: tensor(np.where(cond, a, b)),
    full_like=lambda x, v: tensor(np.full_like(x, v)),
    arange=lambda a, b, out=None, **kw: out.copy_(np.arange(a, b)),
)


def methods(backend):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'GraphReadyPagedRegionCache')
    names = {'_store_gather_slots', '_store_gather_slots_batch',
             '_prepare_layer_gather'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in names]
    cls.bases = []
    ns = {'torch': backend, 'INVALID_GATHER_IDX': -1}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[cls], type_ignores=[])), str(SOURCE), 'exec'), ns)
    return ns['GraphReadyPagedRegionCache']


class RequestRowsHostTest(unittest.TestCase):
    def make(self, page):
        c = methods(HOST)()
        c.geometry = SimpleNamespace(page=page)
        c.max_blocks, c.max_visible_len = 4, 4
        # Each request has unrelated, noncontiguous physical pages.
        c.tables = [{0: 7, 1: 3, 2: 11, 3: 9},
                    {0: 5, 1: 13, 2: 2, 3: 6},
                    {0: 17, 1: 4, 2: 16, 3: 8}]
        c.block_table_buf = tensor([list(row.values()) for row in c.tables], np.int32)
        c.gather_indices_buf = tensor(np.full((2, 3, 4), -1, dtype=np.int32))
        c.visible_lens_buf = tensor(np.zeros((2, 3), dtype=np.int32))
        return c

    def expected(self, c, req, ids):
        return [-1 if req < 0 or i < 0 else
                c.tables[req][int(i) // c.geometry.page] * c.geometry.page
                + int(i) % c.geometry.page for i in ids]

    def test_single_rows_after_eos_and_reorder(self):
        for page in (3, 4, 64):
            c = self.make(page)
            ids = tensor([0, page - 1, page + 1, -1], np.int32)
            for reqs in ([0, 1, 2], [1], [2, 0], [2, 1, 0]):
                for row, req in enumerate(reqs):
                    kw = {'request_id': req} if 'request_id' in inspect.signature(
                        c._store_gather_slots).parameters else {}
                    c._store_gather_slots(1, row, ids, 4, **kw)
                    with self.subTest(page=page, reqs=reqs, row=row):
                        self.assertEqual(c.gather_indices_buf[1, row].tolist(),
                                         self.expected(c, req, ids))

    def test_batched_padding_and_slot_reuse(self):
        for page in (3, 4, 64):
            c = self.make(page)
            for reqs in ([0, 1, 2], [1, -1, -1], [2, 0, -1], [2, 1, 0]):
                ids = tensor([[0, page, 2 * page + 1, -1] if r >= 0
                              else [-1] * 4 for r in reqs], np.int32)
                counts = tensor([3 if r >= 0 else 0 for r in reqs], np.int32)
                args = (reqs,) if 'request_ids' in inspect.signature(
                    c._store_gather_slots_batch).parameters else ()
                c._store_gather_slots_batch(0, ids, counts, *args)
                with self.subTest(page=page, reqs=reqs):
                    self.assertEqual(c.gather_indices_buf[0].tolist(),
                                     [self.expected(c, r, ids[b]) for b, r in enumerate(reqs)])
                    self.assertEqual(c.visible_lens_buf[0].tolist(), counts.tolist())
                # A replacement request reuses request slot 0 with new pages.
                c.tables[0] = {0: 22, 1: 23, 2: 24, 3: 25}
                c.block_table_buf[0] = list(c.tables[0].values())

    def test_interval_metadata_passes_request_identity(self):
        c = self.make(4)
        c.states = {r: SimpleNamespace(logical_length=8) for r in range(3)}
        c.pending_lengths = {}
        c._intervals_for_layer = lambda *args: [(0, 2), (5, 7)]
        c._gather_logical_tmp = tensor(np.empty(4, dtype=np.int32))
        reqs = [2, 0, -1]
        c._prepare_layer_gather(1, reqs)
        for row, req in enumerate(reqs[:2]):
            self.assertEqual(c.gather_indices_buf[1, row].tolist(),
                             self.expected(c, req, [0, 1, 5, 6]))
        self.assertEqual(c.visible_lens_buf[1].tolist(), [4, 4, 0])

    def test_selected_metadata_passes_request_identity(self):
        c = self.make(4)
        c.states = {r: SimpleNamespace(logical_length=8) for r in range(3)}
        c.pending_lengths = {}
        c._intervals_for_layer = lambda *args: [(0, 8)]
        reqs = [2, 0, -1]
        selected = {2: tensor([1, 5], np.int32), 0: tensor([0, 7], np.int32)}
        c._prepare_layer_gather(0, reqs, selected)
        for row, req in enumerate(reqs[:2]):
            self.assertEqual(c.gather_indices_buf[0, row, :2].tolist(),
                             self.expected(c, req, selected[req]))
        self.assertEqual(c.visible_lens_buf[0].tolist(), [2, 2, 0])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'native PyTorch unavailable')
class RequestRowsNativeTest(unittest.TestCase):
    def test_cpu_helpers(self):
        c = methods(torch)()
        c.geometry = SimpleNamespace(page=4)
        c.max_blocks = 2
        c.block_table_buf = torch.tensor([[7, 8], [3, 9]], dtype=torch.int32)
        c.gather_indices_buf = torch.full((1, 2, 3), -1, dtype=torch.int32)
        c.visible_lens_buf = torch.zeros((1, 2), dtype=torch.int32)
        c._store_gather_slots_batch(0, torch.tensor([[0, 5, -1], [-1, -1, -1]],
            dtype=torch.int32), torch.tensor([2, 0]), [1, -1])
        self.assertEqual(c.gather_indices_buf.tolist(), [[[12, 37, -1], [-1, -1, -1]]])

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_gather_replay_after_eos_and_replacement(self):
        from sparse_attn.cache import GraphReadyPagedRegionCache
        from sparse_attn.ops.paged_region_kernels import sparse_paged_attention_gather
        from sparse_attn.policies.region_geometry import RegionGeometry
        c = GraphReadyPagedRegionCache(1, RegionGeometry(sink=2, near=32,
            window=4, page=16, max_new_tokens=16), max_requests=4,
            max_physical_pages=64, max_blocks=8, max_visible_len=16,
            max_query_tokens=4, num_kv_heads=1, num_query_heads=2,
            head_dim_k=32, head_dim_v=32, dtype=torch.float32, device='cuda')
        tags = {0: 1., 1: 4., 2: 7.}

        def add(rid):
            c.add_request(rid)
            length = 12 + rid * 5
            k = torch.zeros((1, 1, length, 32), device='cuda')
            c.write_prefill(0, rid, k, torch.full_like(k, tags[rid]))
            c.freeze(rid, [False], [False])

        for rid in tags:
            add(rid)
        q = torch.zeros((4, 2, 32), device='cuda')
        q_to_row = torch.arange(4, dtype=torch.int32, device='cuda')

        def prepare(reqs):
            c.new_step(reqs)
            c.plan_common(reqs, query_lengths=[0] * 4)
            ids = torch.tensor([[0, 1] if r >= 0 else [-1, -1] for r in reqs],
                               dtype=torch.int32, device='cuda')
            counts = torch.tensor([2 if r >= 0 else 0 for r in reqs],
                                  dtype=torch.int32, device='cuda')
            c._store_gather_slots_batch(0, ids, counts, reqs)

        def attend():
            return sparse_paged_attention_gather(c.layers[0], q,
                c.gather_indices_buf[0], c.visible_lens_buf[0], q_to_row, scale=32 ** -.5)

        prepare([0, 1, 2, -1])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                attend()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = attend()
        for reqs in ([2, 1, 0, -1], [1, 2, -1, -1]):
            prepare(reqs)
            graph.replay()
            for row, rid in enumerate(reqs):
                if rid >= 0:
                    torch.testing.assert_close(out[row], torch.full_like(out[row], tags[rid]))
        c.remove_request(0)
        torch.cuda.synchronize()
        c.new_step([1, 2, -1, -1])
        tags[0] = 9.
        add(0)
        prepare([2, 0, 1, -1])
        graph.replay()
        for row, rid in enumerate([2, 0, 1]):
            torch.testing.assert_close(out[row], torch.full_like(out[row], tags[rid]))


if __name__ == '__main__':
    unittest.main()
