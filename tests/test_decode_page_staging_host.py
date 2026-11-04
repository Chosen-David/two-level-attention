"""Source-bound host regression; NumPy facade does not validate Torch/CUDA.

NFA_CACHE_SOURCE can point to the preserved baseline for paired checks.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

SOURCE = Path(os.environ.get('NFA_CACHE_SOURCE',
    Path(__file__).resolve().parents[1] / 'sparse_attn/cache/paged_region_cache.py'))


class Array(np.ndarray):
    assignments = 0

    def __setitem__(self, key, value):
        Array.assignments += 1
        super().__setitem__(key, value)

    def fill_(self, value):
        self.fill(value)
        return self

    def zero_(self):
        return self.fill_(0)

    def copy_(self, value, non_blocking=False):
        np.copyto(self, value)
        return self


def array(shape, value=-1):
    return np.full(shape, value, dtype=np.int32).view(Array)


def cache(layers=4, page=4, blocks=64):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'GraphReadyPagedRegionCache')
    names = {'_sync_block_table', '_write_block_table_row', '_prepare_common'}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'PAD_SLOT': -1}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    c = NS(n_layers=layers, geometry=NS(page=page, max_new_tokens=128),
           max_blocks=blocks, layers=[NS(PAD_PAGE_ID=1000000)],
           INVALID_REQ_ID=-1, states={}, pending_lengths={})
    c._cpu_block_table_staging = array((layers, 4, blocks), 1000000)
    c.block_table_buf = array((layers, 4, blocks), 1000000)
    for name, shape in [('slot_mapping', (layers, 32)), ('active_req_ids', (4,)),
                        ('valid_req_ids', (4,)), ('planned_lengths', (4,)),
                        ('query_start_loc', (5,)), ('query_to_req', (32,))]:
        setattr(c, '_cpu_' + name + '_staging', array(shape))
        setattr(c, name + '_buf', array(shape))
    counters = [0] * layers
    def allocate(layer, count):
        start = counters[layer]
        counters[layer] += count
        return [999999 - layer * 10000 - i for i in range(start, start + count)]
    c._alloc_pages = allocate
    for name in names:
        setattr(c, name, scope[name].__get__(c))
    return c


def admit(c, rid, length):
    # Model add_request + fully initialized prefill/freeze rows. Native tests
    # separately exercise those methods with the real allocator and tensors.
    c._cpu_block_table_staging[:, rid].fill_(c.layers[0].PAD_PAGE_ID)
    pages = (length + c.geometry.page - 1) // c.geometry.page
    state = NS(logical_length=length, prompt_length=length,
               block_tables=[dict(enumerate(c._alloc_pages(i, pages)))
                             for i in range(c.n_layers)])
    c.states[rid] = state
    c._write_block_table_row(rid, state)


class DecodePageStagingHost(unittest.TestCase):
    def test_ragged_reorder_chunks_padding_and_reuse(self):
        for layers in (1, 4, 36):
            for page in (4, 16, 64):
                with self.subTest(layers=layers, page=page):
                    c = cache(layers, page)
                    for lifetime in range(2):
                        admit(c, 1, page * 2 + lifetime)
                        admit(c, 3, page + 1)
                        for step in range(20):
                            ids = [1, 3, -1] if step % 2 else [3, 1, -1]
                            lengths = [1, 5 if step % 3 == 0 else 1, 0]
                            before = {r: c.states[r].logical_length for r in (1, 3)}
                            c._prepare_common(ids, lengths)
                            flat = 0
                            for rid, qlen in zip(ids, lengths):
                                if rid < 0:
                                    continue
                                state = c.states[rid]
                                target = np.full((layers, c.max_blocks), 1000000)
                                for layer, table in enumerate(state.block_tables):
                                    for logical, physical in table.items():
                                        target[layer, logical] = physical
                                    for t in range(qlen):
                                        pos = before[rid] + t
                                        expected = table[pos // page] * page + pos % page
                                        self.assertEqual(c.slot_mapping_buf[layer, flat+t], expected)
                                np.testing.assert_array_equal(c.block_table_buf[:, rid], target)
                                np.testing.assert_array_equal(c._cpu_block_table_staging[:, rid], target)
                                state.logical_length += qlen
                                flat += qlen
                        # Re-admit the same slots with fresh, non-identity pages.
                        c.states.clear()

    def test_cross_page_assignments_only_append(self):
        c = cache(36, 64, 2064)
        admit(c, 1, 131072)
        before = Array.assignments
        c._prepare_common([1], [1])
        # 36 page cells + 36 slot cells + 3 request fields + 2 query offsets
        # + 1 query-to-request cell. No assignments to 36 * 2048 old pages.
        self.assertEqual(Array.assignments - before, 78)

    def test_append_only_and_capacity_guards_remain(self):
        c = cache()
        admit(c, 1, 8)
        state = c.states[1]
        with self.assertRaises(ValueError):
            c._sync_block_table(state, 7, request_id=1)
        with self.assertRaises(RuntimeError):
            c._sync_block_table(state, 137, request_id=1)
        del state.block_tables[0][0]
        with self.assertRaisesRegex(RuntimeError, 'append-only'):
            c._prepare_common([1], [1])


if __name__ == '__main__':
    unittest.main()
