"""CPU staging regression, optional native cache checks and CPU microbenchmark.

Host tests execute source methods with a NumPy tensor facade. They do not certify
CUDA execution. --benchmark requires real Torch and measures CPU staging only.
NFA_CACHE_SOURCE selects a preserved baseline for regression/microbenchmark runs.
"""
import ast
from collections import deque
import heapq
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('NFA_CACHE_SOURCE', ROOT / 'sparse_attn/cache/paged_region_cache.py'))


def methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'GraphReadyPagedRegionCache')
    names = {'write_prefill', '_write_block_table_row', 'add_request', 'remove_request'}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    def state(**kwargs):
        return NS(logical_length=0, prompt_length=0, **kwargs)
    scope = {'RequestState': state, 'RequestPhase': NS(PREFILL='PREFILL'), 'deque': deque}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return {name: scope[name] for name in names}


class Array(np.ndarray):
    device = 'cpu'
    is_cuda = False
    fills = 0
    assignments = 0

    def fill_(self, value):
        Array.fills += 1
        self.fill(value)
        return self

    def __setitem__(self, index, value):
        Array.assignments += 1
        super().__setitem__(index, value)

    def copy_(self, value):
        np.copyto(self, value)
        return self


def array(value):
    return np.asarray(value).view(Array)


def host_cache(layers, page, capacity):
    c = NS(n_layers=layers, max_requests=4, max_blocks=capacity,
           geometry=NS(page=page), states={}, request_phases={}, active_request_id=None)
    c.layers = [NS(PAD_PAGE_ID=capacity, num_kv_heads=2,
                   kv_pool_k=array(np.full((capacity+1, 1, 2, page, 3), -1.)),
                   kv_pool_v=array(np.full((capacity+1, 1, 2, page, 2), -1.)))
                for _ in range(layers)]
    c._cpu_block_table_staging = array(np.full((layers, 4, capacity), capacity, dtype=np.int32))
    # Independent layers and reversed page IDs exercise non-identity mappings.
    free = [list(range(capacity)) for _ in range(layers)]
    c._alloc_pages = lambda layer, count: [capacity-1-heapq.heappop(free[layer]) for _ in range(count)]
    c._free_pages = lambda layer, ids: [heapq.heappush(free[layer], capacity-1-p) for p in ids]
    for name, fn in methods().items():
        setattr(c, name, fn.__get__(c))
    return c


def expected_rows(c):
    expected = np.full(c._cpu_block_table_staging.shape, c.layers[0].PAD_PAGE_ID, dtype=np.int32)
    for rid, state in c.states.items():
        for layer, table in enumerate(state.block_tables):
            for logical, physical in table.items():
                if 0 <= logical < c.max_blocks:
                    expected[layer, rid, logical] = physical
    return expected


class PrefillLayerStaging(unittest.TestCase):
    def test_interleaved_layers_overwrite_remove_and_reuse(self):
        for layers in (1, 4, 36):
            for page in (4, 16):
                for lengths in ((1, 3), (17, 35), (64, 65)):
                    with self.subTest(layers=layers, page=page, lengths=lengths):
                        c = host_cache(layers, page, 64)
                        for lifetime in range(2):
                            for rid in (1, 3):
                                c.add_request(rid)
                            order = list(range(layers))[::-1] if lifetime else list(range(layers))
                            for layer in order:
                                for rid, length in zip((1, 3), lengths):
                                    # Non-contiguous head-major inputs, distinct K/V dimensions.
                                    k = array(np.arange(length*6, dtype=float).reshape(1,length,2,3).transpose(0,2,1,3))
                                    v = array(np.arange(length*4, dtype=float).reshape(1,length,2,2).transpose(0,2,1,3))
                                    before = Array.fills
                                    out = c.write_prefill(layer, rid, k, v)
                                    self.assertIs(out[0], k)
                                    self.assertIs(out[1], v)
                                    self.assertEqual(Array.fills-before, 1)
                                    np.testing.assert_array_equal(c._cpu_block_table_staging, expected_rows(c))
                                    c.write_prefill(layer, rid, k+1, v+1)
                                    np.testing.assert_array_equal(c._cpu_block_table_staging, expected_rows(c))
                                    for token in range(length):
                                        phys = c.states[rid].block_tables[layer][token//page]
                                        np.testing.assert_array_equal(c.layers[layer].kv_pool_k[phys,0,:,token%page,:], k[0,:,token,:]+1)
                            # Request-wide changes still refresh every row, including holes.
                            for rid in (1, 3):
                                for layer in range(layers):
                                    table = c.states[rid].block_tables[layer]
                                    for logical in list(table):
                                        if (logical+layer)%2:
                                            c._free_pages(layer, [table.pop(logical)])
                                    table[-1] = 0
                                    table[c.max_blocks] = 0
                                c._write_block_table_row(rid, c.states[rid])
                                np.testing.assert_array_equal(c._cpu_block_table_staging, expected_rows(c))
                                for table in c.states[rid].block_tables:
                                    del table[-1]
                                    del table[c.max_blocks]
                                c.remove_request(rid)
                                np.testing.assert_array_equal(c._cpu_block_table_staging, expected_rows(c))

    def test_linear_prefill_page_assignments(self):
        c = host_cache(36, 4, 20)
        c.add_request(1)
        k = array(np.zeros((1,2,65,3)))
        v = array(np.zeros((1,2,65,2)))
        before = Array.assignments
        for layer in range(c.n_layers):
            c.write_prefill(layer, 1, k, v)
        self.assertEqual(Array.assignments-before, 36*17)

    def native_cache(self, device):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('PyTorch unavailable')
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        spec = importlib.util.spec_from_file_location('prefill_staging_cache', SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from sparse_attn.policies.region_geometry import RegionGeometry
        c = module.GraphReadyPagedRegionCache(4,
            RegionGeometry(sink=2,near=8,window=3,page=4,max_new_tokens=8),
            max_requests=4,max_physical_pages=80,max_blocks=32,max_visible_len=64,
            max_query_tokens=1,num_kv_heads=2,head_dim_k=4,head_dim_v=4,
            dtype=torch.float32,device=device)
        for lifetime in range(2):
            for rid in (1,3): c.add_request(rid)
            for layer in (3,1,2,0):
                for rid,n in ((1,35),(3,17)):
                    k = torch.arange(n*8,device=device,dtype=torch.float32).reshape(1,n,2,4).transpose(1,2)
                    out = c.write_prefill(layer,rid,k,k+10)
                    self.assertIs(out[0], k)
                    np.testing.assert_array_equal(c._cpu_block_table_staging.numpy(), expected_rows(c))
                    for tok in range(n):
                        phys=c.states[rid].block_tables[layer][tok//4]
                        torch.testing.assert_close(c.layers[layer].kv_pool_k[phys,0,:,tok%4,:],k[0,:,tok,:],rtol=0,atol=0)
            for rid in (1,3):
                c.freeze(rid,[False,True,False,True],[False,False,True,True])
                np.testing.assert_array_equal(c._cpu_block_table_staging.numpy(), expected_rows(c))
            if device=='cuda': torch.cuda.synchronize()
            for rid in (1,3): c.remove_request(rid)
            if device=='cuda':
                torch.cuda.synchronize()
                c._reclaim_pending_free()

    def test_native_cpu_freeze_reuse(self): self.native_cache('cpu')
    def test_native_cuda_freeze_reuse(self): self.native_cache('cuda')


def benchmark():
    import torch
    fn = methods()['_write_block_table_row']
    scoped = 'layer_idx' in inspect.signature(fn).parameters
    results=[]
    for layers in (1,36):
        for pages in (512,2048):
            c=NS(n_layers=layers,max_blocks=pages+16,layers=[NS(PAD_PAGE_ID=pages+16)],
                 _cpu_block_table_staging=torch.full((layers,1,pages+16),pages+16,dtype=torch.int32))
            tables=[dict(enumerate(range(pages-1,-1,-1))) for _ in range(layers)]
            durations=[]
            for repeat in range(7):
                state=NS(block_tables=[{} for _ in range(layers)])
                start=time.perf_counter()
                for layer in range(layers):
                    state.block_tables[layer]=tables[layer]
                    if scoped: fn(c,0,state,layer_idx=layer)
                    else: fn(c,0,state)
                elapsed=(time.perf_counter()-start)*1000
                if repeat: durations.append(elapsed)
                target=torch.tensor(list(range(pages-1,-1,-1)),dtype=torch.int32)
                for layer in range(layers):
                    torch.testing.assert_close(c._cpu_block_table_staging[layer,0,:pages],target,rtol=0,atol=0)
            results.append(dict(layers=layers,pages=pages,median_ms=statistics.median(durations),samples_ms=durations))
    print(json.dumps(dict(source=str(SOURCE),scope='real Torch CPU staging only, no KV copies/model/GPU',results=results),indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv: benchmark()
    else: unittest.main()
