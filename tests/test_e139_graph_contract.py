"""CPU contracts; CUDA capture, performance and kernel equivalence remain untested."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bs = load('batch_select', 'sparse_attn/ops/batch_select.py')
checks = load('checks', 'exp/bench/e139_graph_checks.py')

class Index:
    _TILE_FINE = {32: (64, 4)}
    _TILE_COARSE = {32: (64, 4)}
    _triton_ok = True
    def __init__(self, alpha=.25, gamma=.5, sink=64, swa=64):
        self.args = SimpleNamespace(tia_block_size=64, tia_level1_topk=8,
                                    tia_level2_topk=256)
        self.alpha, self.beta, self.gamma = alpha, .5, gamma
        self.sink_tokens, self.sliding_window_size = sink, swa
        self._sink_pool_lo_blk = sink // 64
        self.far_method, self.near_method = 'minmax', 'avg'
    def prepare_index(self, k, cu):
        x = k[..., :32].contiguous()
        p = x.reshape(1, 32, 64, 1, 32)
        return dict(k_min=p.amin(2), k_max=p.amax(2), k_avg=p.mean(2), k_qat=x)
    def _subspace_indices(self, dev):
        return torch.arange(32, device=dev)
    def _fast_score_fine(self, k, q, dtype):
        return torch.einsum('thd,hgd->hgt', k, q.reshape(1, 2, 32)).reshape(2, -1).to(dtype)
    def _fast_score_coarse(self, lo, hi, q, dtype):
        return (self._fast_score_fine(hi, q.clamp(min=0), dtype)
                + self._fast_score_fine(lo, q.clamp(max=0), dtype))

class Contracts(unittest.TestCase):
    def test_actual_callsite_geometry_and_cross_layer_reuse(self):
        torch.manual_seed(157)
        q, k = torch.randn(1, 2048, 2, 128), torch.randn(1, 2048, 1, 128)
        keys, cache = [], {}
        def cached(key, body, inputs):
            keys.append(key)
            if key not in cache:
                cache[key] = body(**inputs)
            return cache[key]
        def run(idx):
            return bs.batched_group_sels(idx, q, k, 128 ** -.5, [2048], 1024)
        with patch.object(bs, '_gate_ok', return_value=True), patch.object(
                torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=7)), patch.object(torch.cuda, 'device', return_value=nullcontext()):
            for flag in ('0', '1'):
                keys.clear(); cache.clear()
                with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='1', TLI_PREFILL_L2_TOPK_CAND=flag), patch.object(bs, '_graph_cached_run', side_effect=cached):
                    indices = [Index(), Index(alpha=.5), Index(), Index(gamma=.125),
                               Index(sink=128), Index(swa=128)]
                    outputs = [run(i) for i in indices]
                self.assertEqual(keys[0], keys[2])
                self.assertEqual(len(set(keys)), 5)
                with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='0', TLI_PREFILL_L2_TOPK_CAND=flag):
                    for idx, out in zip(indices, outputs):
                        checks.require_masks(out, run(idx), 1)
                self.assertFalse(torch.equal(outputs[0][0], outputs[1][0]))

    def test_schema_and_backend_signature(self):
        x = torch.zeros(2, 3)
        def key(t, idx=None, stream=1):
            return bs._graph_signature(idx or Index(), {'x': t}, geometry=(1,),
                                       modes=(True, False), output_dtype=torch.float32, stream=stream)
        self.assertEqual(key(x), key(x.clone()))
        for other in (None, x.double(), x.bool(), x.long(), x[:1], torch.zeros(3, 2).t(), torch.zeros(2, 3, device='meta'), x.clone().requires_grad_()):
            self.assertNotEqual(key(x), key(other))
        self.assertNotEqual(key(x), key(x, stream=2))
        idx = Index(); idx._triton_ok = False
        self.assertNotEqual(key(x), key(x, idx))
        idx = Index(); idx._TILE_FINE = {32: (32, 4)}
        self.assertNotEqual(key(x), key(x, idx))
        idx = Index(); idx._fast_score_fine = lambda *args: None
        self.assertNotEqual(key(x), key(x, idx))

    def test_scalar_signature_boundaries(self):
        def signature(geometry=(1,), modes=(True, False)):
            return bs._graph_signature(Index(), {'x': torch.ones(1)},
                geometry=geometry, modes=modes, output_dtype=torch.float32, stream=1)
        self.assertEqual(signature(), signature())
        for invalid in ((True,), (1.0,), (float('nan'),), (float('inf'),), (None,), ((False,),)):
            with self.subTest(geometry=invalid), self.assertRaises(ValueError):
                signature(geometry=invalid)
        for invalid in ((1, False), (True, 0), (True, float('nan'))):
            with self.subTest(modes=invalid), self.assertRaises(ValueError):
                signature(modes=invalid)
        for invalid in (True, 1, float('nan'), torch.ones(2).to_sparse()):
            with self.subTest(input_type=type(invalid)), self.assertRaises(ValueError):
                bs._input_schema({'x': invalid})
        with self.assertRaises(ValueError):
            bs._input_schema({1: torch.ones(1)})

    def test_values_and_noncontiguous_copy(self):
        # Tensor data are deliberately absent from the signature, including NaN.
        # They are replay inputs; rejecting numerical failures belongs to gates.
        a = torch.arange(6.).reshape(3, 2).t()
        b = torch.full((3, 2), float('nan')).t()
        self.assertFalse(a.is_contiguous())
        self.assertEqual(bs._input_schema({'x': a}), bs._input_schema({'x': b}))
        class Graph:
            calls = 0
            def replay(self): self.calls += 1
        g = Graph()
        static = {'x': a.contiguous()}
        ent = (g, static, 'output', None, bs._input_schema({'x': a}), bs._input_schema(static))
        with patch.dict(bs._GRAPH_CACHE, {'test': ent}, clear=True):
            bs._graph_cached_run('test', None, {'x': a + 10})
            self.assertTrue(torch.equal(static['x'], a + 10))
            bs._graph_cached_run('test', None, {'x': b})
            self.assertTrue(bool(torch.isnan(static['x']).all()))
            with self.assertRaises(RuntimeError): checks.require_outputs(static['x'], a)
            self.assertEqual(g.calls, 2)
            # Static-buffer mutation is also rejected before replay.
            static['x'] = torch.ones(1, 3)
            with self.assertRaises(ValueError): bs._graph_cached_run('test', None, {'x': a})
            self.assertEqual(g.calls, 2)

    def test_real_hit_checks_before_copy_or_replay(self):
        class Graph:
            calls = 0
            def replay(self): self.calls += 1
        g = Graph(); static = {'x': torch.zeros(2, 3), 'avg': None}
        source = {'x': torch.ones(2, 3), 'avg': None}
        ent = (g, static, 'output', None, bs._input_schema(source), bs._input_schema(static))
        with patch.dict(bs._GRAPH_CACHE, {'test': ent}, clear=True):
            self.assertEqual(bs._graph_cached_run('test', None, source), 'output')
            self.assertEqual(g.calls, 1)
            for bad in ({'x': torch.ones(1, 3), 'avg': None},
                        {'x': torch.ones(2, 3).double(), 'avg': None},
                        {'x': torch.ones(2, 3, dtype=torch.bool), 'avg': None},
                        {'x': torch.ones(2, 3, dtype=torch.long), 'avg': None},
                        {'x': torch.ones(2, 3, device='meta'), 'avg': None},
                        {'x': torch.ones(2, 3, requires_grad=True), 'avg': None},
                        {'x': torch.ones(3, 2).t(), 'avg': None},
                        {'x': torch.ones(2, 3)},
                        {'x': torch.ones(2, 3), 'avg': torch.ones(1)}):
                with self.assertRaises(ValueError): bs._graph_cached_run('test', None, bad)
            self.assertEqual(g.calls, 1)
            self.assertTrue(torch.equal(static['x'], source['x']))

    def test_hard_gates_subprocess_exit(self):
        code = "import sys; sys.path.insert(0, %r); import torch; from e139_graph_checks import *; " % str(ROOT / 'exp/bench')
        cases = [
            ("require_masks([torch.ones(2,dtype=torch.bool)], [torch.ones(2,dtype=torch.bool)], 1)", True),
            ("require_outputs(torch.ones(2), torch.ones(2))", True),
            ("require_masks(None, [], 1)", False),
            ("require_masks([], [], 1)", False),
            ("require_masks([torch.ones(2,dtype=torch.bool)], [torch.zeros(2,dtype=torch.bool)], 1)", False),
            ("require_outputs(torch.ones(2), torch.zeros(2))", False),
            ("require_outputs(torch.tensor([float('nan')]), torch.ones(1))", False),
            ("require_outputs(torch.tensor([float('inf')]), torch.tensor([float('inf')]))", False),
            ("require_outputs(torch.ones(2), torch.ones(1))", False),
            ("require_graph(dict(fallback=0,replay=0),dict(fallback=1,replay=1),'replay')", False),
            ("require_graph(dict(fallback=0,replay=0),dict(fallback=0,replay=0),'replay')", False),
        ]
        for optimized in (False, True):
            for statement, success in cases:
                with self.subTest(optimized=optimized, statement=statement):
                    result = subprocess.run([sys.executable] + (['-O'] if optimized else []) + ['-c', code + statement], capture_output=True)
                    expected_exit = 0 if success else 1
                    print(f'EXIT optimized={optimized} expected={expected_exit} actual={result.returncode} case={statement}', flush=True)
                    self.assertEqual(result.returncode, expected_exit, result.stderr.decode())

if __name__ == '__main__': unittest.main()
