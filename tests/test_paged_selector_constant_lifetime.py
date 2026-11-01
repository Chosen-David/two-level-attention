"""Source-bound ownership tests. Host doubles do not validate CUDA execution."""
import ast
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'sparse_attn/indexer/nfa_indexer.py'


def load_method(name, torch_module):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NFAIndexer')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {'torch': torch_module}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[name]


class Device:
    def __init__(self, kind='cpu'):
        self.type = kind


class HostTorch:
    long = np.int64

    def __init__(self):
        self.refs = []
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)

    def track(self, value):
        self.refs.append(weakref.ref(value))
        return value

    def arange(self, n, *, device):
        return self.track(np.arange(n, dtype=np.int64))

    def tensor(self, value, *, dtype=None, device):
        return self.track(np.array(value, dtype=dtype))


class ConstantLifetimeTest(unittest.TestCase):
    def setUp(self):
        self.torch = HostTorch()
        self.method = load_method('_selection_constants', self.torch)
        self.owner = SimpleNamespace()
        self.device = Device()

    def call(self, length, *, sparse=False, retain=True):
        return self.method(self.owner, self.device, np.float32,
                           (length + 63) // 64, length, 64, 0, 4, sparse,
                           retain_for_graph=retain)

    def test_growing_eager_lengths_do_not_pin_constants(self):
        for sparse in (False, True):
            for length in range(4097, 4161):
                entry = self.call(length, sparse=sparse, retain=False)
                values = entry[1]
                np.testing.assert_array_equal(values['ar_blk'], np.arange((length + 63) // 64))
                np.testing.assert_array_equal(values['ar_bs'], np.arange(64))
                if sparse:
                    self.assertIsNone(values['ar_tok'])
                else:
                    np.testing.assert_array_equal(values['ar_tok'], np.arange(length))
                self.assertTrue(np.isneginf(values['inf_t']))
                self.assertEqual(int(values['zero_l']), 0)
                self.assertEqual(int(values['far_lo']), 0)
                self.assertTrue(bool(values['sink_cross']))
                del entry, values
        gc.collect()
        self.assertFalse(getattr(self.owner, '_graph_const_entries', {}))
        self.assertFalse(hasattr(self.owner, '_graph_const_cache'))
        self.assertTrue(all(ref() is None for ref in self.torch.refs))

    def test_graph_addresses_survive_eager_and_new_geometry(self):
        old = self.call(4096)
        ref = weakref.ref(old[1]['ar_tok'])
        self.assertIs(old, self.call(4096))
        new = self.call(8192)
        entries = dict(self.owner._graph_const_entries)
        for length in range(8193, 8257):
            self.call(length, retain=False)
        gc.collect()
        self.assertEqual(list(entries), list(self.owner._graph_const_entries))
        for key, value in entries.items():
            self.assertIs(value, self.owner._graph_const_entries[key])
        self.assertIs(self.owner._graph_const_cache, new)
        del old
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertIs(self.call(4096)[1]['ar_tok'], ref())

    def test_transient_call_can_reuse_existing_pinned_entry(self):
        old = self.call(4096)
        count = len(self.torch.refs)
        self.assertIs(self.call(4096, retain=False), old)
        self.assertEqual(len(self.torch.refs), count)

    def test_sparse_and_dense_geometries_do_not_alias(self):
        dense = self.call(4096)
        sparse = self.call(4096, sparse=True)
        self.assertEqual(len(self.owner._graph_const_entries), 2)
        self.assertIsNot(dense, sparse)
        self.assertIsNone(sparse[1]['ar_tok'])

    def test_transient_capture_rejected_before_allocation_or_reuse(self):
        self.device = Device('cuda')
        self.call(4096)
        count = len(self.torch.refs)
        self.torch.cuda.is_current_stream_capturing = lambda: True
        for length in (4096, 8192):
            with self.assertRaisesRegex(RuntimeError, 'eager-only'):
                self.call(length, retain=False)
        self.assertEqual(len(self.torch.refs), count)
        self.assertEqual(len(self.owner._graph_const_entries), 1)

    def test_wrapper_preserves_default_and_forwards_opt_out(self):
        method = load_method('compute_selection_graph', self.torch)
        seen = []
        owner = SimpleNamespace(compute_mask_graph=lambda *a, **kw: seen.append(kw))
        method(owner, None, None, 1., None)
        method(owner, None, None, 1., None, selection_mask='mask', retain_graph_constants=False)
        self.assertTrue(seen[0]['retain_graph_constants'])
        self.assertFalse(seen[1]['retain_graph_constants'])
        self.assertEqual(seen[1]['selection_mask'], 'mask')

    def test_real_paged_call_opts_out(self):
        source = ROOT / 'sparse_attn/ops/paged_region_kernels_nfa.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'region_selection_compact')
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'compute_selection_graph')
        seen = []
        q = SimpleNamespace(transpose=lambda *a: SimpleNamespace(contiguous=lambda: 'query'))
        env = dict(idx=SimpleNamespace(compute_selection_graph=lambda *a, **kw: seen.append(kw)),
                   q=q, dictionary={}, scale=1., vl=None, visible='visible')
        eval(compile(ast.Expression(call), str(source), 'eval'), env)
        self.assertIs(seen[0].get('retain_graph_constants', True), False)
        self.assertEqual(seen[0]['selection_mask'], 'visible')


class NativeCudaLifetimeTest(unittest.TestCase):
    def test_selected_ids_match_retained_constants_at_changing_lengths(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('native torch unavailable')
        import torch
        if not torch.cuda.is_available():
            self.skipTest('native CUDA unavailable')
        from sparse_attn.indexer.nfa_indexer import NFAIndexer
        from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
        torch.manual_seed(760)
        idx = NFAIndexer(nfa_args())
        idx.layer_idx = 0
        transient = NFAIndexer(nfa_args())
        transient.layer_idx = 0
        with torch.inference_mode(), patch.dict('os.environ', {
                'NFA_SPARSE_SELECTION': '0', 'NFA_LIVE_NORMALIZATION': '0',
                'NFA_COMPACT_NORMALIZATION': '0', 'NFA_COMPACT_RANKING': '0',
                'NFA_L2_BYPASS': '0'}):
            for length in (4097, 4098, 4160, 9217):
                bs = idx.args.tia_block_size
                pad = (length + bs - 1) // bs * bs
                k = torch.randn(1, pad, 2, 32, device='cuda', dtype=torch.bfloat16)
                pages = k.reshape(1, -1, bs, 2, 32)
                index = dict(k_min=pages.amin(2), k_max=pages.amax(2),
                             k_avg=pages.mean(2), k_qat=k[:, :length])
                q = torch.randn(1, 1, 8, 128, device='cuda', dtype=k.dtype)
                vl = torch.tensor([length], device='cuda', dtype=torch.int64)
                visible = torch.ones(length, device='cuda', dtype=torch.bool)
                visible[512:1024] = False
                expected = idx.compute_selection_graph(q, index, 128**-.5, vl,
                                                        selection_mask=visible)
                actual = transient.compute_selection_graph(q, index, 128**-.5, vl,
                    selection_mask=visible, retain_graph_constants=False)
                torch.testing.assert_close(actual.indices, expected.indices, rtol=0, atol=0)
                torch.testing.assert_close(actual.counts, expected.counts, rtol=0, atol=0)
                self.assertFalse(getattr(transient, '_graph_const_entries', {}))

    def test_two_graphs_survive_transient_eager_calls(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('native torch unavailable')
        import torch
        if not torch.cuda.is_available():
            self.skipTest('native CUDA unavailable')
        method = load_method('_selection_constants', torch)
        owner = SimpleNamespace()
        device = torch.device('cuda')
        def constants(length, retain=True):
            return method(owner, device, torch.float32, (length + 63)//64,
                          length, 64, 0, 4, False, retain_for_graph=retain)
        graphs, outputs = [], []
        for length in (4096, 8192):
            entry = constants(length)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = entry[1]['ar_tok'] + 7
            graphs.append(graph)
            outputs.append(out)
        for length in range(8193, 8257):
            constants(length, False)
        gc.collect()
        self.assertEqual(len(owner._graph_const_entries), 2)
        for graph, out, length in zip(graphs, outputs, (4096, 8192)):
            graph.replay()
            torch.testing.assert_close(out, torch.arange(length, device=device) + 7, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
