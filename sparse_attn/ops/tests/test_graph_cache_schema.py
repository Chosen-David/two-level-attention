"""CPU tests of CUDA-graph cache routing, not GPU kernel equivalence.

The CUDA API double records captures; real CPU tensors exercise copy_ including
its broadcasting and dtype conversion. No CUDA execution is simulated as a pass.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import batch_select as bs


class Stream:
    def wait_stream(self, other):
        pass


class Graph:
    def replay(self):
        pass


class GraphCacheSchemaTest(unittest.TestCase):
    def setUp(self):
        self.previous = bs._GRAPH_CACHE
        bs._GRAPH_CACHE = {}
        self.cuda = SimpleNamespace(Stream=Stream, current_stream=Stream,
                                    stream=lambda _: nullcontext(), CUDAGraph=Graph,
                                    graph=lambda _: nullcontext())
        self.captures = []

    def tearDown(self):
        bs._GRAPH_CACHE = self.previous

    def call(self, shape, dtype=torch.float32, optional=False):
        def body(x, avg):
            self.captures.append((tuple(x.shape), x.dtype, avg is None))
            return torch.empty_like(x)
        x = torch.ones(shape, dtype=dtype)
        with patch.object(bs.torch, 'cuda', self.cuda):
            return bs._graph_cached_run(('same-geometry',), body,
                                        {'x': x, 'avg': x if optional else None})

    def test_broadcastable_head_change_cannot_replay_old_shape(self):
        self.call((2, 8, 4, 32))
        out = self.call((2, 1, 4, 32))
        self.assertEqual(tuple(out.shape), (2, 1, 4, 32))
        self.assertEqual(len(bs._GRAPH_CACHE), 2)

    def test_dtype_conversion_cannot_hide_schema_change(self):
        self.call((2, 1, 4, 32), torch.float32)
        out = self.call((2, 1, 4, 32), torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(len(bs._GRAPH_CACHE), 2)

    def test_optional_input_disappearance_cannot_reuse_stale_tensor(self):
        self.call((2, 1, 4, 32), optional=True)
        self.call((2, 1, 4, 32), optional=False)
        self.assertEqual(len(bs._GRAPH_CACHE), 2)

    def test_identical_schema_reuses_across_new_tensor_instances(self):
        first = self.call((2, 1, 4, 32))
        calls = len(self.captures)
        second = self.call((2, 1, 4, 32))
        self.assertIs(first, second)
        self.assertEqual(len(self.captures), calls)
        self.assertEqual(len(bs._GRAPH_CACHE), 1)

    def test_failed_replay_evicts_only_its_schema_entry(self):
        self.call((2, 8, 4, 32))
        self.call((2, 1, 4, 32))
        first_graph = next(iter(bs._GRAPH_CACHE.values()))[0]
        with patch.object(first_graph, 'replay', side_effect=RuntimeError('replay failed')):
            with self.assertRaisesRegex(RuntimeError, 'replay failed'):
                self.call((2, 8, 4, 32))
        self.assertEqual(len(bs._GRAPH_CACHE), 1)
        self.call((2, 8, 4, 32))
        self.assertEqual(len(bs._GRAPH_CACHE), 2)

    def test_query_output_dtype_is_part_of_real_call_site_key(self):
        # Exercise actual key construction. Only the CUDA eligibility gate,
        # index preparation, and graph executor are substituted.
        args = SimpleNamespace(tia_block_size=64, tia_level1_topk=4,
                               tia_level2_topk=128, tia_subspace='tail',
                               tia_l2_subspace='same')
        idx = SimpleNamespace(args=args, sink_tokens=64, sliding_window_size=64,
                              _sink_pool_lo_blk=1, alpha=.25, beta=.25, gamma=.5,
                              far_method='minmax', near_method='avg',
                              _subspace_indices=lambda dev: torch.arange(32, device=dev))
        k = torch.zeros(1, 256, 2, 128, dtype=torch.bfloat16)
        coarse = k[..., :32].reshape(1, 4, 64, 2, 32).mean(2)
        idx._fast_score_coarse = lambda: None
        idx._fast_score_fine = lambda: None
        idx._TILE_COARSE = {}; idx._TILE_FINE = {}; idx._triton_ok = True
        idx.prepare_index = lambda *a: dict(k_min=coarse, k_max=coarse,
                                            k_avg=coarse, k_qat=k[..., :32])
        keys = []
        def capture_key(key, body, inputs):
            keys.append(key)
            return []
        with patch.object(bs.torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=0)), patch.object(bs.torch.cuda, 'device', return_value=nullcontext()), patch.object(bs, '_gate_ok', return_value=True), patch.object(
            bs, '_graph_cached_run', side_effect=capture_key
        ), patch.dict('os.environ', {'TLI_PREFILL_SELECT_GRAPH': '1',
                                    'TLI_PREFILL_L2_TOPK_CAND': '0'}):
            for dtype in (torch.bfloat16, torch.float32):
                q = torch.zeros(1, 256, 8, 128, dtype=dtype)
                bs.batched_group_sels(idx, q, k, 128 ** -.5, [192, 256], 128)
        self.assertEqual(len(keys), 2)
        self.assertNotEqual(keys[0], keys[1])


if __name__ == '__main__':
    unittest.main()
