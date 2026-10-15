"""CPU host-route tests, not CUDA capture or GPU performance validation."""
import os
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import batch_select as bs


class RequiredGraphDispatchTest(unittest.TestCase):
    def setUp(self):
        # Dev additionally binds graph identity to device/stream. Stub only the
        # CUDA context metadata; these remain CPU host-route tests, not capture.
        stream = patch.object(torch.cuda, 'current_stream',
                              return_value=SimpleNamespace(cuda_stream=7))
        device = patch.object(torch.cuda, 'device', return_value=nullcontext())
        stream.start(); device.start()
        self.addCleanup(stream.stop); self.addCleanup(device.stop)
        args = SimpleNamespace(tia_block_size=64, tia_level1_topk=4,
                               tia_level2_topk=128, tia_subspace='tail',
                               tia_l2_subspace='same')
        self.idx = SimpleNamespace(
            args=args, sink_tokens=64, sliding_window_size=64,
            _TILE_COARSE={}, _TILE_FINE={}, _triton_ok=True,
            _sink_pool_lo_blk=1, alpha=.25, beta=.25, gamma=.5,
            far_method='minmax', near_method='avg',
            _subspace_indices=lambda dev: torch.arange(32, device=dev))
        self.k = torch.zeros(1, 256, 2, 128)
        self.q = torch.zeros(1, 256, 8, 128)
        coarse = self.k[..., :32].reshape(1, 4, 64, 2, 32).mean(2)
        self.idx.prepare_index = lambda *a: dict(
            k_min=coarse, k_max=coarse, k_avg=coarse, k_qat=self.k[..., :32])
        # Scoring and CUDA context metadata are substituted; real CPU glue executes.
        def score(keys, query, dtype):
            return torch.zeros(query.shape[0], keys.shape[0], dtype=dtype)
        self.idx._fast_score_fine = score
        self.idx._fast_score_coarse = lambda lo, hi, query, dtype: score(lo, query, dtype)
        self.args = (self.idx, self.q, self.k, 128 ** -.5, [192, 256], 128)
        self.env = patch.dict(os.environ, {'TLI_PREFILL_SELECT_GRAPH': '1',
                                         'TLI_PREFILL_L2_TOPK_CAND': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        gate = patch.object(bs, '_gate_ok', return_value=True)
        gate.start()
        self.addCleanup(gate.stop)

    def test_default_fallback_reproduces_false_graph_parity(self):
        with patch.dict(os.environ, {'TLI_PREFILL_SELECT_GRAPH': '0'}):
            expected = bs.batched_group_sels(*self.args)
        self.assertIsNotNone(expected)
        with patch.object(bs, '_graph_cached_run', side_effect=RuntimeError('capture failed')):
            actual = bs.batched_group_sels(*self.args)
        self.assertIsNotNone(actual)
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_required_graph_propagates_capture_or_replay_failure(self):
        for message in ('capture failed', 'replay failed'):
            with self.subTest(message=message), patch.object(
                    bs, '_graph_cached_run', side_effect=RuntimeError(message)):
                with self.assertRaisesRegex(RuntimeError, message):
                    bs.batched_group_sels(*self.args, require_graph=True)

    def test_required_graph_rejects_disabled_mode(self):
        with patch.dict(os.environ, {'TLI_PREFILL_SELECT_GRAPH': '0'}):
            with self.assertRaisesRegex(ValueError, 'require_graph'):
                bs.batched_group_sels(*self.args, require_graph=True)

    def test_required_graph_returns_executor_result(self):
        result = object()
        with patch.object(bs, '_graph_cached_run', return_value=result) as graph:
            self.assertIs(bs.batched_group_sels(*self.args, require_graph=True), result)
        graph.assert_called_once()

    def test_required_graph_rejects_noop_and_unsupported_calls(self):
        cases = []
        empty = list(self.args); empty[4] = []; cases.append(empty)
        unaligned = list(self.args); unaligned[4] = [193]; cases.append(unaligned)
        prefix = list(self.args); prefix[5] = 129; cases.append(prefix)
        unequal = list(self.args); unequal[1] = self.q[:, :-1]; cases.append(unequal)
        heads = list(self.args); heads[1] = self.q[:, :, :3]; cases.append(heads)
        for args in cases:
            with self.subTest(ends=args[4], qshape=args[1].shape):
                with self.assertRaisesRegex(RuntimeError, 'required graph'):
                    bs.batched_group_sels(*args, require_graph=True)
        with patch.object(bs, '_gate_ok', return_value=False):
            self.assertIsNone(bs.batched_group_sels(*self.args))
            with self.assertRaisesRegex(RuntimeError, 'required graph'):
                bs.batched_group_sels(*self.args, require_graph=True)


if __name__ == '__main__':
    unittest.main()
