"""Host dispatch and CPU fallback tests, not CUDA selector validation."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sparse_attn.indexer import TLIIndexer
from sparse_attn.ops import batch_select as batch, eager_prefill


class CudaMetadata:
    shape = (1, 256, 2, 128)
    is_cuda = True
    device = torch.device('cpu')  # Metadata-only host dispatch; no CUDA execution.
    dtype = torch.float32

    def dim(self):
        return 4


def gate_index(sink):
    return SimpleNamespace(
        args=SimpleNamespace(tia_block_size=64), _triton_ok=True,
        far_select='4bit', near_select='4bit', alpha=.25, beta=.25, gamma=0.,
        per_q_head=False, sigma_select='none', moba_gate=False,
        sinkswa_only=False, enable_async=False, _basis=None, _basis_all=None,
        l2_subspace='same', enable_subspace=True, subspace='tail',
        static_pair=False, sink_tokens=sink, sliding_window_size=64,
    )


class BatchSinkGuardTest(unittest.TestCase):
    def setUp(self):
        # These fixtures allocate CPU tensors only. CPU-only Torch cannot
        # query CUDA stream capture; indexing/scoring/masks remain real.
        capture = patch('torch.cuda.is_current_stream_capturing', return_value=False)
        capture.start()
        self.addCleanup(capture.stop)

    def test_zero_sink_is_not_a_protected_scatter_sentinel(self):
        self.assertFalse(batch._gate_ok(gate_index(0), CudaMetadata(), CudaMetadata()))

    def test_block_aligned_protected_sinks_remain_supported(self):
        for sink in (64, 128, 192):
            with self.subTest(sink=sink):
                self.assertTrue(batch._gate_ok(gate_index(sink), CudaMetadata(), CudaMetadata()))
        for sink in (1, 4, 63, 65):
            with self.subTest(sink=sink):
                self.assertFalse(batch._gate_ok(gate_index(sink), CudaMetadata(), CudaMetadata()))

    def test_public_batch_returns_fallback_before_tensor_work(self):
        with patch.object(torch, 'tensor', side_effect=AssertionError('tensor work')):
            self.assertIsNone(batch.batched_group_sels(
                gate_index(0), CudaMetadata(), CudaMetadata(), 1., [256], 128))

    def test_required_graph_does_not_hide_unsupported_configuration(self):
        with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='1'), patch.object(
                torch, 'tensor', side_effect=AssertionError('tensor work')):
            with self.assertRaisesRegex(RuntimeError, 'unsupported selector'):
                batch.batched_group_sels(gate_index(0), CudaMetadata(),
                                         CudaMetadata(), 1., [256], 128, require_graph=True)

    def test_real_zero_sink_grouped_fallback_matches_sequential(self):
        gen = torch.Generator().manual_seed(190355)
        q = torch.randn(1, 320, 2, 128, generator=gen)
        k = torch.randn(1, 320, 1, 128, generator=gen)
        v = torch.randn(1, 320, 1, 128, generator=gen)
        results, selections = [], []
        env = dict(TLI_E151C_ORACLE_QUOTA='', TLI_PREFILL_RUNTIME='candidate',
                   TLI_PREFILL_SELECT_GROUP='64', TLI_PREFILL_PREFETCH='0',
                   TLI_PREFILL_FUSED_ATTN='0', TLI_PREFILL_SELECT_GRAPH='0')
        for enabled in ('0', '1'):
            with patch.dict(os.environ, dict(env, TLI_PREFILL_BATCH_SELECT=enabled), clear=True):
                idx = TLIIndexer(SimpleNamespace(
                    tia_block_size=64, tia_level1_topk=4, tia_level2_topk=96,
                    tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
                    tli_enable_layer_skip=False, tli_enable_kmeans=False,
                    tli_subspace='tail', tli_sink_tokens=0, tli_swa_size=64,
                    tli_alpha=.25, tli_beta=.25, tli_gamma=0.))
                masks = []
                original = idx.prepare_mask
                def record(*args, **kwargs):
                    result = original(*args, **kwargs)
                    masks.append(result[0].clone())
                    return result
                with patch.object(idx, 'prepare_mask', side_effect=record):
                    results.append(eager_prefill.sparse_prefill_attn(
                        idx, q, k, v, chunk_size=2048, dense_below=128))
                selections.append(masks)
        self.assertEqual(len(selections[0]), 3)
        self.assertEqual(len(selections[1]), 3)
        for left, right in zip(*selections):
            self.assertTrue(torch.equal(left, right))
        torch.testing.assert_close(*results, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()

