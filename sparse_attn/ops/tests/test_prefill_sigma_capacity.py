"""Variable-cardinality sigma selections must not be truncated to a K2 cap."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sparse_attn.indexer import TLIIndexer
from sparse_attn.ops.eager_prefill import sparse_prefill_attn


def environment(runtime='candidate', group=0):
    return patch.dict(os.environ, {
        'TLI_PREFILL_RUNTIME': runtime, 'TLI_PREFILL_SELECT_GROUP': str(group),
        'TLI_PREFILL_PREFETCH': '0', 'TLI_PREFILL_BATCH_SELECT': '0',
        'TLI_PREFILL_FUSED_ATTN': '0',
    }, clear=True)


def make_index(mode):
    return TLIIndexer(SimpleNamespace(
        tia_block_size=64, tia_level1_topk=8, tia_level2_topk=256,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        tli_enable_layer_skip=False, tli_enable_kmeans=False,
        tli_subspace='tail', tli_sink_tokens=64, tli_swa_size=64,
        tli_alpha=.5, tli_beta=.5, tli_gamma=.5,
        tli_sigma_select=mode, tli_sigma=8.,
    ))


class PrefillSigmaCapacityTest(unittest.TestCase):
    def setUp(self):
        # These fixtures allocate CPU tensors only. CPU-only Torch cannot
        # query CUDA stream capture; indexing/scoring/masks remain real.
        capture = patch('torch.cuda.is_current_stream_capturing', return_value=False)
        capture.start()
        self.addCleanup(capture.stop)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_real_sigma_masks_match_absolute_position_oracle(self):
        length = 1025
        for mode in ('far', 'near', 'mid'):
            for dtype in (torch.float32, torch.bfloat16):
                for group in (0, 64):
                    with self.subTest(mode=mode, dtype=dtype, group=group), environment(group=group):
                        q = torch.zeros(1, length * 2, 4, 128, dtype=dtype)[:, ::2]
                        k = torch.zeros(1, length * 2, 2, 128, dtype=dtype)[:, ::2]
                        v = (torch.arange(length, dtype=torch.float32) / length).to(dtype)
                        v = v.view(1, length, 1, 1).expand_as(k)
                        idx = make_index(mode)
                        original, selections = idx.prepare_mask, {}

                        def record(*args):
                            mask, block = original(*args)
                            selections[args[2].shape[1]] = mask.clone()
                            return mask, block

                        idx.prepare_mask = record
                        actual = sparse_prefill_attn(idx, q, k, v, chunk_size=128, dense_below=128)
                        expected = torch.empty_like(actual)
                        for row in range(length):
                            end = min(128 + ((row - 128) // (group or 128) + 1) * (group or 128), length)
                            for h in range(4):
                                if row < 128:
                                    positions = torch.arange(row + 1)
                                else:
                                    allowed = selections[end][0, 0, h // 2, :row + 1].clone()
                                    allowed[:64] = True
                                    allowed[max(0, row - 63):] = True
                                    positions = allowed.nonzero().flatten()
                                # Q=0: attention is the exact visible-value mean.
                                expected[0, row, h] = v[0, positions, h // 2].float().mean(0).to(dtype)
                        tol = 0.004 if dtype == torch.bfloat16 else 2e-6
                        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_default_outer_2048_preserves_full_sigma_support(self):
        length = 4097
        q = torch.zeros(1, length, 2, 128)
        k = torch.zeros(1, length, 1, 128)
        v = (torch.arange(length) / length).view(1, length, 1, 1).expand_as(k)
        with environment():
            actual = sparse_prefill_attn(make_index('mid'), q, k, v)
        expected = (torch.arange(length) / (2 * length)).view(1, length, 1, 1).expand_as(q)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_only_sigma_disables_static_capacity(self):
        q, k = torch.zeros(1, 7, 2, 4), torch.zeros(1, 7, 1, 4)
        for mode in (None, 'none', 'far', 'near', 'mid'):
            with self.subTest(mode=mode), environment():
                idx = SimpleNamespace(
                    args=SimpleNamespace(tia_level2_topk=2, tia_block_size=1),
                    sink_tokens=1, sliding_window_size=1,
                    prepare_mask=lambda q, ids, k, cu, scale: (
                        torch.zeros(1, 1, 1, k.shape[1], dtype=torch.bool), 1),
                )
                if mode is not None:
                    idx.sigma_select = mode
                caps = []

                def capture(qc, *args, **kwargs):
                    caps.append(kwargs['w_sel_cap'])
                    return torch.zeros_like(qc)

                with patch('sparse_attn.ops.eager_prefill._chunk_rows_attn', side_effect=capture):
                    sparse_prefill_attn(idx, q, k, k, chunk_size=4, dense_below=0)
                self.assertEqual(caps, [6, 6] if mode in (None, 'none') else [0, 0])


if __name__ == '__main__':
    unittest.main()

