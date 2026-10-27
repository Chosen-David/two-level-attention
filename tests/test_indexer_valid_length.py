"""Actual CPU Indexer/TLI contract tests; not CUDA Graph validation.

CPU-only Torch cannot query CUDA stream capture. Only that query is stubbed;
TLI indexing, quantization, scores, selection and mask padding execute normally.
Run from repository root with PYTHONPATH=. python -m unittest discover -s tests
-p test_indexer_valid_length.py (also under python -O).
"""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.indexer.base import Indexer
from sparse_attn.indexer.tli_indexer import TLIIndexer


def make_indexer():
    parser = add_sparse_attn_args(argparse.ArgumentParser())
    args = parser.parse_args([
        '--method', 'tli', '--tia_block_size', '16',
        '--tia_level1_topk', '4', '--tia_level2_topk', '64',
        '--tli_near_len', '64', '--tli_near_pages', '2',
        '--tli_near_tokens', 'off', '--tli_sink_tokens', '4',
        '--tli_swa_size', '16', '--tli_enable_layer_skip', 'false',
        '--tli_enable_kmeans', 'false',
    ])
    return TLIIndexer(args)


class LegacyIndexer(Indexer):
    """Old signature with real tensor score/mask, no valid_length kwarg."""
    def prepare_index(self, k, cu):
        return k

    def compute_score(self, q, ids, k, scale):
        return torch.einsum('bqhd,bthd->bqht', q, k) * scale

    def compute_mask(self, ids, scores):
        return torch.ones_like(scores, dtype=torch.bool)

    def get_block_size(self):
        return 1


class ValidLengthTests(unittest.TestCase):
    def setUp(self):
        self.capture_query = patch('torch.cuda.is_current_stream_capturing',
                                   return_value=False)
        self.capture_query.start()
        self.addCleanup(self.capture_query.stop)
        torch.manual_seed(23)
        self.q = torch.randn(1, 1, 4, 128)
        self.k = torch.randn(1, 192, 2, 128)

    def call(self, indexer, length, k=None, query_pos=None):
        k = self.k if k is None else k
        pos = 160 if query_pos is None else query_pos
        return indexer.prepare_mask(self.q, torch.tensor([pos]), k,
                                    torch.tensor([0, k.shape[1]]),
                                    valid_length=length)

    def test_invalid_rejected_before_prepare_index(self):
        for value in (-1, 0, 193, True, False, 2.0, 2.5, '3', torch.tensor(3)):
            with self.subTest(value=value):
                indexer = make_indexer()
                with patch.object(indexer, 'prepare_index', wraps=indexer.prepare_index) as spy:
                    with self.assertRaises((TypeError, ValueError)):
                        self.call(indexer, value)
                    spy.assert_not_called()

    def test_invalid_rejected_with_existing_incremental_state(self):
        indexer = make_indexer()
        self.call(indexer, 161)
        state = indexer._inc
        with patch.object(indexer, 'prepare_index', wraps=indexer.prepare_index) as spy:
            with self.assertRaises(ValueError):
                self.call(indexer, -1)
            spy.assert_not_called()
        self.assertIs(indexer._inc, state)

    def test_full_tli_fixed_prefix_matches_variable_width(self):
        # Partial page and growth across a page boundary, with actual GQA.
        for length in (1, 4, 16, 17, 63, 64, 65, 159, 160, 161, 192):
            with self.subTest(length=length):
                original = self.k.clone()
                expected, block = self.call(make_indexer(), None,
                    self.k[:, :length].clone(), length-1)
                actual, actual_block = self.call(make_indexer(), length,
                    query_pos=length-1)
                self.assertEqual(block, actual_block)
                self.assertTrue(torch.equal(actual[..., :length], expected))
                self.assertFalse(actual[..., length:].any())
                self.assertTrue(torch.equal(self.k, original))

    def test_noncontiguous_keys_and_numpy_integer(self):
        key = torch.randn(1, 384, 2, 128)[:, ::2]
        self.assertFalse(key.is_contiguous())
        actual, _ = self.call(make_indexer(), np.int64(161), key)
        expected, _ = self.call(make_indexer(), 161, key.contiguous())
        self.assertTrue(torch.equal(actual, expected))

    def test_incremental_growth_matches_fresh_indexer(self):
        indexer = make_indexer()
        for length in (159, 160, 161, 162):
            with self.subTest(length=length):
                actual, _ = self.call(indexer, length, query_pos=length-1)
                expected, _ = self.call(make_indexer(), length, query_pos=length-1)
                self.assertTrue(torch.equal(actual, expected))

    def test_none_and_full_capacity_agree(self):
        legacy, _ = self.call(make_indexer(), None, query_pos=191)
        full, _ = self.call(make_indexer(), 192, query_pos=191)
        self.assertTrue(torch.equal(legacy, full))

    def test_legacy_signature_and_tail_clear(self):
        indexer = LegacyIndexer(None)
        q = self.q[:, :, :2]
        for length in (None, 3, 192):
            mask, _ = indexer.prepare_mask(q, torch.tensor([2]), self.k,
                torch.tensor([0, 192]), valid_length=length)
            n = 192 if length is None else length
            self.assertTrue(mask[..., :n].all())
            self.assertFalse(mask[..., n:].any())


if __name__ == '__main__':
    unittest.main()
