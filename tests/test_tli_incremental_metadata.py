"""Real CPU tensors: incremental index metadata and full-rebuild equivalence.

These tests exercise the Python fallback, not CUDA kernels or model output.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sparse_attn.indexer import TLIIndexer


def make_args(**overrides):
    values = dict(tia_block_size=64, tia_level1_topk=128,
                  tia_level2_topk=1024, tia_level2_cmp_ratio=4,
                  tia_enable_async_topk=False, tli_enable_layer_skip=False,
                  tli_enable_kmeans=False, tli_subspace="tail",
                  tli_l2_subspace="same", tli_l2_quant=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class IncrementalMetadataTensorTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(621)

    def keys(self, heads=4, dtype=torch.float32):
        return torch.randn(1, 257, heads, 128,
                           generator=self.generator, dtype=dtype)

    def prepare(self, indexer, key):
        return indexer.prepare_index(key, torch.tensor([0, key.shape[1]]))

    def assert_matches_fresh(self, actual, key, args):
        expected = self.prepare(TLIIndexer(args), key)
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
            else:
                self.assertEqual(actual[name], value)

    def test_head_count_changes_rebuild_instead_of_broadcast_or_shape_error(self):
        for old_heads, new_heads in ((4, 1), (1, 4)):
            with self.subTest(old=old_heads, new=new_heads):
                args = make_args()
                indexer = TLIIndexer(args)
                self.prepare(indexer, self.keys(old_heads)[:, :256])
                key = self.keys(new_heads)
                self.assert_matches_fresh(self.prepare(indexer, key), key, args)

    def test_dtype_change_does_not_keep_old_precision_buffers(self):
        args = make_args()
        indexer = TLIIndexer(args)
        key = self.keys()
        self.prepare(indexer, key[:, :256])
        key = key.to(torch.float64)
        self.assert_matches_fresh(self.prepare(indexer, key), key, args)

    def test_page_size_change_rebuilds_coarse_statistics(self):
        args = make_args()
        indexer = TLIIndexer(args)
        key = self.keys()
        self.prepare(indexer, key[:, :256])
        args.tia_block_size = 32
        self.assert_matches_fresh(self.prepare(indexer, key), key, args)

    def test_same_metadata_growth_keeps_incremental_path_and_exact_values(self):
        for width in ("same", "full"):
            for quantized in (False, True):
                with self.subTest(l2=width, quantized=quantized):
                    args = make_args(tli_l2_subspace=width,
                                     tli_l2_quant=quantized)
                    indexer = TLIIndexer(args)
                    key = self.keys()
                    self.prepare(indexer, key[:, :256])
                    with patch.object(indexer, "_prepare_index_inc",
                                      wraps=indexer._prepare_index_inc) as inc:
                        actual = self.prepare(indexer, key)
                    inc.assert_called_once()
                    self.assert_matches_fresh(actual, key, args)

    def test_clear_discards_previous_request_cache(self):
        args = make_args()
        indexer = TLIIndexer(args)
        self.prepare(indexer, self.keys()[:, :256])
        indexer.clear()
        key = self.keys()
        with patch.object(indexer, "_prepare_index_inc",
                          wraps=indexer._prepare_index_inc) as inc:
            actual = self.prepare(indexer, key)
        inc.assert_not_called()
        self.assert_matches_fresh(actual, key, args)


if __name__ == "__main__":
    unittest.main()
