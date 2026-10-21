"""Real CPU tensors: incremental index metadata and full-rebuild equivalence.

These tests exercise the Python fallback, not CUDA kernels or model output.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sparse_attn.indexer import TLIIndexer
import sparse_attn.indexer.tli_indexer as tli_module


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

    def test_tail_rebuild_excludes_complete_pages_at_boundaries(self):
        # Includes partial pages, exact boundaries, and multi-page appends.
        # Observe the actual statistics input, not a duplicate formula helper.
        for width in ("same", "full"):
            for quantized in (False, True):
                for previous in (63, 64, 65, 128):
                    for added in (1, 81):
                        with self.subTest(l2=width, quantized=quantized,
                                          previous=previous, added=added):
                            args = make_args(tli_l2_subspace=width,
                                             tli_l2_quant=quantized)
                            indexer = TLIIndexer(args)
                            key = self.keys()[:, :previous + added]
                            self.prepare(indexer, key[:, :previous])
                            with patch.object(tli_module, "rearrange",
                                              wraps=tli_module.rearrange) as reshape, patch.object(
                                    indexer, "min_max_per_token_quant",
                                    wraps=indexer.min_max_per_token_quant) as quant:
                                actual = self.prepare(indexer, key)
                            tails = [call.args[0].shape[1] for call in reshape.call_args_list
                                     if call.args[1] == "b (t bs) h d -> b t bs h d"]
                            expected_tokens = previous % 64 + added
                            self.assertEqual(tails, [((expected_tokens + 63) // 64) * 64])
                            if quantized:
                                quant.assert_called_once()
                                self.assertEqual(quant.call_args.args[0].shape[0],
                                                 expected_tokens)
                            else:
                                quant.assert_not_called()
                            self.assert_matches_fresh(actual, key, args)

    def test_current_pooled_selector_scores_and_masks_match_fresh(self):
        for previous in (511, 512):
            with self.subTest(previous=previous):
                args = make_args(tia_level1_topk=4, tia_level2_topk=256,
                                 tli_near_pages=2, tli_near_len=128,
                                 tli_near_tokens="off", tli_sink_tokens=4,
                                 tli_swa_size=128)
                key = torch.randn(1, previous + 17, 4, 128,
                                  generator=self.generator)
                query = torch.randn(1, 1, 8, 128, generator=self.generator)
                q_ids = torch.tensor([key.shape[1] - 1])
                incremental, fresh = TLIIndexer(args), TLIIndexer(args)
                self.prepare(incremental, key[:, :previous])
                outputs = []
                for indexer in (incremental, fresh):
                    scores = indexer.compute_score(query, q_ids,
                                                   self.prepare(indexer, key),
                                                   128 ** -.5)
                    mask = indexer.compute_mask(q_ids, scores)
                    outputs.append((scores, mask))
                for name in ("score_coarse", "score_fine", "score_avg"):
                    a, b = (output[0].get(name) for output in outputs)
                    if a is not None:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    else:
                        self.assertIsNone(b)
                self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))

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
