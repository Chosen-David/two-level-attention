"""CPU contract checks of the real MLA forward; no model weights or GPU.

Identity RoPE, cache and all-selected indexer doubles isolate rejection and
state semantics. This is not production selector/kernel validation.
"""
import unittest
from unittest.mock import patch

import torch

from sparse_attn.ops.tests import test_deepseek_request_lifecycle as lifecycle


class DecodeMaskTest(unittest.TestCase):
    setUp = lifecycle.DeepseekRequestLifecycleTest.setUp
    call = lifecycle.DeepseekRequestLifecycleTest.call

    def prefix(self):
        cache = lifecycle.CacheDouble()
        self.call(torch.randn(1, 3, 8), cache)
        return cache

    def assert_rejected_without_mutation(self, mask):
        cache = self.prefix()
        old_k, old_v = cache.k, cache.v
        k_bytes, v_bytes = old_k.clone(), old_v.clone()
        counts = self.attn.indexer.clears, self.attn.indexer.selects
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            self.forward(self.attn, torch.randn(1, 1, 8), attention_mask=mask,
                         past_key_value=cache, use_cache=True)
        self.assertIs(cache.k, old_k)
        self.assertIs(cache.v, old_v)
        torch.testing.assert_close(cache.k, k_bytes, rtol=0, atol=0)
        torch.testing.assert_close(cache.v, v_bytes, rtol=0, atol=0)
        self.assertEqual(cache.get_seq_length(0), 3)
        self.assertEqual((self.attn.indexer.clears, self.attn.indexer.selects), counts)

    def test_nonidentity_values_rejected_before_cache_update(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for value in (-float("inf"), float("inf"), float("nan"),
                          -1e4, .125, torch.finfo(dtype).tiny):
                with self.subTest(dtype=dtype, value=value):
                    mask = torch.zeros(1, 1, 1, 4, dtype=dtype)
                    mask[..., 0] = value
                    self.assert_rejected_without_mutation(mask)

    def test_non_tensor_and_sparse_masks_rejected_before_cache_update(self):
        for mask in ([0., 0., 0., 0.], torch.zeros(1, 1, 1, 4).to_sparse()):
            with self.subTest(type=type(mask), layout=getattr(mask, "layout", None)):
                self.assert_rejected_without_mutation(mask)

    def test_identity_masks_match_dense_reference(self):
        for dtype in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                cache = self.prefix()
                hidden = torch.randn(1, 1, 8)
                # Negative zero is an identity too; use a noncontiguous mask.
                mask = None if dtype is None else torch.full(
                    (1, 1, 1, 8), -0., dtype=dtype)[..., ::2]
                out, _, _ = self.forward(self.attn, hidden, attention_mask=mask,
                                         past_key_value=cache, use_cache=True)
                q = self.attn.q_proj(hidden).view(1, 1, 2, 192).transpose(1, 2)
                p = (q @ cache.k.transpose(-2, -1) * self.attn.softmax_scale).softmax(-1)
                expected = (p @ cache.v).transpose(1, 2).reshape(1, 1, 256)
                torch.testing.assert_close(out, self.attn.o_proj(expected))
                self.assertEqual(cache.get_seq_length(0), 4)

    def test_none_decode_does_not_reduce_mask_values(self):
        cache = self.prefix()
        with patch("torch.all", side_effect=AssertionError("unexpected value check")):
            self.forward(self.attn, torch.randn(1, 1, 8), attention_mask=None,
                         past_key_value=cache, use_cache=True)

    def test_prefill_keeps_nonzero_additive_mask_semantics(self):
        hidden = torch.randn(1, 3, 8)
        mask = torch.zeros(1, 1, 3, 3).masked_fill(
            torch.ones(3, 3, dtype=torch.bool).triu(1), -float("inf"))
        mask[..., 0] += .25
        cache = lifecycle.CacheDouble()
        out, weights, _ = self.forward(self.attn, hidden, attention_mask=mask,
                                       past_key_value=cache, use_cache=True,
                                       output_attentions=True)
        q = self.attn.q_proj(hidden).view(1, 3, 2, 192).transpose(1, 2)
        p = (q @ cache.k.transpose(-2, -1) * self.attn.softmax_scale + mask).softmax(-1)
        torch.testing.assert_close(weights, p)
        expected = (p @ cache.v).transpose(1, 2).reshape(1, 3, 256)
        torch.testing.assert_close(out, self.attn.o_proj(expected))

    def test_retry_after_rejected_mask_matches_untouched_cache(self):
        prefix, token = torch.randn(1, 3, 8), torch.randn(1, 1, 8)
        actual_cache, expected_cache = lifecycle.CacheDouble(), lifecycle.CacheDouble()
        self.call(prefix, actual_cache)
        self.call(prefix, expected_cache)
        mask = torch.zeros(1, 1, 1, 4)
        mask[..., 0] = -float("inf")
        with self.assertRaisesRegex(ValueError, "all zero"):
            self.forward(self.attn, token, attention_mask=mask,
                         past_key_value=actual_cache, use_cache=True)
        actual, _, _ = self.call(token, actual_cache)
        expected, _, _ = self.call(token, expected_cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
