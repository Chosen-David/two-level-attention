"""CPU contract checks of production forward; no model weights or GPU."""
import unittest

import torch

from sparse_attn.ops.tests import test_deepseek_request_lifecycle as lifecycle


class DecodeMaskTest(unittest.TestCase):
    setUp = lifecycle.DeepseekRequestLifecycleTest.setUp
    call = lifecycle.DeepseekRequestLifecycleTest.call

    def prefix(self):
        cache = lifecycle.CacheDouble()
        self.call(torch.randn(1, 3, 8), cache)
        return cache

    def test_identity_masks_match_dense_reference(self):
        for dtype in (None, torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                cache = self.prefix()
                hidden = torch.randn(1, 1, 8)
                mask = None if dtype is None else torch.zeros(1, 1, 1, 4, dtype=dtype)
                out, _, _ = self.forward(self.attn, hidden, attention_mask=mask,
                                         past_key_value=cache, use_cache=True)
                q = self.attn.q_proj(hidden).view(1, 1, 2, 192).transpose(1, 2)
                p = (q @ cache.k.transpose(-2, -1) * self.attn.softmax_scale).softmax(-1)
                expected = (p @ cache.v).transpose(1, 2).reshape(1, 1, 256)
                torch.testing.assert_close(out, self.attn.o_proj(expected))
                self.assertEqual(cache.get_seq_length(0), 4)

    def test_bad_masks_rejected_without_mutation(self):
        masks = [torch.zeros(1, 4), torch.zeros(1, 1, 1, 3),
                 torch.zeros(1, 1, 1, 5), torch.zeros(1, 2, 1, 4),
                 torch.zeros(1, 1, 1, 4, dtype=torch.bool),
                 torch.zeros(1, 1, 1, 4, dtype=torch.int64), [0, 0, 0, 0],
                 torch.zeros(1, 1, 1, 4, device="meta"),
                 torch.zeros(1, 1, 1, 4).to_sparse()]
        for value in (-float("inf"), float("inf"), float("nan"), -1e9, 0.1, -1e-30):
            mask = torch.zeros(1, 1, 1, 4)
            mask[..., 0] = value
            masks.append(mask)
        for i, mask in enumerate(masks):
            with self.subTest(case=i):
                cache = self.prefix()
                old_k, old_v = cache.k, cache.v
                selects, clears = self.attn.indexer.selects, self.attn.indexer.clears
                with self.assertRaisesRegex((ValueError, RuntimeError), "attention_mask"):
                    self.forward(self.attn, torch.randn(1, 1, 8), attention_mask=mask,
                                 past_key_value=cache, use_cache=True)
                self.assertIs(cache.k, old_k)
                self.assertIs(cache.v, old_v)
                self.assertEqual(self.attn.indexer.selects, selects)
                self.assertEqual(self.attn.indexer.clears, clears)

    def test_invalid_decode_contract_before_cache_update(self):
        for batch, length, use_cache in ((2, 1, True), (1, 2, True), (1, 0, True), (1, 1, False)):
            with self.subTest(batch=batch, length=length, use_cache=use_cache):
                cache = self.prefix()
                old_k, old_v = cache.k, cache.v
                with self.assertRaises(RuntimeError):
                    self.forward(self.attn, torch.randn(batch, length, 8),
                                 past_key_value=cache, use_cache=use_cache)
                self.assertIs(cache.k, old_k)
                self.assertIs(cache.v, old_v)


if __name__ == "__main__":
    unittest.main()
