"""Oracle budgets require an explicit, covered layer when prefill consumes them."""
import os
import unittest
from unittest.mock import patch

import torch

from test_e151c_prefill_routing import make_index
from sparse_attn.ops import eager_prefill


class QuotaLayerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'TLI_E151C_ORACLE_QUOTA': '', 'TLI_PREFILL_SELECT_GROUP': '0',
            'TLI_PREFILL_PREFETCH': '0', 'TLI_PREFILL_BATCH_SELECT': '0',
            'TLI_PREFILL_FUSED_ATTN': '0', 'TLI_PREFILL_SELECT_GRAPH': '0',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        gen = torch.Generator().manual_seed(15120)
        self.q = torch.randn(1, 512, 2, 128, generator=gen)
        self.k = torch.randn(1, 512, 1, 128, generator=gen)
        self.cu = torch.tensor([0, 512], dtype=torch.int32)

    def selector(self, layer, active=True):
        idx = make_index()
        idx.layer_idx = layer
        idx._e151c_quota = [0, 32, 128]
        idx._e151c_on = active
        return idx

    def mask(self, idx):
        return idx.prepare_mask(self.q[:, -1:], self.cu[1:] - 1,
                                self.k, self.cu, 128 ** -.5)[0]

    def test_missing_invalid_and_uncovered_layer_rejected(self):
        for layer in (None, -1, True, False, 1.0, '1', 3):
            with self.subTest(layer=layer):
                with self.assertRaisesRegex(ValueError, 'E151c.*layer_idx'):
                    self.mask(self.selector(layer))

    def test_invalid_layer_rejected_before_score_access(self):
        for layer in (None, -1, 3):
            with self.subTest(layer=layer):
                with self.assertRaisesRegex(ValueError, 'E151c.*layer_idx'):
                    self.selector(layer).compute_mask(None, {})

    def test_each_valid_layer_uses_its_own_quota(self):
        for layer, quota in enumerate((0, 32, 128)):
            with self.subTest(layer=layer):
                mask = self.mask(self.selector(layer))
                self.assertEqual(mask[..., 64:256].sum().item(), quota)
                self.assertEqual(mask.sum().item(), 256)
                self.assertTrue(mask[..., :64].all())
                self.assertTrue(mask[..., 448:512].all())

    def test_decode_does_not_require_prefill_quota_layer(self):
        expected = self.mask(self.selector(0, active=False))
        for layer in (None, -1, 3):
            with self.subTest(layer=layer):
                self.assertTrue(torch.equal(expected, self.mask(self.selector(layer, active=False))))

    def test_prefill_failure_clears_activation_flag(self):
        for runtime in ('reference', 'candidate'):
            idx = self.selector(None, active=False)
            with self.subTest(runtime=runtime), patch.dict(os.environ, TLI_PREFILL_RUNTIME=runtime):
                with self.assertRaisesRegex(ValueError, 'E151c.*layer_idx'):
                    eager_prefill.sparse_prefill_attn(idx, self.q, self.k, self.k, dense_below=0)
                self.assertFalse(idx._e151c_on)


if __name__ == '__main__':
    unittest.main()
