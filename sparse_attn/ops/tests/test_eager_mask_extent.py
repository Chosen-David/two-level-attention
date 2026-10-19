"""A decode block mask must cover K/V before any broadcasting or dispatch."""
import itertools
import os
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import eager_decoding as ed


class MaskExtentTest(unittest.TestCase):
    def inputs(self, length, width, block, rows=1, hetero=False):
        q = torch.zeros(1, 1, 2, 4)
        k = torch.zeros(1, length, 1, 4)
        vd = 3 if hetero else 4
        v = torch.arange(length, dtype=torch.float32).view(1, length, 1, 1).expand(
            1, length, 1, vd).contiguous()
        mask = torch.ones(1, 1, rows, width, dtype=torch.bool)
        return q, k, v, mask, block, torch.tensor([0, length])

    def test_single_slot_must_not_broadcast_to_all_keys(self):
        # Previously returns mean([0,1,2]) == 1 for a one-token selection
        # mask: the token dimension silently broadcasts across all three keys.
        for fast, rows, hetero in itertools.product(('0', '1'), (1, 2), (False, True)):
            with self.subTest(fast=fast, rows=rows, hetero=hetero), patch.dict(
                    os.environ, {'TLI_E127_FAST': fast}):
                with self.assertRaisesRegex(ValueError, 'mask.*cover'):
                    ed.eager_decoding_attn(*self.inputs(3, 1, 1, rows, hetero),
                                          allow_kv_hetero=hetero)

    def test_short_masks_rejected_at_all_entry_points(self):
        for fn, spec in itertools.product(
                (ed.eager_decoding_attn, ed._eager_decoding_attn_dense,
                 ed._eager_decoding_attn_gather),
                ((3, 0, 1), (3, 1, 1), (5, 1, 2), (65, 1, 64), (16385, 1, 1))):
            with self.subTest(fn=fn.__name__, spec=spec):
                with self.assertRaisesRegex(ValueError, 'mask.*cover'):
                    fn(*self.inputs(*spec))

    def test_invalid_extent_rejected_before_dispatch(self):
        args = self.inputs(3, 1, 1)
        with patch.object(ed, '_eager_decoding_attn_dense') as dense, patch.object(
                ed, '_eager_decoding_attn_gather') as gather:
            with self.assertRaisesRegex(ValueError, 'mask.*cover'):
                ed.eager_decoding_attn(*args)
            dense.assert_not_called()
            gather.assert_not_called()

    def test_complete_and_padded_block_masks_keep_selection(self):
        for fast, length, block, rows, hetero in itertools.product(
                ('0', '1'), (1, 3, 5, 65), (1, 2, 64), (1, 2), (False, True)):
            width = (length + block - 1) // block
            for extra in (0, 2):
                args = list(self.inputs(length, width + extra, block, rows, hetero))
                args[3].zero_()
                args[3][..., 0] = True
                args[3][..., width:] = True  # Padding is cropped, never attended.
                expected = torch.full((1, 1, 2, 3 if hetero else 4),
                                      (min(block, length) - 1) / 2)
                with self.subTest(fast=fast, length=length, block=block, rows=rows,
                                  hetero=hetero, extra=extra), patch.dict(
                                      os.environ, {'TLI_E127_FAST': fast}):
                    actual = ed.eager_decoding_attn(*args, allow_kv_hetero=hetero)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
