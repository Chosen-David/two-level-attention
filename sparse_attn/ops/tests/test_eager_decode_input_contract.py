"""Decode-only consumer contract: reject before dispatch or tensor allocation."""
import os
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import eager_decoding as ed


class DecodeInputContractTest(unittest.TestCase):
    def inputs(self):
        return [torch.ones(1, 1, 2, 4), torch.ones(1, 3, 1, 4),
                torch.ones(1, 3, 1, 4), torch.ones(1, 1, 1, 3, dtype=torch.bool),
                1, torch.tensor([0, 3])]

    def test_multi_query_rejected_instead_of_silent_zero_tail(self):
        # Baseline returns first query = ones and second query = zeros.
        # The selected-token reference is ones for BOTH queries, but this
        # decode-only API must reject the unsupported multi-query call.
        for fast in ('0', '1'):
            args = self.inputs()
            args[0] = torch.ones(1, 2, 2, 4)
            args[3] = torch.ones(1, 2, 1, 3, dtype=torch.bool)
            with self.subTest(fast=fast), patch.dict(os.environ, {'TLI_E127_FAST': fast}):
                with self.assertRaisesRegex(ValueError, 'single-query'):
                    ed.eager_decoding_attn(*args)

    def test_all_entry_points_reject_before_dispatch_or_allocation(self):
        invalid = []
        for slot, shape in [(0, (1, 2, 2, 4)), (0, (2, 1, 2, 4)),
                            (0, (1, 2, 4)), (1, (2, 3, 1, 4)),
                            (2, (2, 3, 1, 4)), (3, (1, 2, 1, 3)),
                            (3, (2, 1, 1, 3)), (3, (1, 1, 3))]:
            args = self.inputs()
            args[slot] = torch.ones(shape, dtype=args[slot].dtype)
            invalid.append(args)
        for fn in (ed.eager_decoding_attn, ed._eager_decoding_attn_dense,
                   ed._eager_decoding_attn_gather):
            for args in invalid:
                with self.subTest(fn=fn.__name__, shapes=[a.shape for a in args[:4]]):
                    with patch.object(torch, 'empty', side_effect=AssertionError('allocation')):
                        with self.assertRaisesRegex(ValueError, 'single-query'):
                            fn(*args)

    def test_valid_shared_and_per_query_head_masks_preserved(self):
        for fast in ('0', '1'):
            for rows in (1, 2):
                args = self.inputs()
                args[3] = torch.ones(1, 1, rows, 3, dtype=torch.bool)
                with self.subTest(fast=fast, rows=rows), patch.dict(
                        os.environ, {'TLI_E127_FAST': fast}):
                    torch.testing.assert_close(ed.eager_decoding_attn(*args), args[0])

    def test_qk_width_mismatch_never_reaches_fused_launch(self):
        for width in (32, 128):
            args = [torch.ones(1, 1, 2, width), torch.ones(1, 16384, 1, 64),
                    torch.ones(1, 16384, 1, 64),
                    torch.ones(1, 1, 1, 16384, dtype=torch.bool),
                    1, torch.tensor([0, 16384])]
            with self.subTest(width=width), patch.dict(
                    os.environ, {'TLI_E127_FAST': '1', 'TLI_E127_W_UPPER': '32'}):
                with patch.object(ed, '_HAS_TRITON', True), patch.object(
                        ed, '_EAGER_TRITON_OK', {'ok': True}), patch.object(
                        ed, '_eager_gather_triton', return_value=object()) as launch:
                    with self.assertRaisesRegex(ValueError, 'Q/K head_dim'):
                        ed.eager_decoding_attn(*args)
                    launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
