"""CPU tensor regressions; a launch spy checks routing, not GPU execution."""
import os
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import eager_decoding as ed


class HeterogeneousKVDispatchTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TLI_E127_FAST": "1", "TLI_E127_W_UPPER": "32"})
        self.env.start()
        self.flags = patch.object(ed, "_ERR_FLAG", {})
        self.flags.start()

    def tearDown(self):
        self.flags.stop()
        self.env.stop()

    def inputs(self, dk=64, dv=32, layout="contiguous", per_qh=False, length=16384):
        gen = torch.Generator().manual_seed(142)
        q = torch.randn(1, 1, 4, dk, generator=gen)
        k = torch.randn(1, length, 2, dk, generator=gen)
        v = torch.randn(1, length, 2, dv, generator=gen)
        if layout == "token_gap":
            v = torch.randn(1, 2 * length, 2, dv, generator=gen)[:, ::2]
        elif layout == "head_major":
            v = torch.randn(1, 2, length, dv, generator=gen).transpose(1, 2)
        mask = torch.zeros(1, 1, 4 if per_qh else 2, length, dtype=torch.bool)
        for row in range(mask.shape[-2]):
            mask[0, 0, row, [0, 3 + row, length - 1]] = True
        return q, k, v, mask, 1, torch.tensor([0, length], dtype=torch.int64)

    def reference(self, inputs):
        q, k, v, mask, _, _ = inputs
        # Direct selected-token reference; no production packing/dispatch logic.
        rows = []
        for h in range(q.shape[-2]):
            kvh = h // 2
            mask_row = h if mask.shape[-2] == q.shape[-2] else kvh
            chosen = mask[0, 0, mask_row].nonzero().flatten()
            logits = (k[0, chosen, kvh] @ (q[0, 0, h] * k.shape[-1] ** -.5))
            rows.append(logits.softmax(-1) @ v[0, chosen, kvh])
        return torch.stack(rows)[None, None]

    def test_incompatible_kv_never_reaches_fused_launch(self):
        for dk, dv, layout in [(64, 32, "contiguous"), (64, 128, "contiguous"),
                              (64, 64, "token_gap"), (64, 64, "head_major"),
                              (192, 128, "contiguous")]:
            for per_qh in (False, True):
                with self.subTest(dk=dk, dv=dv, layout=layout, per_qh=per_qh):
                    inputs = self.inputs(dk, dv, layout, per_qh)
                    with patch.object(ed, "_HAS_TRITON", True), patch.object(
                        ed, "_EAGER_TRITON_OK", {"ok": True}), patch.object(
                        ed, "_eager_gather_triton", return_value=None) as launch:
                        out = ed.eager_decoding_attn(*inputs, allow_kv_hetero=True)
                    self.assertEqual(launch.call_count, 0, "incompatible K/V reached fused launch")
                    torch.testing.assert_close(out, self.reference(inputs), atol=1e-5, rtol=1e-5)
                    ed.eager_decoding_check_errors()

    def test_equal_layout_keeps_fused_route(self):
        for allow in (False, True):
            for per_qh in (False, True):
                for layout in ("contiguous", "token_gap", "head_major"):
                    with self.subTest(allow=allow, per_qh=per_qh, layout=layout):
                        inputs = list(self.inputs(64, 64, layout, per_qh))
                        # Same strided K/V ABI is supported, not just contiguous.
                        inputs[1] = torch.empty_strided(inputs[2].shape, inputs[2].stride())
                        sentinel = object()
                        with patch.object(ed, "_HAS_TRITON", True), patch.object(
                            ed, "_EAGER_TRITON_OK", {"ok": True}), patch.object(
                            ed, "_eager_gather_triton", return_value=sentinel) as launch:
                            out = ed.eager_decoding_attn(*inputs, allow_kv_hetero=allow)
                        self.assertIs(out, sentinel)
                        launch.assert_called_once()

    def test_default_contract_still_rejects_heterogeneous_kv(self):
        for dk, dv, layout in [(64, 32, "contiguous"), (64, 64, "token_gap")]:
            with self.subTest(layout=layout):
                with self.assertRaises(ValueError):
                    ed.eager_decoding_attn(*self.inputs(dk, dv, layout))

    def test_direct_fused_wrapper_rejects_before_allocation(self):
        q, k, v, mask, _, cu = self.inputs()
        with patch.object(ed.torch, "full", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "K/V"):
                ed._eager_gather_triton(q[0], k[0], v[0], mask[0, 0], False, 2, 32, cu, 1.)

    def test_short_dense_fallback_supports_mla_dimensions(self):
        for fast in ("0", "1"):
            with self.subTest(fast=fast), patch.dict(os.environ, {"TLI_E127_FAST": fast}):
                inputs = self.inputs(192, 128, length=11)
                out = ed.eager_decoding_attn(*inputs, allow_kv_hetero=True)
                torch.testing.assert_close(out, self.reference(inputs), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
