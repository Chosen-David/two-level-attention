"""CPU probes of the real fused wrapper's query ABI, not GPU execution.

The launch double reads Q with the kernel's actual address formula. It echoes
that operand instead of computing attention, isolating layout from numerics.
"""
import os
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import eager_decoding as ed


class KernelDouble:
    def __init__(self, run):
        self.run = run
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return self.run


class EagerQueryLayoutTest(unittest.TestCase):
    def invoke(self, q, public=False):
        h, d, length = q.shape[-2], q.shape[-1], 16384 if public else 7
        k = torch.zeros(length, 2, d, dtype=q.dtype)
        v = torch.zeros_like(k)
        m = torch.ones(2, length, dtype=torch.bool)
        cu = torch.tensor([0, length], dtype=torch.int32)
        observed = []

        def query_abi(idx, count, qs, keys, values, out, kt, kh, qh, oh, **meta):
            observed.append(qs)
            # Exactly Q + qh * stride_qh + offs_d in _eager_attn_kernel.
            # This is an ABI address probe, NOT a CPU replacement for Triton.
            read = qs.as_strided((h, d), (qh, 1))
            out.copy_(read)

        expand = KernelDouble(lambda *args, **kwargs: None)
        attn = KernelDouble(query_abi)
        with patch.object(ed, '_eager_expand_kernel', expand, create=True), patch.object(
                ed, '_eager_attn_kernel', attn, create=True), patch.object(
                ed, '_ERR_FLAG', {}), patch.object(ed, '_HAS_TRITON', True), patch.object(
                ed, '_EAGER_TRITON_OK', {'ok': True}), patch.dict(
                os.environ, {'TLI_E127_FAST': '1', 'TLI_E127_W_UPPER': '32'}):
            if public:
                out = ed.eager_decoding_attn(q[None], k[None], v[None],
                                           m[None, None], 1, cu, softmax_scale=.3)
            else:
                out = ed._eager_gather_triton(q, k, v, m, False, h // 2, 32, cu, .3)
        self.assertEqual(len(observed), 1, 'fused wrapper was not reached')
        return out, observed[0]

    def test_transposed_query_operand_matches_kernel_unit_stride_abi(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for public in (False, True):
                with self.subTest(dtype=dtype, public=public):
                    q = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
                    q = q.transpose(-1, -2).to(dtype)
                    self.assertEqual(q.stride(-1), 4)
                    out, qs = self.invoke(q, public)
                    self.assertEqual(qs.stride(-1), 1)
                    expected = (q[0] * .3).to(torch.float32).to(dtype)
                    torch.testing.assert_close(out, expected[None, None], atol=0, rtol=0)

    def test_gapped_query_and_storage_offset_remain_supported(self):
        q = torch.arange(256, dtype=torch.float32).reshape(1, 8, 32)[:, 1::2, 3:19:2]
        self.assertGreater(q.storage_offset(), 0)
        self.assertEqual(q.stride(-1), 2)
        out, qs = self.invoke(q)
        self.assertEqual(qs.stride(-1), 1)
        torch.testing.assert_close(out, (q[0] * .3)[None, None], atol=0, rtol=0)

    def test_standard_query_does_not_request_a_contiguous_copy(self):
        q = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        with patch.object(torch.Tensor, 'contiguous', side_effect=AssertionError('extra copy')):
            out, qs = self.invoke(q)
        self.assertEqual(qs.stride(-1), 1)
        torch.testing.assert_close(out, (q[0] * .3)[None, None], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
