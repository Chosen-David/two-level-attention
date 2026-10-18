"""Real wrapper/address-ABI probes, not execution of CUDA/Triton kernels."""
import os
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import eager_decoding as ed


class KernelDouble:
    def __init__(self, run):
        self.run = run

    def __getitem__(self, grid):
        return self.run


class EagerCuLayoutTest(unittest.TestCase):
    def invoke(self, cu, public, check_layout=True):
        length = 16384
        q = torch.ones(1, 2, 4)
        k = torch.zeros(length, 1, 4)
        v = torch.zeros_like(k)
        mask = torch.zeros(1, length, dtype=torch.bool)
        mask[:, 0] = True
        observed = []

        def expand(mask, idx, counts, flag, cuk, *args, **kwargs):
            observed.append(cuk)
            # Kernel ABI: tl.load(CUK), tl.load(CUK + 1), no stride argument.
            physical = cuk.as_strided((2,), (1,))
            flag.fill_(0 if physical.tolist() == [0, length] else 3)

        def attn(idx, count, qs, keys, values, out, *args, **kwargs):
            out.zero_()  # No numerical attention claim from this launch double.

        with patch.object(ed, '_eager_expand_kernel', KernelDouble(expand), create=True), \
                patch.object(ed, '_eager_attn_kernel', KernelDouble(attn), create=True), \
                patch.object(ed, '_ERR_FLAG', {}), \
                patch.object(ed, '_HAS_TRITON', True), \
                patch.object(ed, '_EAGER_TRITON_OK', {'ok': True}), \
                patch.dict(os.environ, {'TLI_E127_FAST': '1', 'TLI_E127_W_UPPER': '32'}):
            if public:
                ed.eager_decoding_attn(q[None], k[None], v[None], mask[None, None], 1, cu)
            else:
                ed._eager_gather_triton(q, k, v, mask, False, 2, 32, cu, .5)
            self.assertEqual(len(observed), 1, 'fused wrapper was not reached')
            ed.eager_decoding_check_errors()
        torch.testing.assert_close(observed[0], cu.to(torch.int32), rtol=0, atol=0)
        if check_layout:
            self.assertEqual(observed[0].stride(0), 1)
        return observed[0]

    def test_gapped_length_array_preserves_logical_values(self):
        for dtype in (torch.int32, torch.int64):
            for public in (False, True):
                with self.subTest(dtype=dtype, public=public):
                    cu = torch.tensor([99, 0, 777, 16384, 888], dtype=dtype)[1:4:2]
                    self.assertEqual(cu.stride(0), 2)
                    self.assertEqual(cu.storage_offset(), 1)
                    self.invoke(cu, public)

    def test_contiguous_int32_no_copy_including_storage_offset(self):
        for offset in (0, 1):
            for public in (False, True):
                with self.subTest(offset=offset, public=public):
                    cu = torch.tensor([99] * offset + [0, 16384], dtype=torch.int32)[offset:]
                    with patch.object(torch.Tensor, 'contiguous', side_effect=AssertionError('extra copy')):
                        observed = self.invoke(cu, public)
                    self.assertIs(observed, cu)

    def test_invalid_logical_length_cannot_hide_behind_adjacent_value(self):
        for public in (False, True):
            with self.subTest(public=public):
                cu = torch.tensor([0, 16384, 16383], dtype=torch.int32)[::2]
                with self.assertRaisesRegex(RuntimeError, 'cu_seqlens_k'):
                    self.invoke(cu, public, check_layout=False)


if __name__ == '__main__':
    unittest.main()
