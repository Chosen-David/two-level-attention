"""CPU regression for the FASA adapter's accepted boolean-mask contract.

Execute the production helper bodies without importing the upstream module's
unconditional CUDA allocation. No GPU or model output is simulated as tested.
"""
import ast
from pathlib import Path
import unittest

import torch
from sparse_attn.baseline_runtime import (
    BatchCtx, batch_decode_scope, require_unmasked_decode, batch_valid_len_from_mask,
)

SOURCE = Path(__file__).resolve().parents[1] / 'exp/vendor_baselines/fasa_vendor/qwen3_fasa_patch.py'


def load_function(name):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


class FasaBooleanMaskTests(unittest.TestCase):
    def setUp(self):
        self.decode = load_function('_qwen3_decode_eager')

    def test_actual_4d_masks_produce_flat_batch_lengths_and_fail_closed(self):
        boolean = torch.tensor([[[[False, True]]], [[[True, True]]]])
        additive = torch.zeros(boolean.shape).masked_fill(~boolean, float('-inf'))
        for mask in (boolean, additive):
            self.assertEqual(batch_valid_len_from_mask(mask), [1, 2])
            with batch_decode_scope(BatchCtx(2, [2, 2], [0, 0], [True, True], 0)):
                with self.assertRaisesRegex(ValueError, 'derived lengths'):
                    require_unmasked_decode(torch.zeros(2, 1, 4), mask,
                                            object(), torch.tensor([1]))
        for mask in (boolean[:, :, 0], boolean.expand(2, 2, 1, 2)):
            with self.assertRaises(ValueError):
                batch_valid_len_from_mask(mask)

    def test_guard_accepted_batch_boolean_padding_gets_zero_probability(self):
        mask = torch.tensor([[[[False, True]]], [[[True, True]]]])
        hidden = torch.zeros(2, 1, 4)
        with batch_decode_scope(BatchCtx(2, [1, 2], [0, 0], [True, True], 0)):
            require_unmasked_decode(hidden, mask, object(), torch.tensor([1]))
        q = torch.zeros(2, 2, 1, 2)
        k = torch.zeros(2, 2, 2, 2)
        v = torch.zeros_like(k)
        v[0, :, 0] = 100  # Any leaked padding has a detectable effect.
        indices = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        selected_v = v.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 2))
        actual, weights = self.decode(q, k, selected_v, mask, .5, indices)
        expected_mask = mask.expand(2, 2, 1, 2).gather(-1, indices.unsqueeze(2))
        self.assertTrue(torch.equal(weights.masked_select(~expected_mask),
                                    torch.zeros_like(weights.masked_select(~expected_mask))))
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_short_row_with_budget_exceeding_visible_tokens(self):
        core = load_function('_fasa_batch_core')
        q = torch.zeros(2, 2, 1, 2)
        k = torch.zeros(2, 2, 2, 2)
        v = torch.zeros_like(k)
        v[0, :, 0] = 100
        mask = torch.tensor([[[[False, True]]], [[[True, True]]]])
        records = {1: {'head_selection': torch.tensor([[0, 1], [0, 1]]),
                       'selection_masks': torch.ones(2, 2, dtype=torch.bool)}}
        selected_k, selected_v, indices = core(q, k, v, 1, 2, records, mask[:, 0])
        out, weights = self.decode(q, selected_k, selected_v, mask, .5, indices)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertEqual(int((weights[0] == 0).sum()), 2)

    def test_permuted_boolean_and_additive_masks_are_equivalent(self):
        generator = torch.Generator().manual_seed(19)
        q = torch.randn(2, 3, 1, 4, generator=generator)
        k = torch.randn(2, 3, 3, 4, generator=generator)
        v = torch.randn(2, 3, 3, 4, generator=generator)
        indices = torch.tensor([[[3, 0, 2]] * 3, [[2, 3, 0]] * 3])
        mask = torch.tensor([[[[False, False, True, True]]],
                             [[[False, True, True, True]]]])
        additive = torch.zeros(mask.shape).masked_fill(~mask, float('-inf'))
        actual = self.decode(q, k, v, mask, .5, indices)
        expected = self.decode(q, k, v, additive, .5, indices)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        gathered = mask.expand(2, 3, 1, 4).gather(-1, indices.unsqueeze(2))
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, gathered, scale=.5)
        torch.testing.assert_close(actual[0], reference.transpose(1, 2), rtol=1e-6, atol=1e-6)

    def test_no_mask_all_visible_and_additive_bias_keep_their_semantics(self):
        q = torch.tensor([[[[1., 0.]]]])
        k = torch.tensor([[[[1., 0.], [0., 1.]]]])
        v = torch.tensor([[[[2., 4.], [5., 7.]]]])
        masks = [None, torch.ones(1, 1, 1, 2, dtype=torch.bool),
                 torch.tensor([[[[0., -2.]]]])]
        for mask in masks:
            with self.subTest(mask=mask):
                out, _ = self.decode(q, k, v, mask, .7)
                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, scale=.7)
                torch.testing.assert_close(out, expected.transpose(1, 2), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
