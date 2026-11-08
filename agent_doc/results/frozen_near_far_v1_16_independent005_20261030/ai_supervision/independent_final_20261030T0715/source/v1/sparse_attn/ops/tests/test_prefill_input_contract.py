"""Fail-closed prefill shape/chunk checks, including under python -O."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from sparse_attn.ops.eager_prefill import sparse_prefill_attn


class PrefillInputContractTest(unittest.TestCase):
    def setUp(self):
        self.q = torch.zeros(1, 4, 2, 8)
        self.k = torch.zeros(1, 4, 1, 8)
        self.indexer = SimpleNamespace(
            args=SimpleNamespace(tia_level2_topk=1024),
            sink_tokens=128, sliding_window_size=128, prepare_mask=Mock(),
        )

    def rejected_before_allocation(self, q=None, k=None, v=None, **kwargs):
        with patch('sparse_attn.ops.eager_prefill.torch.empty_like') as allocate:
            with self.assertRaises(ValueError):
                sparse_prefill_attn(
                    self.indexer, self.q if q is None else q,
                    self.k if k is None else k, self.k if v is None else v,
                    **kwargs,
                )
            allocate.assert_not_called()
        self.indexer.prepare_mask.assert_not_called()

    def test_nonpositive_chunks_rejected(self):
        for chunk in (0, -1, -2048):
            with self.subTest(chunk=chunk):
                self.rejected_before_allocation(chunk_size=chunk)

    def test_nonintegral_chunk_rejected(self):
        for chunk in (1.5, '2048', True):
            with self.subTest(chunk=chunk):
                self.rejected_before_allocation(chunk_size=chunk)

    def test_negative_environment_chunk_rejected(self):
        with patch.dict(os.environ, {'TLI_SPARSE_PREFILL_CHUNK': '-2048'}):
            self.rejected_before_allocation()

    def test_batches_rejected_even_with_optimized_interpreter(self):
        for name, tensor in (('q', self.q), ('k', self.k), ('v', self.k)):
            with self.subTest(tensor=name):
                self.rejected_before_allocation(**{name: tensor.expand(2, -1, -1, -1)})

    def test_incompatible_shapes_rejected(self):
        cases = (
            {'q': self.q[0]}, {'v': self.k[0]},
            {'q': self.q[:, :3]}, {'v': self.k[:, :3]},
            {'q': torch.zeros(1, 4, 3, 8), 'k': torch.zeros(1, 4, 2, 8),
             'v': torch.zeros(1, 4, 2, 8)},
            {'q': torch.zeros(1, 4, 0, 8)},
            {'k': torch.zeros(1, 4, 0, 8), 'v': torch.zeros(1, 4, 0, 8)},
            {'q': self.q[..., :4]}, {'v': self.k[..., :4]},
        )
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                self.rejected_before_allocation(**case)

    def test_empty_sequence_preserves_empty_output(self):
        out = sparse_prefill_attn(self.indexer, self.q[:, :0], self.k[:, :0], self.k[:, :0])
        self.assertEqual(tuple(out.shape), (1, 0, 2, 8))
        self.indexer.prepare_mask.assert_not_called()


if __name__ == '__main__':
    unittest.main()
