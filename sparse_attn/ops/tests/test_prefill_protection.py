"""CPU regression for the public prefill entry's configured protection regions.

Run: python -m unittest discover -s sparse_attn/ops/tests -v
The deterministic selector isolates adapter semantics from ranking/quantization.
"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from sparse_attn.ops.eager_prefill import sparse_prefill_attn


class ProtectionIndexer:
    """A valid shared/per-query-head mask: sink plus the chunk-end SWA."""

    def __init__(self, sink, block_size, mask_heads):
        self.args = SimpleNamespace(tia_block_size=block_size, tia_level2_topk=128)
        self.sink_tokens = sink
        self.sink_blocks = 2  # Legacy metadata must not override token config.
        self.sliding_window_size = 64
        self.mask_heads = mask_heads
        self.prefixes = []

    def prepare_mask(self, q, q_ids, k, cu, scale):
        length = k.shape[1]
        self.prefixes.append(length)
        mask = torch.zeros(1, 1, self.mask_heads, length, dtype=torch.bool)
        mask[..., :self.sink_tokens] = True
        mask[..., max(0, length - self.sliding_window_size):] = True
        return mask, 1  # Token mask, matching TLIIndexer.get_block_size().


class PrefillProtectionTest(unittest.TestCase):
    def setUp(self):
        # These fixtures allocate CPU tensors only. CPU-only Torch cannot
        # query CUDA stream capture; indexing/scoring/masks remain real.
        capture = patch('torch.cuda.is_current_stream_capturing', return_value=False)
        capture.start()
        self.addCleanup(capture.stop)

    def test_real_indexer_sinkswa_only_does_not_restore_disabled_sink(self):
        from sparse_attn.indexer import TLIIndexer

        torch.set_num_threads(1)
        args = SimpleNamespace(
            tia_block_size=64, tia_level1_topk=128, tia_level2_topk=128,
            tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
            tli_enable_layer_skip=False, tli_enable_kmeans=False,
            tli_sink_tokens=0, tli_swa_size=64, tli_sinkswa_only=True,
        )
        indexer = TLIIndexer(args)
        length = 273
        q = torch.zeros(1, length, 2, 128)
        k = torch.zeros(1, length, 1, 128)
        v = torch.arange(length, dtype=torch.float32).view(1, length, 1, 1)
        v = v.expand(1, length, 1, 128)
        actual = sparse_prefill_attn(indexer, q, k, v, chunk_size=128, dense_below=128)
        # At the last query, both chunk-end and row-local SWA are [209,272].
        torch.testing.assert_close(actual[0, -1], torch.full((2, 128), 240.5))

    def test_configured_sink_and_swa_match_absolute_position_reference(self):
        # Zero logits make outputs exactly the mean of the allowed V rows.
        # Distinct values expose both unwanted and missing protected tokens.
        torch.set_num_threads(1)
        for sink, block in ((0, 64), (64, 64), (128, 64), (192, 64), (128, 32)):
            for heads in (1, 2):  # Shared KV mask and per-query-head mask, GQA=2.
                with self.subTest(sink=sink, block=block, mask_heads=heads):
                    length, chunk = 273, 128  # Includes a partial final chunk.
                    # Non-contiguous sequence storage, matching prefix views.
                    q = torch.zeros(1, length * 2, 2, 4)[:, ::2]
                    k = torch.zeros(1, length * 2, 1, 4)[:, ::2]
                    v = torch.arange(length * 2, dtype=torch.float32).view(
                        1, length * 2, 1, 1
                    )[:, ::2].expand(1, length, 1, 4)
                    indexer = ProtectionIndexer(sink, block, heads)
                    actual = sparse_prefill_attn(
                        indexer, q, k, v, chunk_size=chunk, dense_below=chunk
                    )
                    expected = torch.empty_like(q)
                    for row in range(length):
                        end = min((row // chunk + 1) * chunk, length)
                        if row < chunk:
                            positions = list(range(row + 1))
                        else:
                            positions = [t for t in range(row + 1) if (
                                t < sink or t >= end - 64 or t >= row - 63
                            )]
                        expected[0, row] = v[0, positions].mean(dim=0)
                    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-6)
                    self.assertEqual(indexer.prefixes, [256, 273])

    def test_default_outer_chunk_remains_2048(self):
        from unittest.mock import patch
        indexer = ProtectionIndexer(0, 64, 1)
        q = torch.zeros(1, 2051, 2, 4)
        k = torch.zeros(1, 2051, 1, 4)
        seen = []

        def capture(q_c, k_c, v_c, sel, pos_lo, *args, **kwargs):
            seen.append((pos_lo, q_c.shape[1], kwargs['sink_tok'], kwargs['swa_tok']))
            return torch.zeros_like(q_c)

        with patch.dict('os.environ', {}, clear=True), patch(
            'sparse_attn.ops.reference_prefill._chunk_rows_attn', side_effect=capture
        ):
            sparse_prefill_attn(indexer, q, k, k)
        self.assertEqual(seen, [(0, 2048, 0, 0), (2048, 3, 0, 64)])
        self.assertEqual(indexer.prefixes, [2051])


if __name__ == '__main__':
    unittest.main()

