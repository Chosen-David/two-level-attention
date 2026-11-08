"""An empty row must not change another row/head's selected support."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sparse_attn.ops.eager_prefill import sparse_prefill_attn


class EmptyRowFallbackTest(unittest.TestCase):
    def test_fallback_is_local_to_empty_row_and_head(self):
        torch.set_num_threads(1)
        for runtime in ('reference', 'candidate'):
            for heads in (1, 2, 4):  # shared KV, two KV heads, per-query head
                with self.subTest(runtime=runtime, mask_heads=heads):
                    hkv = 2 if heads == 2 else 1
                    q = torch.zeros(1, 8, 4, 4)
                    k = torch.zeros(1, 8, hkv, 4)
                    v = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)
                    v = v.expand(1, 8, hkv, 4)
                    mask = torch.zeros(1, 1, heads, 8, dtype=torch.bool)
                    mask[..., 0, 6] = True
                    if heads > 1:
                        mask[..., 1:, 0] = True
                    indexer = SimpleNamespace(
                        args=SimpleNamespace(tia_level2_topk=4, tia_block_size=1),
                        sink_tokens=0, sliding_window_size=0,
                        prepare_mask=lambda *args: (mask, 1),
                    )
                    with patch.dict(os.environ, {
                        'TLI_PREFILL_RUNTIME': runtime,
                        'TLI_PREFILL_SELECT_GROUP': '0',
                        'TLI_PREFILL_PREFETCH': '0',
                    }, clear=True):
                        actual = sparse_prefill_attn(
                            indexer, q, k, v, chunk_size=4, dense_below=4,
                        )
                    expected = torch.empty_like(q)
                    for row in range(8):
                        for head in range(4):
                            mask_head = head if heads == 4 else head // (4 // heads)
                            visible = list(range(row + 1)) if row < 4 else [
                                col for col in range(row + 1) if mask[0, 0, mask_head, col]
                            ]
                            if not visible:
                                visible = [row]
                            expected[0, row, head] = sum(visible) / len(visible)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
