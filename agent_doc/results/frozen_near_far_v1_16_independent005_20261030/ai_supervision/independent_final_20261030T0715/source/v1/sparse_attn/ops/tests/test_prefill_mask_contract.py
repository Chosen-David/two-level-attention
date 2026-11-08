"""Selector metadata must not broadcast or discard dimensions in prefill."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sparse_attn.ops.eager_prefill import sparse_prefill_attn


class PrefillMaskContractTest(unittest.TestCase):
    routes = (('reference', 0), ('candidate', 0), ('candidate', 2))

    def env(self, runtime, group):
        return patch.dict(os.environ, {
            'TLI_PREFILL_RUNTIME': runtime, 'TLI_PREFILL_SELECT_GROUP': str(group),
            'TLI_PREFILL_PREFETCH': '0', 'TLI_PREFILL_BATCH_SELECT': '0',
            'TLI_PREFILL_FUSED_ATTN': '0',
        }, clear=True)

    def indexer(self, prepare):
        return SimpleNamespace(
            args=SimpleNamespace(tia_level2_topk=2, tia_block_size=1),
            sink_tokens=0, sliding_window_size=0,
            _e151c_quota=[1], _e151c_on=False, prepare_mask=prepare,
        )

    def test_invalid_metadata_rejected_before_row_attention(self):
        q, k = torch.zeros(1, 7, 4, 4), torch.zeros(1, 7, 2, 4)
        for runtime, group in self.routes:
            prefix = 2 if group else 4
            good = torch.ones(1, 1, 2, prefix, dtype=torch.bool)
            cases = {
                'short_zero': (good[..., :0], 1),
                'short_singleton': (good[..., :1], 1),
                'rank3': (good[0], 1),
                'extra_batch': (good.expand(2, -1, -1, -1), 1),
                'extra_query': (good.expand(-1, 2, -1, -1), 1),
                'missing_kv_head': (good[:, :, :1], 1),
                'wrong_heads': (torch.ones(1, 1, 3, prefix, dtype=torch.bool), 1),
                'float': (good.float(), 1),
                'integer': (good.int(), 1),
                'sparse': (good.to_sparse(), 1),
                'device': (torch.ones_like(good, device='meta'), 1),
                'not_tensor': ([], 1),
                'block_mask': (good, 4),
                'block_zero': (good, 0),
                'block_negative': (good, -1),
                'block_bool': (good, True),
                'block_float': (good, 1.0),
                'block_tensor': (good, torch.tensor(1)),
            }
            module = 'reference_prefill' if runtime == 'reference' else 'eager_prefill'
            for name, result in cases.items():
                idx = self.indexer(lambda *args: result)
                with self.subTest(runtime=runtime, group=group, case=name), self.env(runtime, group):
                    with patch('sparse_attn.ops.' + module + '._chunk_rows_attn') as rows:
                        with self.assertRaisesRegex(ValueError, 'prefill selection'):
                            sparse_prefill_attn(idx, q, k, k, chunk_size=4, dense_below=0)
                        rows.assert_not_called()
                    self.assertFalse(idx._e151c_on)

    def test_singleton_cannot_silently_become_dense(self):
        q = torch.zeros(1, 7, 2, 4)
        k = torch.zeros(1, 7, 1, 4)
        v = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1).expand_as(k)
        idx = self.indexer(lambda *args: (torch.ones(1, 1, 1, 1, dtype=torch.bool), 1))
        with self.env('reference', 0), self.assertRaisesRegex(ValueError, 'prefill selection'):
            # Old reference returns means [0,.5,1,...,3] via singleton broadcast.
            sparse_prefill_attn(idx, q, k, v, chunk_size=4, dense_below=0)

    def test_valid_padding_stride_heads_and_short_inputs(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        for runtime, group in self.routes:
            for length in (0, 1, 7, 9):
                for heads in (2, 4):
                    with self.subTest(runtime=runtime, group=group, length=length, heads=heads):
                        q = torch.zeros(1, length * 2, 4, 4)[:, ::2]
                        k = torch.zeros(1, length * 2, 2, 4)[:, ::2]
                        v = torch.arange(length, dtype=torch.float32).view(1, length, 1, 1)
                        v = v.expand(1, length, 2, 4)
                        def prepare(q_c, ids, k_c, cu, scale):
                            n = k_c.shape[1]
                            mask = torch.zeros(1, 1, heads, (n + 3) * 2, dtype=torch.bool)[..., ::2]
                            mask[..., n:] = True  # padded future keys must be discarded
                            for h in range(heads):
                                mask[0, 0, h, (h + 1) % n] = True
                                mask[0, 0, h, n - 1] = True
                            return mask, 1
                        idx = self.indexer(prepare)
                        idx.sink_tokens, idx.sliding_window_size = 1, 2
                        with self.env(runtime, group):
                            actual = sparse_prefill_attn(idx, q, k, v, chunk_size=4, dense_below=4)
                        expected = torch.empty_like(q)
                        for row in range(length):
                            for h in range(4):
                                if row < 4:
                                    visible = list(range(row + 1))
                                else:
                                    size = group or 4
                                    end = min(4 + ((row - 4) // size + 1) * size, length)
                                    mh = h if heads == 4 else h // 2
                                    selected = {(mh + 1) % end, end - 1}
                                    protected = {0, max(0, row - 1), row}
                                    visible = sorted(x for x in selected | protected if x <= row)
                                expected[0, row, h] = sum(visible) / len(visible)
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                        self.assertFalse(idx._e151c_on)


if __name__ == '__main__':
    unittest.main()
