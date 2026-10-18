"""CPU host/glue regression; does not validate CUDA topk tie behavior."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sparse_attn.ops import batch_select as batch


class BatchTopkBlockSizeTest(unittest.TestCase):
    def run_selector(self, bs, enabled):
        length = 1024
        k = torch.zeros(1, length, 1, 128)
        q = torch.zeros(1, length, 2, 128)
        coarse = k[..., :32].reshape(1, length // bs, bs, 1, 32).mean(2)
        idx = SimpleNamespace(
            args=SimpleNamespace(tia_block_size=bs, tia_level1_topk=2,
                                 tia_level2_topk=512),
            sink_tokens=bs, sliding_window_size=bs, _sink_pool_lo_blk=1,
            alpha=.25, beta=.5, gamma=.5, far_method='minmax', near_method='avg',
            _subspace_indices=lambda dev: torch.arange(32, device=dev),
            prepare_index=lambda *a: dict(k_min=coarse, k_max=coarse,
                                         k_avg=coarse, k_qat=k[..., :32]))
        def score(keys, query, dtype):
            return torch.arange(keys.shape[0], dtype=dtype).expand(query.shape[0], -1) / 100
        idx._fast_score_fine = score
        idx._fast_score_coarse = lambda lo, hi, query, dtype: score(lo, query, dtype)
        with patch.object(batch, '_gate_ok', return_value=True), patch.dict(
                os.environ, {'TLI_PREFILL_SELECT_GRAPH': '0',
                             'TLI_PREFILL_SHARED_GQA_INDICES': '0',
                             'TLI_PREFILL_L2_TOPK_CAND': enabled}):
            return batch.batched_group_sels(idx, q, k, 128 ** -.5, [512, 1024], 256)

    def test_non64_blocks_retain_full_width_selection(self):
        for bs in (16, 32, 128):
            with self.subTest(block_size=bs):
                expected = self.run_selector(bs, '0')
                actual = self.run_selector(bs, '1')
                for a, b in zip(actual, expected):
                    self.assertTrue(torch.equal(a, b), f'block size {bs}: changed selection')

    def test_64_block_candidate_specialization_remains_reachable(self):
        with patch.object(batch, '_zero_pad_pool', wraps=batch._zero_pad_pool) as padding:
            result = self.run_selector(64, '1')
        self.assertIsNotNone(result)
        self.assertGreater(padding.call_count, 0)


if __name__ == '__main__':
    unittest.main()
