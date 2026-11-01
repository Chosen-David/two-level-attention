"""Configured oracle quotas must not silently select a different algorithm."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sparse_attn.indexer import NFAIndexer


UNSUPPORTED = (
    dict(nfa_gamma="off"),
    dict(nfa_alpha=0),
    dict(nfa_beta=0),
    dict(nfa_alpha=0, nfa_beta=0),
    dict(nfa_sinkswa_only=True),
    dict(nfa_moba=True),
    dict(nfa_sigma_select='far'),
    dict(nfa_sigma_select='near'),
    dict(nfa_sigma_select='mid'),
)


def make_index(overrides, quota=None):
    args = dict(
        tia_block_size=64, tia_level1_topk=8, tia_level2_topk=256,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        nfa_enable_layer_skip=False, nfa_enable_kmeans=False,
        nfa_subspace='tail', nfa_sink_tokens=64, nfa_swa_size=64,
        nfa_alpha=.5, nfa_beta=.5, nfa_gamma=.5,
    )
    args.update(overrides)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'quota.json'
        path.write_text(json.dumps(quota))
        with patch.dict(os.environ, NFA_E151C_ORACLE_QUOTA=(
                '' if quota is None else str(path))):
            return NFAIndexer(SimpleNamespace(**args))


class QuotaModeTests(unittest.TestCase):
    def test_unsupported_modes_reject_configured_quota(self):
        for config in UNSUPPORTED:
            for quota in ([0], [32]):
                with self.subTest(config=config, quota=quota):
                    with self.assertRaisesRegex(ValueError, 'E151c'):
                        make_index(config, quota)

    def test_unconfigured_modes_still_construct(self):
        for config in UNSUPPORTED:
            with self.subTest(config=config):
                self.assertIsNone(make_index(config)._e151c_quota)

    def test_supported_modes_preserve_quota(self):
        for config in ({}, dict(nfa_gamma=0), dict(nfa_gamma=1),
                       dict(nfa_per_q_head=True),
                       dict(nfa_enable_kmeans=True, nfa_far_select='cluster'),
                       dict(nfa_enable_kmeans=True, nfa_far_select='sim_greedy')):
            with self.subTest(config=config):
                self.assertEqual(make_index(config, [0, 32])._e151c_quota, [0, 32])


if __name__ == '__main__':
    unittest.main()
