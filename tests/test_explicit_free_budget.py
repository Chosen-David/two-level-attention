"""Host-only explicit nt=off protected-token budget contracts.

Uses the complete production constructor chain loader, not a tensor substitute.
No selector, GPU correctness, model quality or performance claim is made here.
"""
import os
import unittest
from unittest.mock import patch

from test_explicit_near_page_budget import config, load_constructors


class ExplicitFreeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def setUp(self):
        env = patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA='')
        env.start()
        self.addCleanup(env.stop)

    def test_off_rejects_insufficient_protected_budget(self):
        for bs in (32, 64, 128):
            for sink in (0, 4, 128):
                for deficit in (1, bs):
                    with self.subTest(bs=bs, sink=sink, deficit=deficit):
                        with self.assertRaisesRegex(ValueError, 'K2'):
                            self.indexer(config(
                                tia_block_size=bs, tli_sink_tokens=sink,
                                tli_swa_size=bs, tia_level2_topk=sink + bs - deficit,
                                tli_near_tokens='off'))

    def test_off_accepts_zero_and_positive_mid_budget(self):
        for bs in (32, 64, 128):
            for sink in (0, 4, 128):
                for extra in (0, 1, 256):
                    with self.subTest(bs=bs, sink=sink, extra=extra):
                        idx = self.indexer(config(
                            tia_block_size=bs, tli_sink_tokens=sink,
                            tli_swa_size=bs, tia_level2_topk=sink + bs + extra,
                            tli_near_tokens='off'))
                        self.assertIsNone(idx.gamma)
                        self.assertIsNone(idx.nt_near_tokens)
                        self.assertEqual(idx.nt_near_pages, 2)
                        self.assertEqual(idx.nt_near_len, 512)

    def test_numeric_budget_contract_preserved(self):
        for nt in (0, 32, 128):
            for delta in (-1, 0, 1):
                with self.subTest(nt=nt, delta=delta):
                    args = config(tli_near_tokens=nt,
                                  tia_level2_topk=128 + nt + delta)
                    if delta < 0:
                        with self.assertRaises(ValueError):
                            self.indexer(args)
                    else:
                        idx = self.indexer(args)
                        self.assertEqual(idx.nt_near_tokens, nt)
                        self.assertEqual(idx.gamma, 1.0)


if __name__ == '__main__':
    unittest.main()
