"""Host-only ratio admission checks; no tensor/selector equivalence claim."""
import os
import unittest
from unittest.mock import patch

from test_explicit_near_page_budget import config, load_constructors


class PartitionRatioContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def setUp(self):
        env = patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA='')
        env.start()
        self.addCleanup(env.stop)

    def legacy(self, **changes):
        return config(tli_near_pages=None, tli_near_tokens=None,
                      tli_near_len=None, **changes)

    def test_invalid_ratios_rejected_before_selector(self):
        for field in ('tli_alpha', 'tli_beta'):
            for value in (-1., -.125, 1.0001, 2., float('nan'),
                          float('inf'), float('-inf')):
                for other in (0., .25):
                    args = dict(tli_alpha=other, tli_beta=other)
                    args[field] = value
                    with self.subTest(field=field, value=value, other=other):
                        with self.assertRaisesRegex(ValueError, field):
                            self.indexer(self.legacy(**args))

    def test_valid_endpoints_and_interior_preserved(self):
        for alpha in (0., .125, .25, .875, 1.):
            for beta in (0., .125, .5, .875, 1.):
                for gamma in (None, 0., .625, 1., 'off'):
                    with self.subTest(alpha=alpha, beta=beta, gamma=gamma):
                        idx = self.indexer(self.legacy(
                            tli_alpha=alpha, tli_beta=beta, tli_gamma=gamma))
                        self.assertEqual(idx.alpha, alpha)
                        self.assertEqual(idx.beta, beta)
                        self.assertEqual(idx.gamma, None if gamma == 'off'
                                         else 1. if gamma is None else gamma)

    def test_defaults_and_explicit_geometry_unchanged(self):
        idx = self.indexer(self.legacy())
        self.assertEqual((idx.alpha, idx.beta, idx.gamma), (0., 0., 1.))
        for nt in (0, 32, 'off'):
            idx = self.indexer(config(tli_near_tokens=nt))
            self.assertEqual((idx.alpha, idx.beta), (1., 1.))
            self.assertEqual(idx.nt_near_tokens, None if nt == 'off' else nt)
        with self.assertRaises(ValueError):
            self.indexer(config(tli_alpha=.25))


if __name__ == '__main__':
    unittest.main()
