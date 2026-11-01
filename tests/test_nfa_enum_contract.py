"""Configuration guards survive -O, including fixed-quota construction.

Executes complete production constructors through the existing host-only loader;
no tensor or GPU equivalence is claimed.
"""
import os
import unittest
from unittest.mock import patch

from test_explicit_near_page_budget import config, load_constructors


CHOICES = {
    'nfa_subspace': ('full', 'rope', 'nope', 'tail', 'random', 'highfreq', 'mla_rope'),
    'nfa_l2_subspace': ('same', 'full'),
    'nfa_near_select': ('4bit', 'cluster', 'sim_greedy'),
    'nfa_sim_dims': ('subspace', 'nope', 'tail', 'full'),
}


class NFAEnumContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def setUp(self):
        env = patch.dict(os.environ, NFA_E151C_ORACLE_QUOTA='')
        env.start()
        self.addCleanup(env.stop)

    def test_invalid_choices_rejected_in_both_quota_modes(self):
        for fixed in (False, True):
            geometry = {} if fixed else dict(nfa_near_pages=None,
                                             nfa_near_tokens=None, nfa_near_len=None)
            for name in CHOICES:
                for value in ('invalid', '', None, 0):
                    with self.subTest(fixed=fixed, name=name, value=value):
                        with self.assertRaisesRegex(ValueError, name):
                            self.indexer(config(**geometry, **{name: value}))

    def test_all_documented_choices_preserved(self):
        attrs = {'nfa_subspace': 'subspace', 'nfa_l2_subspace': 'l2_subspace',
                 'nfa_near_select': 'near_select', 'nfa_sim_dims': 'sim_dims'}
        for fixed in (False, True):
            geometry = {} if fixed else dict(nfa_near_pages=None,
                                             nfa_near_tokens=None, nfa_near_len=None)
            for name, values in CHOICES.items():
                for value in values:
                    with self.subTest(fixed=fixed, name=name, value=value):
                        idx = self.indexer(config(**geometry, **{name: value}))
                        self.assertEqual(getattr(idx, attrs[name]), value)
                        self.assertEqual(idx.nt_near_pages, 2 if fixed else None)
                        self.assertEqual(idx.nt_near_tokens, 32 if fixed else None)
                        self.assertEqual(idx.nt_near_len, 512 if fixed else None)


if __name__ == '__main__':
    unittest.main()
