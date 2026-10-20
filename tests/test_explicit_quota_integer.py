"""Fixed quotas are integer counts, including direct Python construction.

Runs full production constructor bodies, not tensor selection or GPU kernels.
"""
import os
import unittest
from unittest.mock import patch

from test_explicit_near_page_budget import config, load_constructors


class IndexCount:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class ExplicitQuotaIntegerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def setUp(self):
        env = patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA='')
        env.start()
        self.addCleanup(env.stop)

    def test_noninteger_counts_rejected_without_truncation(self):
        for name in ('tli_near_pages', 'tli_near_tokens', 'tli_near_len'):
            for value in (-.5, .5, 1.5, 1.0, False, True, float('nan'), float('inf')):
                with self.subTest(name=name, value=value):
                    # nb=nt=0 makes the negative fractional truncation loophole
                    # reachable independently of the separate capacity guards.
                    args = config(tli_near_pages=0, tli_near_tokens=0,
                                  tli_near_len=1024)
                    setattr(args, name, value)
                    with self.assertRaisesRegex(ValueError, name):
                        self.indexer(args)

    def test_integer_protocol_and_existing_integer_strings(self):
        for factory in (int, str, IndexCount):
            for nb, nt, nl in ((0, 0, 0), (1, 0, 64), (2, 32, 512), (4, 128, 4096)):
                with self.subTest(factory=factory.__name__, values=(nb, nt, nl)):
                    idx = self.indexer(config(tli_near_pages=factory(nb),
                                              tli_near_tokens=factory(nt),
                                              tli_near_len=factory(nl)))
                    self.assertEqual(tuple(int(v) for v in (idx.nt_near_pages,
                                     idx.nt_near_tokens, idx.nt_near_len)), (nb, nt, nl))

    def test_off_and_missing_geometry_contracts_preserved(self):
        for off in ('off', ' OFF ', 'Off'):
            idx = self.indexer(config(tli_near_tokens=off))
            self.assertIsNone(idx.nt_near_tokens)
            self.assertIsNone(idx.gamma)
        for name in ('tli_near_pages', 'tli_near_tokens', 'tli_near_len'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.indexer(config(**{name: None}))


if __name__ == '__main__':
    unittest.main()
