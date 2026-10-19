"""Fixed geometry cannot silently route to a different selection contract.

Host-only complete production constructors; no tensor/kernel execution.
"""
import os
import unittest
from unittest.mock import patch

from test_explicit_near_page_budget import config, load_constructors


ALTERNATE_MODES = (
    dict(tli_sinkswa_only=True),
    dict(tli_moba=True),
    dict(tli_sigma_select='far'),
    dict(tli_sigma_select='near'),
    dict(tli_sigma_select='mid'),
)


class ExplicitQuotaModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def setUp(self):
        env = patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA='')
        env.start()
        self.addCleanup(env.stop)

    def test_fixed_geometry_rejects_alternate_selection(self):
        for mode in ALTERNATE_MODES:
            for nt in (0, 32, 'off'):
                with self.subTest(mode=mode, nt=nt):
                    with self.assertRaisesRegex(ValueError, 'fixed geometry'):
                        self.indexer(config(tli_near_tokens=nt, **mode))

    def test_legacy_alternate_modes_still_construct(self):
        for mode in ALTERNATE_MODES:
            with self.subTest(mode=mode):
                idx = self.indexer(config(tli_near_pages=None,
                                          tli_near_tokens=None,
                                          tli_near_len=None, **mode))
                self.assertIsNone(idx.nt_near_pages)
                self.assertIsNone(idx.nt_near_len)

    def test_ordinary_fixed_geometry_keeps_counts(self):
        for bs in (32, 64, 128):
            for nb in (0, 1, 4):
                for nt in (0, 'off') if nb == 0 else (0, 16, 'off'):
                    with self.subTest(bs=bs, nb=nb, nt=nt):
                        idx = self.indexer(config(
                            tia_block_size=bs, tli_swa_size=bs,
                            tli_near_pages=nb, tli_near_tokens=nt,
                            tli_near_len=nb * bs, tli_sigma_select='none',
                            tli_moba=False, tli_sinkswa_only=False))
                        self.assertEqual(idx.nt_near_pages, nb)
                        self.assertEqual(idx.nt_near_tokens,
                                         None if nt == 'off' else nt)
                        self.assertEqual(idx.nt_near_len, nb * bs)


if __name__ == '__main__':
    unittest.main()
