"""Fixed geometry and per-layer oracle quotas are different contracts.

Host-only complete production constructors; no tensor/GPU parity claim.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_explicit_near_page_budget import config, load_constructors


class ExplicitQuotaOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def make_index(self, quota, **changes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'quota.json'
            path.write_text(json.dumps(quota))
            with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=(
                    '' if quota is None else str(path))):
                return self.indexer(config(**changes))

    def test_numeric_fixed_geometry_rejects_oracle_override(self):
        for nt in (0, 32, 128):
            for quota in ([0], [32], [128]):
                with self.subTest(nt=nt, quota=quota):
                    with self.assertRaisesRegex(ValueError, 'E151c'):
                        self.make_index(quota, tli_near_tokens=nt)

    def test_free_competition_still_rejects_oracle(self):
        with self.assertRaisesRegex(ValueError, 'E151c'):
            self.make_index([32], tli_near_tokens='off')

    def test_legacy_oracle_remains_available(self):
        for quota in ([0], [32, 128], [0.5, 1.5, 2.5]):
            with self.subTest(quota=quota):
                idx = self.make_index(quota, tli_near_pages=None,
                    tli_near_tokens=None, tli_near_len=None,
                    tli_alpha=.5, tli_beta=.5, tli_gamma=.5)
                self.assertEqual(idx._e151c_quota, [round(x) for x in quota])
                self.assertFalse(idx._e151c_on)

    def test_fixed_geometry_without_oracle_unchanged(self):
        for nt in (0, 32, 128, 'off'):
            with self.subTest(nt=nt):
                idx = self.make_index(None, tli_near_tokens=nt)
                self.assertIsNone(idx._e151c_quota)
                self.assertEqual(idx.nt_near_tokens, None if nt == 'off' else nt)


if __name__ == '__main__':
    unittest.main()
