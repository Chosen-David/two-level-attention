"""E151c JSON configuration must not silently turn keys/characters into quotas."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sparse_attn.indexer import TLIIndexer


class QuotaInputTests(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quota.json'
            path.write_text(json.dumps(value))
            with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=str(path)):
                return TLIIndexer(SimpleNamespace(
                    tia_block_size=64, tia_level1_topk=8, tia_level2_topk=256,
                    tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
                    tli_enable_layer_skip=False, tli_enable_kmeans=False,
                    tli_subspace='tail', tli_sink_tokens=64, tli_swa_size=64,
                    tli_alpha=.5, tli_beta=.5, tli_gamma=.5))

    def test_non_arrays_and_empty_arrays_rejected(self):
        for value in ('32', {'32': 999}, [], None, 32):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.load(value)

    def test_invalid_elements_rejected(self):
        for value in (True, False, '32', None, {}, [], -1, -.1,
                      float('nan'), float('inf'), float('-inf'), 10 ** 400):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.load([32, value])

    def test_numeric_rounding_is_preserved(self):
        values = [0, -0., 32, 32.4, 32.5, 33.5, 128, 10000]
        idx = self.load(values)
        self.assertEqual(idx._e151c_quota, [int(round(float(x))) for x in values])
        self.assertFalse(idx._e151c_on)


if __name__ == '__main__':
    unittest.main()
