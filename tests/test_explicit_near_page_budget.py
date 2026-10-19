"""Host-only E155 constructor contracts; no selector or GPU parity claim.

Execute the complete production constructor ASTs, including the inheritance
chain, without importing the tensor stack. Only the metrics factory is stubbed;
any Torch use fails. This remains runnable on audit hosts without Torch.
"""
import ast
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_constructors():
    class NoTensorWork:
        def __getattr__(self, name):
            raise AssertionError('unexpected tensor work: ' + name)

    scope = dict(os=os, math=math, json=json, torch=NoTensorWork(),
                 DEFAULT_MASK='unused-with-layer-skip-disabled',
                 get_metrics=lambda: None)
    for filename, name in (('base.py', 'Indexer'),
                           ('tia_indexer.py', 'TIAIndexer'),
                           ('tli_indexer.py', 'TLIIndexer')):
        path = ROOT / 'sparse_attn' / 'indexer' / filename
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == name)
        node.body = [n for n in node.body
                     if isinstance(n, ast.FunctionDef) and n.name == '__init__']
        if len(node.body) != 1:
            raise RuntimeError('production constructor missing: ' + name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['TLIIndexer']


def config(**changes):
    values = dict(tia_block_size=64, tia_level1_topk=4,
                  tia_level2_topk=256, tia_level2_cmp_ratio=4,
                  tia_enable_async_topk=False, tli_enable_layer_skip=False,
                  tli_sink_tokens=64, tli_swa_size=64,
                  tli_near_pages=2, tli_near_tokens=32, tli_near_len=512)
    values.update(changes)
    return SimpleNamespace(**values)


class ExplicitNearPageBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = load_constructors()

    def setUp(self):
        env = patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA='')
        env.start()
        self.addCleanup(env.stop)

    def test_over_budget_near_pages_rejected(self):
        # All other geometric constraints are satisfied. nt=off still uses
        # L1 double-pool selection and must obey the same page budget.
        for k1 in (0, 1, 4, 128):
            for extra in (1, 3):
                for nt in (0, 32, 'off'):
                    nb = k1 + extra
                    with self.subTest(k1=k1, nb=nb, nt=nt):
                        with self.assertRaisesRegex(ValueError, 'tia_level1_topk'):
                            self.indexer(config(tia_level1_topk=k1,
                                                tli_near_pages=nb,
                                                tli_near_tokens=nt,
                                                tli_near_len=nb * 64))

    def test_budget_endpoints_and_interior_preserved(self):
        for bs in (32, 64, 128):
            for k1 in (0, 1, 4, 128):
                for nb in sorted({0, k1 // 2, k1}):
                    for nt in (0, 'off') if nb == 0 else (0, 32, 'off'):
                        with self.subTest(bs=bs, k1=k1, nb=nb, nt=nt):
                            idx = self.indexer(config(
                                tia_block_size=bs, tia_level1_topk=k1,
                                tli_swa_size=bs, tli_near_pages=nb,
                                tli_near_tokens=nt, tli_near_len=nb * bs))
                            self.assertEqual(idx.nt_near_pages, nb)
                            self.assertEqual(idx.nt_near_tokens, None if nt == 'off' else nt)
                            self.assertEqual(idx.gamma, None if nt == 'off' else 1.0)
                            self.assertEqual((idx.alpha, idx.beta), (1., 1.))

    def test_other_geometry_checks_preserved(self):
        for changes in (dict(tli_near_pages=-1), dict(tli_near_len=127),
                        dict(tli_near_tokens=129), dict(tli_near_tokens=-1),
                        dict(tia_level2_topk=128), dict(tli_near_pages=None),
                        dict(tli_near_tokens=None), dict(tli_near_len=None),
                        dict(tli_alpha=0), dict(tli_beta=1), dict(tli_gamma='off')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.indexer(config(**changes))

    def test_legacy_ratios_unchanged(self):
        for ratios, expected in (({}, (0., 0., 1.)),
                                 (dict(tli_alpha=.25, tli_beta=.5, tli_gamma=.75),
                                  (.25, .5, .75)),
                                 (dict(tli_gamma='off'), (0., 0., None))):
            with self.subTest(ratios=ratios):
                idx = self.indexer(config(tli_near_pages=None,
                                          tli_near_tokens=None,
                                          tli_near_len=None, **ratios))
                self.assertEqual((idx.alpha, idx.beta, idx.gamma), expected)


if __name__ == '__main__':
    unittest.main()
