"""Host admission checks; these do not execute tensor or CUDA kernels."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'sparse_attn/indexer/tli_indexer.py'


def production_helper():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TLIIndexer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_gather_fine_cand')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[method.name]


class AsyncCandidateGateTests(unittest.TestCase):
    def invoke(self, *, valid_length=13, **options):
        calls = []

        def select(_coarse, _average, length, **_means):
            calls.append(length)
            raise RuntimeError('probe stops before tensor operations')

        state = dict(_fine_gather=True, _basis=None, moba_gate=False,
                     per_q_head=False, enable_async=False, sigma_select='none',
                     far_select='4bit', near_select='4bit', prev_mask=None,
                     _select_l1_blocks=select)
        state.update(options)
        indexer = SimpleNamespace(**state)
        previous = indexer.prev_mask
        result = production_helper()(indexer, object(), None,
                                     SimpleNamespace(shape=(1, 16, 2, 4)),
                                     object(), valid_length)
        self.assertEqual(result, (None, None, None, None))
        self.assertIs(indexer.prev_mask, previous)
        return calls

    def test_async_first_and_later_steps_bypass_current_candidate_selection(self):
        for previous in (None, object()):
            with self.subTest(first_step=previous is None):
                self.assertEqual(self.invoke(enable_async=True, prev_mask=previous), [])

    def test_sync_candidate_path_still_reaches_selection(self):
        self.assertEqual(self.invoke(), [13])
        self.assertEqual(self.invoke(valid_length=None), [16])

    def test_existing_unsupported_modes_still_bypass_candidates(self):
        for options in (dict(_fine_gather=False), dict(_basis=object()),
                        dict(moba_gate=True), dict(per_q_head=True),
                        dict(sigma_select='mass'), dict(far_select='cluster'),
                        dict(near_select='sim_greedy')):
            with self.subTest(options=options):
                self.assertEqual(self.invoke(**options), [])


if __name__ == '__main__':
    unittest.main()
