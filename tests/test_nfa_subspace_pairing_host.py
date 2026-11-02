"""Source-bound host checks; no claim of Torch/Triton or GPU execution.

Run directly, or set NFA_SELECTOR_SOURCE to a preserved pre-fix selector.
Execute the actual dimension/query dispatch statements without importing CUDA.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

SOURCE = Path(os.environ.get('NFA_SELECTOR_SOURCE', Path(__file__).resolve().parents[1]
    / 'sparse_attn/indexer/nfa_layer_batch_selector.py'))


class Tensor(np.ndarray):
    def long(self):
        return self.astype(np.int64).view(Tensor)

    def numel(self):
        return self.size


def tensor(values):
    return np.asarray(values).view(Tensor)


def assigned(stmt, name):
    return isinstance(stmt, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in stmt.targets)


def dispatch(indexer, query, prepare=None):
    """Extract actual caller dispatch and actual fine-call arguments, no rewrite."""
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_layer_batched_select_impl')
    body = fn.body
    start = next(i for i, n in enumerate(body) if assigned(n, 's_cap')) + 1
    stop = next(i for i, n in enumerate(body) if assigned(n, 'active_req_t'))
    qstart = next(i for i, n in enumerate(body)
                  if isinstance(n, ast.FunctionDef) and n.name == '_prepare_q') + 1
    qstop = next(i for i, n in enumerate(body) if assigned(n, 'layer_ptrs'))
    coarse_view = next(n for n in body if assigned(n, 'b_q'))
    fine = next(n for n in body if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
                and n.value.func.id == '_fused_fine_probabilities_paged')
    calls = []

    def prepare_q(dims, width):
        calls.append(tuple(dims))
        return tensor(query[..., np.asarray(dims)])

    b, hq, d = query.shape
    env = dict(indexer=indexer, device='cpu', d_full=d, m=b, hkv=1,
               group_size=hq,
               cache=SimpleNamespace(layers=[SimpleNamespace(head_dim_k=d)]),
               torch=SimpleNamespace(int64=np.int64,
                   arange=lambda n, **kw: tensor(np.arange(n, dtype=np.int64))),
               _prepare_q=prepare or prepare_q)
    exec(compile(ast.Module(body=body[start:stop] + body[qstart:qstop]
                            + [coarse_view], type_ignores=[]), str(SOURCE), 'exec'), env)
    # The fine wrapper's positional contract is (..., dims, q_fine, ...).
    fine_dims = eval(compile(ast.Expression(fine.value.args[3]), str(SOURCE), 'eval'), env)
    fine_q = eval(compile(ast.Expression(fine.value.args[4]), str(SOURCE), 'eval'), env)
    return env['dims'], env['b_q_full'], fine_dims, fine_q, calls


def config(d, enabled=True, l2='same', coords=None):
    half = d // 2
    tail = tensor(list(range(half-half//4, half)) + list(range(d-half//4, d)))
    return SimpleNamespace(enable_subspace=enabled, l2_subspace=l2,
        _subspace_indices=lambda device: tail if coords is None else tensor(coords),
        _qat_indices=lambda device: tail)


class PairingHostTest(unittest.TestCase):
    def test_full_fine_counterexample(self):
        query = np.zeros((1, 2, 128), dtype=np.float32)
        query[..., 0] = 1
        keys = np.zeros((2, 128), dtype=np.float32)
        keys[0, 0], keys[1, 48] = 2, 3
        _, _, dims, q, _ = dispatch(config(128, l2='full'), query)
        actual = keys[:, dims] @ q[0, 0, :len(dims)]
        expected = keys @ query[0, 0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(int(actual.argmax()), 0)

    def test_full_coarse_same_fine(self):
        query = np.arange(128, dtype=np.float32).reshape(1, 1, 128)
        dims, q, fd, fq, _ = dispatch(config(128, enabled=False), query)
        np.testing.assert_array_equal(dims, np.arange(128))
        np.testing.assert_array_equal(q, query)
        np.testing.assert_array_equal(fd, np.r_[48:64, 112:128])
        np.testing.assert_array_equal(fq, query[..., fd])

    def test_disabled_named_subspace_uses_qat_for_both_fine_operands(self):
        query = np.arange(128, dtype=np.float32).reshape(1, 1, 128)
        cfg = config(128, enabled=False, coords=np.r_[0:16, 64:80])
        _, _, fd, fq, _ = dispatch(cfg, query)
        np.testing.assert_array_equal(fd, np.r_[48:64, 112:128])
        np.testing.assert_array_equal(fq, query[..., fd])

    def test_dimension_matrix(self):
        cases = 0
        for d in (128, 192, 256):
            for b in (1, 3):
                for hq in (1, 4):
                    query = np.arange(b*hq*d, dtype=np.float32).reshape(b, hq, d)
                    for enabled in (True, False):
                        for l2 in ('same', 'full'):
                            for coords in (np.arange(32), np.arange(d-32, d),
                                           np.arange(0, d, 2)):
                                with self.subTest(d=d, b=b, hq=hq, enabled=enabled,
                                                  l2=l2, coords=len(coords)):
                                    cfg = config(d, enabled, l2, coords)
                                    cd, cq, fd, fq, _ = dispatch(cfg, query)
                                    expected_c = coords if enabled else np.arange(d)
                                    expected_f = (np.arange(d) if l2 == 'full' else
                                        coords if enabled else cfg._qat_indices('cpu'))
                                    np.testing.assert_array_equal(cd, expected_c)
                                    np.testing.assert_array_equal(fd, expected_f)
                                    np.testing.assert_array_equal(cq, query[..., expected_c])
                                    np.testing.assert_array_equal(fq, query[..., expected_f])
                                    cases += 1
        self.assertEqual(cases, 144)

    def test_same_tail_reuses_query_and_full_full_needs_one_prepare(self):
        query = np.zeros((3, 4, 128), dtype=np.float32)
        for enabled, l2 in ((True, 'same'), (False, 'full')):
            _, cq, _, fq, calls = dispatch(config(128, enabled, l2), query)
            self.assertIs(cq, fq)
            self.assertEqual(len(calls), 1)

    def test_non_power_two_fine_query_loads_are_masked(self):
        tree = ast.parse(SOURCE.read_text())
        kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_nfa_fine_paged_kernel')
        loads = [n.value for n in ast.walk(kernel) if assigned(n, 'q')]
        self.assertEqual(len(loads), 2)
        for call in loads:
            kws = {k.arg: k.value for k in call.keywords}
            self.assertIn('mask', kws)
            mask = eval(compile(ast.Expression(kws['mask']), str(SOURCE), 'eval'),
                        {'dim_mask': np.arange(256) < 192})
            np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(192))
            self.assertEqual(ast.literal_eval(kws['other']), 0.0)


if __name__ == '__main__':
    unittest.main()
