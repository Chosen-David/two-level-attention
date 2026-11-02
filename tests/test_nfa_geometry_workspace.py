"""Geometry workspace regression; host emulation is not GPU validation.

NFA_SELECTOR_SOURCE selects a source under test (including pre-fix main).
NFA_GEOMETRY_BASELINE optionally enables exact paired baseline checks.
Only the two pure tensor helpers are extracted; production imports are not
replaced globally, and native checks use real Torch CUDA tensors.
"""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

SOURCE = Path(os.environ.get('NFA_SELECTOR_SOURCE',
    Path(__file__).resolve().parents[1] /
    'sparse_attn/indexer/nfa_layer_batch_selector.py'))
BASELINE = os.environ.get('NFA_GEOMETRY_BASELINE')


class Tensor(np.ndarray):
    def numel(self): return self.size
    def new_full(self, shape, value):
        return tensor(np.full(shape, value, dtype=self.dtype))
    def unsqueeze(self, dim): return tensor(np.expand_dims(self, dim))
    def expand(self, *shape):
        old = (1,) * (len(shape) - self.ndim) + self.shape
        return tensor(np.broadcast_to(self, tuple(
            a if b == -1 else b for a, b in zip(old, shape))))
    def to(self, dtype): return tensor(self.astype(dtype))
    def contiguous(self): return tensor(np.ascontiguousarray(self))


def tensor(x): return np.asarray(x).view(Tensor)


def host_backend():
    axes = []
    def arange(n, **kwargs):
        axes.append(n)
        return tensor(np.arange(n, dtype=np.int64))
    return NS(long=np.int64, int64=np.int64, bool=np.bool_,
        as_tensor=lambda x, **kw: tensor(x), arange=arange,
        clamp=lambda x, min=None, max=None: tensor(np.clip(x, min, max)),
        maximum=lambda a, b: tensor(np.maximum(a, b)),
        full=lambda shape, value, **kw: tensor(np.full(
            shape, value, dtype=kw.get('dtype'))),
        stack=lambda xs, dim=0: tensor(np.stack(xs, axis=dim)),
        cat=lambda xs, dim=0: tensor(np.concatenate(xs, axis=dim)),
        gather=lambda x, dim, i: tensor(np.take_along_axis(x, i, axis=dim)),
        where=lambda c, a, b: tensor(np.where(c, a, b)),
        zeros_like=lambda x: tensor(np.zeros_like(x))), axes


def helpers(path, backend):
    tree = ast.parse(Path(path).read_text())
    names = {'_batched_geometry', '_candidate_assembly'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in names]
    if len(nodes) != 2:
        raise RuntimeError('expected the two production geometry helpers')
    env = {'torch': backend}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env['_batched_geometry'], env['_candidate_assembly']


def config(bs, sink, window, near):
    return NS(args=NS(tia_block_size=bs, tia_level2_topk=1024),
              sink_tokens=sink, sliding_window_size=window,
              nt_near_len=near, _sink_pool_lo_blk=sink // bs)


class GeometryWorkspaceTest(unittest.TestCase):
    def test_no_token_capacity_axis(self):
        backend, axes = host_backend()
        geometry, _ = helpers(SOURCE, backend)
        cap = 131072
        out = geometry(None, None, tensor([0, 1, 513, cap]),
                       config(64, 4, 128, 4096), 'cpu', cap // 64, cap)
        self.assertNotIn(cap, axes,
                         'geometry must not build a full token-capacity axis')
        self.assertFalse({'mid_mask', 'swa_force'} & out.keys())

    @unittest.skipUnless(BASELINE, 'set NFA_GEOMETRY_BASELINE for paired checks')
    def test_paired_geometry_and_candidate_assembly(self):
        backend, _ = host_backend()
        before, old_assembly = helpers(BASELINE, backend)
        after, new_assembly = helpers(SOURCE, backend)
        rng = np.random.default_rng(20261102)
        cases = 0
        for bs in (4, 8, 64):
            for sink in (0, 1, bs - 1, bs, bs + 1):
                for window in (0, bs, 2 * bs):
                    for near in (0, bs, 3 * bs):
                        cap = 16 * bs
                        lengths = tensor(np.array(sorted(set([
                            0, 1, sink, sink + 1, window, window + 1,
                            5 * bs - 1, 5 * bs, 5 * bs + 1, cap])), dtype=np.int64))
                        cfg = config(bs, sink, window, near)
                        a = before(None, None, lengths, cfg, 'cpu', 16, cap)
                        b = after(None, None, lengths, cfg, 'cpu', 16, cap)
                        self.assertEqual(set(a) - {'mid_mask', 'swa_force'}, set(b))
                        for key in b:
                            np.testing.assert_array_equal(a[key], b[key])
                        batch = len(lengths)
                        indices = tensor(np.stack([
                            rng.permutation(16) for _ in range(batch * 2)
                        ]).reshape(batch, 2, 16))
                        for far_on, near_on in ((0, 0), (0, 1), (1, 0), (1, 1)):
                            outputs = []
                            for geo, assemble in ((a, old_assembly), (b, new_assembly)):
                                fm = tensor(np.broadcast_to(
                                    geo['far_pool_mask'][:, None, :] & bool(far_on),
                                    (batch, 2, 16)))
                                nm = tensor(np.broadcast_to(
                                    geo['near_pool_mask'][:, None, :] & bool(near_on),
                                    (batch, 2, 16)))
                                outputs.append(assemble(indices, indices, geo,
                                    tensor(np.arange(batch)), batch, 2, bs, 'cpu', fm, nm))
                            for x, y in zip(*outputs):
                                np.testing.assert_array_equal(x, y)
                            cases += 1
        self.assertEqual(cases, 540)


if importlib.util.find_spec('torch') is not None:
    import torch
else:
    torch = None
CUDA = torch is not None and torch.cuda.is_available()


@unittest.skipUnless(CUDA and BASELINE, 'requires CUDA and NFA_GEOMETRY_BASELINE')
class GeometryCudaTest(unittest.TestCase):
    def test_exact_eager_and_graph_replay_with_new_lengths(self):
        before, _ = helpers(BASELINE, torch)
        after, _ = helpers(SOURCE, torch)
        cfg = config(64, 4, 128, 4096)
        for batch in (1, 4, 32):
            for cap in (32768, 65536, 131072):
                with self.subTest(batch=batch, cap=cap):
                    lengths = torch.linspace(0, cap, batch,
                        device='cuda', dtype=torch.int64)
                    def run(fn):
                        return fn(None, None, lengths, cfg, 'cuda', cap // 64, cap)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run(before)
                            run(after)
                    torch.cuda.current_stream().wait_stream(stream)
                    graphs, outputs = [], []
                    for fn in (before, after):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out = run(fn)
                        graphs.append(graph)
                        outputs.append(out)
                    for delta in (0, 1, 63, 64, 129):
                        lengths.copy_((torch.arange(batch, device='cuda') * 65
                                       + delta) % (cap + 1))
                        for graph in graphs:
                            graph.replay()
                        eager = run(after)
                        for key in eager:
                            torch.testing.assert_close(outputs[0][key], eager[key],
                                                       rtol=0, atol=0)
                            torch.testing.assert_close(outputs[1][key], eager[key],
                                                       rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
