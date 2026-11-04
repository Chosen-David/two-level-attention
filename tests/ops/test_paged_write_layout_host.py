"""Source-bound packed-write regression; NumPy execution is not CUDA validation.

Run directly, optionally with NFA_WRITE_SOURCE pointing at the before source.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(os.environ.get('NFA_WRITE_SOURCE',
    Path(__file__).resolve().parents[2] / 'sparse_attn/ops/paged_region_kernels.py'))


class Tensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.device = 'cuda'
        self.is_cuda = True

    def contiguous(self):
        return self if self.array.flags.c_contiguous else Tensor(
            np.ascontiguousarray(self.array))


class Pointer:
    def __init__(self, array, offset=0):
        self.array, self.offset = array, offset

    def __add__(self, offset):
        return Pointer(self.array, self.offset + offset)


def pointer(tensor):
    a = tensor.array
    root = a
    while isinstance(root.base, np.ndarray):
        root = root.base
    offset = (a.ctypes.data - root.ctypes.data) // a.itemsize
    return Pointer(root.reshape(-1), offset)


class TL:
    int64 = np.int64
    pid = 0
    program_id = lambda self, axis: self.pid
    arange = staticmethod(np.arange)
    static_range = staticmethod(range)
    where = staticmethod(np.where)

    @staticmethod
    def load(p, mask=True, other=0):
        idx = np.asarray(p.offset)
        return np.where(mask, p.array[np.where(mask, idx, 0)], other)

    @staticmethod
    def store(p, value, mask=True):
        idx, value, mask = np.broadcast_arrays(p.offset, value, mask)
        p.array[idx[mask]] = value[mask]


class HostAST(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
            return ast.Call(ast.Name('cast', ast.Load()),
                            [node.func.value, *node.args], [])
        return node


def load_writer():
    tree = ast.parse(SOURCE.read_text())
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and n.name in ('_paged_write_kernel', 'paged_write_decode')]
    module = ast.fix_missing_locations(HostAST().visit(ast.Module(funcs, [])))
    tl = TL()
    ns = dict(tl=tl, triton=SimpleNamespace(next_power_of_2=lambda x: 1 << (x-1).bit_length()),
              cast=lambda x, dtype: np.asarray(x, dtype=dtype))
    launches = []

    def launch(workspace, key, kernel, grid, *args):
        launches.append(args)
        converted = [pointer(a) if isinstance(a, Tensor) else a for a in args]
        for tl.pid in range(grid[0]):
            kernel(*converted)

    ns['_fast_launch'] = launch
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns['paged_write_decode'], launches


class PackedWriteTest(unittest.TestCase):
    def check_layout(self, layout):
        write, launches = load_writer()
        cases = 0
        for b in (1, 3, 8):
            for hk in (1, 2, 8):
                for d, dv in ((16, 16), (24, 32), (64, 128)):
                    k = np.arange(b * hk * d, dtype=np.float32).reshape(b, hk, 1, d)
                    v = np.arange(b * hk * dv, dtype=np.float32).reshape(b, hk, 1, dv) + 5000
                    if layout == 'glm':
                        joint = np.full((b, 1, hk, d + dv), -1000., np.float32)
                        joint[..., d:] = v.transpose(0, 2, 1, 3)
                        v = joint.transpose(0, 2, 1, 3)[..., d:]
                    elif layout == 'feature':
                        kb = np.full((b, hk, 1, 2*d), -2000., np.float32)
                        vb = np.full((b, hk, 1, 2*dv), -3000., np.float32)
                        kb[..., ::2], vb[..., ::2] = k, v
                        k, v = kb[..., ::2], vb[..., ::2]
                    keys, vals = Tensor(k), Tensor(v)
                    page = 4
                    slots = (np.arange(b, dtype=np.int32)*5 + 7) % (9*page) + page
                    if b > 1:
                        slots[-1] = -1
                    kp = np.full((11, hk, page, d), -99., np.float32)
                    vp = np.full((11, hk, page, dv), -99., np.float32)
                    expected_k, expected_v = kp.copy(), vp.copy()
                    for row, slot in enumerate(slots):
                        if slot >= 0:
                            pp, off = divmod(int(slot), page)
                            expected_k[pp, :, off] = k[row, :, 0]
                            expected_v[pp, :, off] = v[row, :, 0]
                    layer = SimpleNamespace(num_kv_heads=hk, head_dim_k=d,
                        head_dim_v=dv, geometry=SimpleNamespace(page=page),
                        PAD_PAGE_ID=0, kv_pool_k=Tensor(kp), kv_pool_v=Tensor(vp))
                    result = write(layer, keys, vals, Tensor(slots))
                    self.assertIs(result[0], keys)
                    self.assertIs(result[1], vals)
                    np.testing.assert_array_equal(kp, expected_k)
                    np.testing.assert_array_equal(vp, expected_v)
                    if layout == 'packed':
                        self.assertIs(launches[-1][0], keys)
                        self.assertIs(launches[-1][1], vals)
                    cases += 1
        self.assertEqual(cases, 27)

    def test_glm_split_view(self):
        self.check_layout('glm')

    def test_feature_stride(self):
        self.check_layout('feature')

    def test_packed_zero_copy(self):
        self.check_layout('packed')


if __name__ == '__main__':
    unittest.main()
