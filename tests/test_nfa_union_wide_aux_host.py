"""final union 的 host AST 整数逻辑与可选 native eager/Graph 对拍。

Host 模式不验证 Triton lowering 或性能；`--cuda` 模式额外执行真实 kernel。
"""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np


SOURCE = Path(os.environ.get('NFA_SELECTOR_SOURCE', str(
    Path(__file__).resolve().parents[1] /
    'sparse_attn/indexer/nfa_layer_batch_selector.py')))


class A(np.ndarray):
    def to(self, dtype):
        return np.asarray(self, dtype=dtype).view(A)


def ar(value):
    return np.asarray(value).view(A)


class P:
    def __init__(self, array, offset=0):
        self.array = array
        self.offset = offset

    def __add__(self, index):
        return P(self.array, self.offset + index)


def load(pointer, mask=True, other=0):
    indices, mask = np.broadcast_arrays(pointer.offset, mask)
    out = np.full(indices.shape, other, pointer.array.dtype)
    selected = indices[mask]
    if np.any(selected < 0) or np.any(selected >= len(pointer.array)):
        raise IndexError('pointer bounds')
    out[mask] = pointer.array[selected]
    return ar(out)


def store(pointer, value, mask=True):
    indices, mask = np.broadcast_arrays(pointer.offset, mask)
    if np.any(indices[mask] < 0) or np.any(indices[mask] >= len(pointer.array)):
        raise IndexError('store bounds')
    pointer.array[indices[mask]] = np.broadcast_to(value, indices.shape)[mask]


def gather(values, indices, axis=0):
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise IndexError('gather index outside run')
    return ar(np.take(values, indices, axis=axis))


def kernel_ast(path, strip_annotations=True):
    tree = ast.parse(Path(path).read_text())
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == '_final_union_physical_kernel')
    function.decorator_list = []
    if strip_annotations:
        for argument in function.args.args:
            argument.annotation = None
    return function


def build(path):
    function = kernel_ast(path)
    tl = NS(
        program_id=lambda index: 0, load=load, store=store,
        arange=lambda start, stop: ar(np.arange(start, stop)),
        minimum=lambda a, b: ar(np.minimum(a, b)),
        maximum=lambda a, b: ar(np.maximum(a, b)),
        where=lambda *args: ar(np.where(*args)), gather=gather,
        sort=lambda value, descending=False: ar(np.sort(value)),
        full=lambda shape, value, dtype: ar(np.full(shape, value, dtype)),
        static_range=range,
        cumsum=lambda value, axis: ar(np.cumsum(value, axis=axis)),
        sum=lambda value, axis: ar(np.sum(value, axis=axis)), int32=np.int32)
    env = {'tl': tl}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), env)
    return env[function.name]


def pow2(value):
    return 1 << (max(1, value) - 1).bit_length()


def native_build(path):
    import importlib.util
    import tempfile
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent host fallback')
    function = kernel_ast(path, strip_annotations=False)
    temporary = tempfile.TemporaryDirectory()
    file = Path(temporary.name) / 'union_kernel.py'
    file.write_text(
        'import triton\nimport triton.language as tl\n@triton.jit\n'
        + ast.unparse(function) + '\n')
    spec = importlib.util.spec_from_file_location('union_kernel', file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['union_kernel'] = module
    spec.loader.exec_module(module)
    kernel = module._final_union_physical_kernel

    def invoke(*pointers, **kwargs):
        tensors = [torch.from_numpy(pointer.array.copy()).cuda()
                   for pointer in pointers]
        kernel[(1,)](*tensors, **kwargs, num_warps=8)
        torch.cuda.synchronize()
        eager = [tensor.cpu().numpy().copy() for tensor in tensors[-2:]]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel[(1,)](*tensors, **kwargs, num_warps=8)
        tensors[-2].fill_(-9)
        tensors[-1].zero_()
        graph.replay()
        torch.cuda.synchronize()
        for pointer, tensor, expected in zip(pointers[-2:], tensors[-2:], eager):
            actual = tensor.cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
            pointer.array[:] = actual

    invoke._temporary = temporary
    return invoke


def run_cases(function):
    rng = np.random.default_rng(731)
    cases = []
    for swa in (0, 128, 896, 1024):
        for seedcap in (0, 128, 129, 892, 1024, 2048):
            for seeded in (0, 1):
                for active in (0, -1):
                    cases.append((swa, seedcap, seeded, active, False))
    # protected-only 使用 W=aux_width；即使 A_PAD>W，也不能回归既有完整集。
    cases.extend(((128, 128, 0, 0, True), (1024, 0, 0, 0, True)))
    for case_index, (swa, seedcap, seeded, active, empty) in enumerate(cases):
        cap, length, sink = 16384, 8192, 4
        prot = sink + swa
        w = prot + seedcap if empty else 1024
        n = 2 * w if empty else w + prot + seedcap
        npad, apad = pow2(n), pow2(prot + seedcap)
        middle = np.sort(rng.choice(
            np.arange(sink, length - swa), size=w, replace=False)).astype(np.int32)
        seeds = rng.integers(sink, length, size=seedcap, dtype=np.int32)
        table = rng.permutation(cap // 64).astype(np.int32)
        out = np.full(n, -9, np.int32)
        visible = np.zeros(1, np.int32)
        arrays = (middle, np.array([0 if empty else w], np.int32),
                  np.array([length], np.int32), np.array([active], np.int32),
                  np.array([0], np.int32), np.array([seeded], np.int32), seeds,
                  np.array([seedcap], np.int32), table, out, visible)
        try:
            function(*[P(array) for array in arrays],
                     W=w, PROT=prot, MAX_SEED=seedcap, N=n, N_PAD=npad,
                     A_PAD=apad, MERGE_BITS=npad.bit_length() - 1,
                     SINK=sink, SWA=swa, CAP=cap, MAX_BLOCKS=cap // 64,
                     PAGE=64, MAX_REQ=1, MAX_VIS=n, INVALID=-1, HKV=1, B=1,
                     LAYER_OFFSET=0, EMPTY=empty)
            expected = set(range(sink)) | set(range(length - swa, length))
            expected |= set() if empty else set(middle)
            if seeded == 0 and active >= 0:
                expected |= set(seeds)
            expected = sorted(expected)
            physical = [int(table[token // 64]) * 64 + token % 64
                        for token in expected]
            np.testing.assert_array_equal(out[:len(expected)], physical)
            np.testing.assert_array_equal(out[len(expected):], -1)
            np.testing.assert_array_equal(visible, [len(expected)])
        except Exception as error:
            raise AssertionError(
                f'case {case_index}: swa={swa}, seedcap={seedcap}, '
                f'seeded={seeded}, active={active}, empty={empty}, '
                f'W={w}, A_PAD={apad}') from error
    return len(cases)


def test_wide_aux_union_host_grid():
    assert run_cases(build(SOURCE)) == 98


if __name__ == '__main__':
    source = next((arg for arg in sys.argv[1:] if arg != '--cuda'), SOURCE)
    function = native_build(source) if '--cuda' in sys.argv else build(source)
    count = run_cases(function)
    mode = 'native eager and CUDA Graph' if '--cuda' in sys.argv else 'AST host integer logic'
    print(f'{count} cases pass; {mode}')
