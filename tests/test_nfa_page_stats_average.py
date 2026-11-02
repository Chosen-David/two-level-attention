"""Rounded-sum fusion checks. Host emulation is not native GPU evidence.

NFA_SELECTOR_SOURCE overrides the tested source; NFA_STATS_BASELINE points to
the unmodified main selector for native eager/Graph paired checks.
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
BASELINE = os.environ.get('NFA_STATS_BASELINE')


def function(path, name):
    nodes = [n for n in ast.walk(ast.parse(Path(path).read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f'expected one {name}')
    return nodes[0]


def cast(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        if dtype == 'bf16':
            bits = values.view(np.uint32)
            # Round-to-nearest-even; preserve NaNs as NaNs.
            rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1))
                       & np.uint32(0xffff0000))
            rounded = np.where(np.isnan(values), np.uint32(0x7fc00000), rounded)
            return rounded.astype(np.uint32).view(np.float32)
        return values.astype(dtype).astype(np.float32)


class Value:
    def __init__(self, values): self.values = np.asarray(values, dtype=np.float32)
    def to(self, dtype): return Value(cast(self.values, dtype))
    def __truediv__(self, divisor): return Value(self.values / divisor)


class AverageHostTest(unittest.TestCase):
    def test_wrapper_has_no_post_launch_tensor_pass(self):
        events = []
        class Tensor:
            device = 'cpu'
            dtype = np.float16
            def numel(self): return 32
            def to(self, dtype): events.append('cast'); return self
            def __truediv__(self, other): events.append('divide'); return self
        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs): events.append('stats')
                return launch
        backend = NS(int64=np.int64, int32=np.int32, float32=np.float32,
            is_tensor=lambda x: False, as_tensor=lambda x, **kw: x,
            empty=lambda *a, **kw: Tensor(), empty_like=lambda x: Tensor())
        env = dict(torch=backend, triton=NS(next_power_of_2=lambda n: 32),
                   _has_triton=lambda: True, _nfa_page_stats_kernel=Kernel())
        node = function(SOURCE, '_page_stats_all_layers')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
        cache = NS(layers=[NS(kv_pool_k=Tensor(), num_kv_heads=2, head_dim_k=128)],
            block_table_buf=None, skip_far_layer_buf=None, skip_near_layer_buf=None,
            layer_kv_ptrs_buf=[0], max_blocks=2048, max_requests=4,
            geometry=NS(page=64, sink=4, near=4096, window=128))
        out = env['_page_stats_all_layers'](cache, [0, 1, 2, 3], [0],
                                           Tensor(), 64, 131072, [1, 2, 3, 4])
        self.assertEqual(len(out), 3)
        self.assertEqual(events, ['stats'])

    def test_actual_fused_store_preserves_rounded_sum(self):
        kernel = function(SOURCE, '_nfa_page_stats_kernel')
        # Execute the actual store expression, not a copied candidate formula.
        start = next(i for i, n in enumerate(kernel.body)
                     if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == 'rounded_sum')
        code = compile(ast.Module(body=kernel.body[start:], type_ignores=[]),
                       str(SOURCE), 'exec')
        rng = np.random.default_rng(20261103)
        raw = rng.integers(0, 2**32, 100000, dtype=np.uint32).view(np.float32)
        # Exercise every BF16 bit pattern, halfway neighborhoods, zeros,
        # infinities, cancellation and the FP16 sum-overflow boundary.
        bf = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
        values = np.concatenate([raw, bf, np.array([
            0., -0., 65504., 65520., 65536., -65536., 1e-40, -1e-40,
            64 * 2048., -64 * 2048., np.inf, -np.inf, np.nan], np.float32)])
        for dtype in ('bf16', np.float16, np.float32):
            for bs in (32, 64, 128):
                with self.subTest(dtype=str(dtype), bs=bs), np.errstate(all='ignore'):
                    captured = []
                    class Pointer:
                        def __init__(self): self.dtype = NS(element_ty=dtype)
                        def __add__(self, other): return self
                    env = dict(vsum=Value(values), OUT_AVG=Pointer(), BS=bs,
                        out_base=0, dim_mask=True,
                        tl=NS(float32=np.float32, store=lambda p, v, **kw:
                              captured.append(v.values)))
                    exec(code, env)
                    expected = cast(cast(values, dtype) / np.float32(bs), dtype)
                    actual = captured[0]
                    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                    finite_or_inf = ~np.isnan(expected)
                    np.testing.assert_array_equal(actual[finite_or_inf].view(np.uint32),
                        expected[finite_or_inf].view(np.uint32))
        # Do not silently turn this optimization into an overflow fix.
        rounded = cast(cast([131072.], np.float16) / 64, np.float16)
        direct = cast(np.array([131072.], np.float32) / 64, np.float16)
        self.assertTrue(np.isinf(rounded[0]))
        self.assertEqual(direct[0], 2048.)


if importlib.util.find_spec('torch') is not None:
    import torch
else:
    torch = None
CUDA = (torch is not None and torch.cuda.is_available()
        and importlib.util.find_spec('triton') is not None)


@unittest.skipUnless(CUDA and BASELINE, 'requires CUDA/Triton and NFA_STATS_BASELINE')
class AverageCudaTest(unittest.TestCase):
    def test_paired_stats_eager_graph_and_metadata_replay(self):
        import sys
        modules = []
        for suffix, path in (('old', BASELINE), ('new', SOURCE)):
            name = 'sparse_attn.indexer._average_test_' + suffix
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        torch.manual_seed(20261103)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for batch in (1, 4, 32):
                for bs, dim_numel in ((32, 31), (64, 32), (128, 64)):
                    with self.subTest(dtype=dtype, batch=batch, bs=bs, dims=dim_numel):
                        nb, hk, dim = 8, 2, 64
                        pools = [torch.randn(batch * nb, hk, bs, dim,
                                 device='cuda', dtype=dtype) for _ in range(2)]
                        # Exposes removal of the low-precision sum round trip.
                        pools[1][0, :, :, 0] = 2048
                        layers = [NS(kv_pool_k=p, num_kv_heads=hk, head_dim_k=dim)
                                  for p in pools]
                        cache = NS(layers=layers, max_blocks=nb, max_requests=batch,
                            geometry=NS(page=bs, sink=4, near=2*bs, window=bs),
                            layer_kv_ptrs_buf=torch.tensor([p.data_ptr() for p in pools],
                                device='cuda', dtype=torch.int64),
                            block_table_buf=torch.arange(batch*nb, device='cuda',
                                dtype=torch.int32).reshape(1,batch,nb).repeat(2,1,1),
                            skip_far_layer_buf=torch.zeros(2,batch,device='cuda',dtype=torch.int32),
                            skip_near_layer_buf=torch.zeros(2,batch,device='cuda',dtype=torch.int32))
                        req = torch.arange(batch-1,-1,-1,device='cuda',dtype=torch.int64)
                        lengths = torch.full((batch,),nb*bs,device='cuda',dtype=torch.int32)
                        dims = torch.arange(dim_numel,device='cuda',dtype=torch.int64)
                        def run(mod):
                            return mod._page_stats_all_layers(cache, req,
                                [pools[1].data_ptr()],dims,bs,nb*bs,lengths,layer_offset=1)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                for mod in modules: run(mod)
                        torch.cuda.current_stream().wait_stream(stream)
                        graphs, outputs = [], []
                        for mod in modules:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph): out = run(mod)
                            graphs.append(graph); outputs.append(out)
                        for step in range(12):
                            # Empty/padded rows, unequal lengths, boundary movement,
                            # request-slot/page reuse and all four region gates.
                            lengths.copy_((torch.arange(batch,device='cuda')*65
                                           + step*bs//2) % (nb*bs+1))
                            rows = torch.arange(batch, device='cuda') + step
                            cache.skip_far_layer_buf[1].copy_(rows % 2)
                            cache.skip_near_layer_buf[1].copy_((rows//2) % 2)
                            if step % 3 == 0:
                                cache.block_table_buf.copy_(cache.block_table_buf.roll(1,-1))
                            for graph in graphs: graph.replay()
                            reference = run(modules[0])
                            eager = run(modules[1])
                            for result in [*outputs, eager]:
                                for a, b in zip(reference, result):
                                    torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
                        del graphs, outputs, cache, pools, layers


if __name__ == '__main__':
    unittest.main()
