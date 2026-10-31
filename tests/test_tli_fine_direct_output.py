"""Source-bound host dispatch tests plus optional native CUDA parity/replay."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

SOURCE = Path(os.environ.get('TLI_SCORER_SOURCE',
    Path(__file__).resolve().parents[1] / 'sparse_attn/indexer/tli_indexer.py'))

def dispatch(source):
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_score')
    block = next(n for n in ast.walk(method) if isinstance(n, ast.Try) and
                 any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and
                     c.func.attr == '_fast_score_fine' and isinstance(c.args[0],ast.Call)
                     and isinstance(c.args[0].func,ast.Attribute)
                     and isinstance(c.args[0].func.value,ast.Name)
                     and c.args[0].func.value.id == 'k_qat'
                     for s in n.body for c in ast.walk(s)))
    return compile(ast.Module(body=[block], type_ignores=[]), str(source), 'exec')

class Tensor:
    def __init__(self, a, counts): self.a, self.counts = a, counts
    def __getitem__(self, key): return Tensor(self.a[key], self.counts)
    def __setitem__(self, key, value):
        self.counts['copies'] += 1
        self.a[key] = value.a
    def squeeze(self, dim): return self
    def to(self, dtype): return self

class HostDispatch(unittest.TestCase):
    def test_success_and_partial_failure(self):
        code = dispatch(SOURCE)
        direct = os.environ.get('TLI_EXPECT_LEGACY') != '1'
        for width in (1,17,64,65):
            for group in (1,4):
                for dtype in ('float16','bfloat16','float32'):
                    for failure in ('none','before','partial'):
                        with self.subTest(width=width,group=group,dtype=dtype,failure=failure):
                            counts = dict(copies=0,temporaries=0)
                            expected = np.arange(2*group*width).reshape(2*group,width).astype(np.float32)
                            output = Tensor(np.full((1,2*group,width),-np.inf),counts)
                            def fast(k,q,dt,out=None):
                                self.assertEqual(dt,dtype)
                                if out is None:
                                    counts['temporaries'] += 1
                                    out = Tensor(np.empty_like(expected),counts)
                                else:
                                    self.assertTrue(np.shares_memory(out.a,output.a))
                                    self.assertTrue(out.a.flags.c_contiguous)
                                if failure == 'before': raise RuntimeError('injected')
                                if failure == 'partial':
                                    out.a.ravel()[:max(1,out.a.size//2)] = 999
                                    raise RuntimeError('injected after writes')
                                out.a[:] = expected
                                return out
                            dummy = Tensor(np.zeros(1),counts)
                            env = dict(self=SimpleNamespace(_fast_score_fine=fast,group_size=group),
                                       score_fine=output,k_qat=dummy,b_q_full=dummy,
                                       q_sq=SimpleNamespace(dtype=dtype),
                                       torch=SimpleNamespace(float32='float32'),
                                       repeat=lambda *a,**k:dummy,
                                       einsum=lambda *a:Tensor(expected.copy(),counts))
                            exec(code,env)
                            self.assertTrue(np.array_equal(output.a[0],expected))
                            self.assertEqual(counts['temporaries'],0 if direct else 1)
                            self.assertEqual(counts['copies'],int(failure!='none') if direct else 1)

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False

@unittest.skipUnless(CUDA, 'native CUDA required')
class NativeOutput(unittest.TestCase):
    def test_exact_output_and_graph_replay(self):
        from sparse_attn.indexer.tli_indexer import TLIIndexer
        idx = TLIIndexer.__new__(TLIIndexer)
        idx._triton_ok = True
        torch.manual_seed(22)
        with torch.inference_mode():
            for dtype in (torch.float16,torch.bfloat16,torch.float32):
                shapes = [(1,32,1),(17,64,4),(65,128,4)]
                if os.environ.get('TLI_NATIVE_BENCH') == '1':
                    shapes += [(8193,32,4),(32769,32,4),(131073,32,4)]
                for width,dim,group in shapes:
                    k = torch.randn(width,2,dim,device='cuda',dtype=dtype)
                    q = torch.randn(2*group,dim,device='cuda',dtype=torch.float32)
                    out = torch.full((2*group,width),float('nan'),device='cuda',dtype=dtype)
                    ref = idx._fast_score_fine(k,q,dtype)
                    ret = idx._fast_score_fine(k,q,dtype,out=out)
                    self.assertIs(ret,out)
                    torch.testing.assert_close(out,ref,rtol=0,atol=0)
                    full = torch.full((1,2*group,width),float('nan'),device='cuda',dtype=dtype)
                    exec(dispatch(SOURCE),dict(self=idx,score_fine=full,
                        k_qat=k.unsqueeze(0),b_q_full=q,q_sq=SimpleNamespace(dtype=dtype),torch=torch))
                    torch.testing.assert_close(full[0],ref,rtol=0,atol=0)
                    # Warm up on a side stream before capture, then change inputs.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        idx._fast_score_fine(k,q,dtype,out=out)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        idx._fast_score_fine(k,q,dtype,out=out)
                    for _ in range(3):
                        k.add_(.03125)
                        graph.replay()
                        ref = idx._fast_score_fine(k,q,dtype)
                        torch.testing.assert_close(out,ref,rtol=0,atol=0)
                    if os.environ.get('TLI_NATIVE_BENCH') == '1':
                        # Paired scorer+copy microbenchmark, not selector or TPOT.
                        import statistics
                        timings = {'temporary_copy': [], 'direct': []}
                        for pair in range(6):
                            for name in (('temporary_copy','direct') if pair%2==0 else ('direct','temporary_copy')):
                                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(50):
                                    if name == 'direct': idx._fast_score_fine(k,q,dtype,out=out)
                                    else: out.copy_(idx._fast_score_fine(k,q,dtype))
                                end.record();end.synchronize()
                                timings[name].append(start.elapsed_time(end)/50)
                        print(dict(dtype=str(dtype),width=width,dim=dim,group=group,
                                   microbench_ms=timings,median_ms={k:statistics.median(v) for k,v in timings.items()}))

if __name__ == '__main__': unittest.main()
