"""Source-bound protected ID regression; native CPU/CUDA tests are optional."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

PATH = Path(os.environ.get('NFA_PROTECTED_SOURCE', Path(__file__).resolve().parents[1] / 'sparse_attn/ops/paged_region_kernels_nfa.py'))
def load_function(torch):
    tree = ast.parse(PATH.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_protected_selection_batch')
    env = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(PATH), 'exec'), env)
    return env[fn.name]

class Tensor(np.ndarray):
    def __new__(cls, x): return np.asarray(x).view(cls)
    def to(self, dtype): return Tensor(self.astype(dtype))
    def expand(self, *shape): return Tensor(np.broadcast_to(self, tuple(a if b == -1 else b for a,b in zip(self.shape,shape))))

def host_backend():
    return NS(int32=np.int32, long=np.int64,
        tensor=lambda x, **kw: Tensor(np.array(x, dtype=kw.get('dtype'))),
        zeros=lambda s, **kw: Tensor(np.zeros(s, dtype=kw.get('dtype'))),
        arange=lambda n, **kw: Tensor(np.arange(n, dtype=kw.get('dtype'))),
        where=lambda c,a,b: Tensor(np.where(c,a,b)),
        minimum=lambda a,b: Tensor(np.minimum(a,b)),
        full_like=lambda x,v: Tensor(np.full_like(x,v)), zeros_like=lambda x: Tensor(np.zeros_like(x)))

def cases():
    for sink in [0,1,4,16,64]:
        for window in [0,1,7,32,128]:
            edge = sink + window
            for short in sorted({0,1,max(0,sink-1),sink,sink+1,max(0,edge-1),edge,edge+1}):
                for lengths in ([short], [edge+17,short], [short,edge+17,1], [0,0]):
                    yield sink,window,list(lengths)

def fixture(lengths, device):
    ids=[7,2,5][:len(lengths)]
    states={rid:NS(logical_length=max(0,n-1)) for rid,n in zip(ids,lengths)}
    pending=dict(zip(ids,lengths))
    # Invalid slot and removed request exercise zero-length rows.
    ids += [-1,6]
    cache=NS(states=states,pending_lengths=pending,gather_indices_buf=NS(device=device))
    return cache,ids

def oracle(length,sink,window):
    return sorted(set(range(min(sink,length))) | set(range(max(0,length-window),length)))

class ProtectedBatchTests(unittest.TestCase):
    def verify(self, backend, device, native=False):
        fn=load_function(backend);ncase=0
        for sink,window,lengths in cases():
            cache,ids=fixture(lengths,device)
            output,counts=fn(cache,0,ids,NS(sink_tokens=sink,sliding_window_size=window))
            if native: output,counts=output.cpu().numpy(),counts.cpu().numpy()
            for row,length in enumerate(lengths+[0,0]):
                expected=oracle(length,sink,window)
                self.assertEqual(int(counts[row]),len(expected),(sink,window,lengths,row))
                self.assertEqual(output[row,:int(counts[row])].tolist(),expected,(sink,window,lengths,row))
            ncase+=1
        return ncase
    def test_host_actual_function(self):
        self.assertGreater(self.verify(host_backend(),'cpu'),500)
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable')
    def test_native_cpu(self):
        import torch
        self.verify(torch,'cpu',True)
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable')
    def test_native_cuda_and_capture(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.verify(torch,'cuda',True)
        fn=load_function(torch);cache,ids=fixture([1024,100,2],'cuda');idx=NS(sink_tokens=4,sliding_window_size=128)
        expected=fn(cache,0,ids,idx)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn(cache,0,ids,idx)
        torch.cuda.current_stream().wait_stream(stream)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): captured=fn(cache,0,ids,idx)
        for _ in range(3):
            g.replay()
            for a,b in zip(captured,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__ == '__main__': unittest.main()
