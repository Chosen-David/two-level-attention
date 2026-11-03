"""Source-bound host checks plus optional native CPU/CUDA pairing."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('NFA_CORE_SOURCE', ROOT / 'sparse_attn/policies/frozen_gate.py'))

def load_core(backend):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_core_tokens_from_probabilities')
    env = {'torch': backend}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[fn.name]

env = {}
exec((ROOT / 'sparse_attn/policies/region_geometry.py').read_text(), env)
Geometry = env['RegionGeometry']

def reference(probs, geometry, topk, backend):
    length = probs.shape[-1]
    scores = probs.mean(dim=(0, 1)).mean(dim=0)
    sink, far, near = geometry.bounds(length)
    ids = backend.arange(length, device=probs.device)
    result = {}
    for name, lo, hi in [('far', sink, far), ('near', far, near)]:
        positions = ids[(ids >= lo) & (ids < hi)]
        result[name] = positions[backend.topk(scores[positions], min(topk, positions.numel())).indices].cpu().tolist()
    return result

class Tensor(np.ndarray):
    @property
    def device(self): return 'cpu'
    def mean(self, dim=None): return np.asarray(self).mean(axis=dim).view(Tensor)
    def numel(self): return self.size
    def cpu(self): return self

class Host:
    aranges = 0
    def arange(self, n, **kw):
        self.aranges += 1
        return np.arange(n).view(Tensor)
    def topk(self, x, k):
        if k < 0: raise RuntimeError('negative k')
        return NS(indices=np.argsort(-np.asarray(x), kind='stable')[:k].view(Tensor))
    def cat(self, xs): return np.concatenate(xs).view(Tensor)

class CoreSlices(unittest.TestCase):
    def test_host_pairing(self):
        rng = np.random.default_rng(20261103)
        backend = Host(); fn = load_core(backend)
        for length in (1, 7, 64, 257, 513):
            for heads in (1, 4, 8):
                for pattern in ('random', 'ties', 'zeros'):
                    a = rng.random((1, heads, min(13,length), length), dtype=np.float32)
                    if pattern == 'ties': a.fill(1 / length)
                    if pattern == 'zeros': a.fill(0)
                    p = a.view(Tensor)
                    for sink in (0, 4, length+1):
                        g = Geometry(sink=sink, near=17, window=8, page=4)
                        for k in (0, 1, 64):
                            self.assertEqual(fn(p,g,k), reference(p,g,k,backend))

    def test_no_full_length_position_allocation(self):
        backend = Host()
        load_core(backend)(np.ones((1,4,3,257),dtype=np.float32).view(Tensor), Geometry(),64)
        self.assertEqual(backend.aranges, 0)

    def native(self, device):
        import torch
        fn=load_core(torch)
        torch.manual_seed(20261103)
        for length in (1,7,64,257,513,32768):
            for pattern in ('random','ties','zeros'):
                p=torch.rand((1,4,min(13,length),length),device=device)
                if pattern=='ties': p.fill_(1/length)
                if pattern=='zeros': p.zero_()
                for sink in (0,4,length+1):
                    g=Geometry(sink=sink,near=17,window=8,page=4)
                    for k in (0,1,64):
                        self.assertEqual(fn(p,g,k),reference(p,g,k,torch))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable')
    def test_native_cpu(self): self.native('cpu')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable')
    def test_native_cuda(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.native('cuda')

if __name__=='__main__': unittest.main()
