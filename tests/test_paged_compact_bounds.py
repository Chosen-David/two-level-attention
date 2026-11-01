"""Host AST guard checks plus optional real Torch wrapper regression.

Host tests execute the actual wrapper prefix through its validation gates;
they do not execute Triton or claim GPU correctness/performance.
"""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

SOURCE = Path(os.environ.get("COMPACT_GUARD_SOURCE",
    Path(__file__).resolve().parents[1] / "sparse_attn/ops/paged_region_kernels.py"))


class Tensor(np.ndarray):
    device = "cpu"
    def clamp(self, lo, hi):
        return np.clip(self, lo, hi).view(Tensor)


def tensor(x, dtype=np.int32):
    return np.asarray(x, dtype=dtype).view(Tensor)


def guard():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "paged_attention_selected")
    end = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and any(isinstance(t,ast.Name) and t.id == "K" for t in n.targets))
    fn.body = fn.body[:end] + [ast.Return(ast.Constant(True))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    fake = SimpleNamespace(int32=np.dtype("int32"), bool=np.dtype("bool"),
        zeros=lambda shape, **kw: tensor(np.zeros(shape), kw["dtype"]),
        arange=lambda n, **kw: tensor(np.arange(n)), any=np.any)
    ns = {"torch":fake}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns["paged_attention_selected"]


class HostGuards(unittest.TestCase):
    def setUp(self):
        self.check = guard()
        self.k = tensor(np.zeros((1,2,8,2)), np.float32)
        self.q = tensor(np.zeros((1,4,1,2)), np.float32)
        self.layer = SimpleNamespace(logical_length=8,
            pages={0:(0,self.k,self.k)}, intervals=lambda:[(0,2),(6,8)])

    def call(self, ids, counts):
        return self.check(self.layer,self.q,
            (tensor(ids)[None],tensor(counts)[None]),1.0)

    def test_active_negative_id(self):
        with self.assertRaises(ValueError):
            self.call([[-1,1],[0,1]],[2,2])

    def test_active_past_end_id(self):
        for bad in (8,64,2**31-1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.call([[0,bad],[0,1]],[2,2])

    def test_negative_count(self):
        with self.assertRaises(ValueError):
            self.call([[0,1],[0,1]],[-1,2])

    def test_count_exceeds_width(self):
        with self.assertRaises(ValueError):
            self.call([[0,1],[0,1]],[3,2])

    def test_zero_count_still_rejected(self):
        with self.assertRaises(ValueError):
            self.call([[0,1],[0,1]],[0,2])

    def test_hidden_id_still_rejected(self):
        with self.assertRaises(ValueError):
            self.call([[0,3],[0,1]],[2,2])

    def test_inactive_garbage_is_ignored(self):
        self.assertTrue(self.call([[0,-1,999],[0,1,-99]],[1,2]))

    def test_valid_geometry_pairs(self):
        rng=np.random.default_rng(71)
        for L in (1,2,7,8,31,128,131072):
            for sink in (0,1,4):
                for window in (1,3,128):
                    s=min(sink,L);w=max(s,L-window)
                    spans=[(a,b) for a,b in ((0,s),(w,L)) if a<b]
                    visible=[i for a,b in spans for i in range(a,b)]
                    self.layer.logical_length=L
                    self.layer.intervals=lambda spans=spans:spans
                    for width in (1,4,17):
                        counts=rng.integers(1,width+1,size=2)
                        ids=np.full((2,width),-123,dtype=np.int32)
                        for h in range(2):
                            ids[h,:counts[h]]=rng.choice(visible,size=counts[h])
                        self.assertTrue(self.call(ids,counts))


try:
    import torch
except ImportError:
    torch=None


@unittest.skipUnless(torch is not None, "requires real Torch")
class NativeWrapper(unittest.TestCase):
    def _check_device(self, device):
        spec=importlib.util.spec_from_file_location("_paged_guard",SOURCE)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        k=torch.arange(32,dtype=torch.float32,device=device).reshape(1,2,8,2)/32
        layer=SimpleNamespace(logical_length=8,pages={0:(0,k,k)},
            intervals=lambda:[(0,2),(6,8)],geometry=SimpleNamespace(page=8))
        q=torch.ones((1,4,1,2),device=device)
        valid=torch.tensor([[[0,1],[6,7]]],dtype=torch.int32,device=device)
        count=torch.tensor([[2,2]],dtype=torch.int32,device=device)
        out=m.paged_attention_selected(layer,q,(valid,count),0.5)
        refs=[]
        for h in range(4):
            keys=k[0,h//2,valid[0,h//2].long()]
            prob=torch.softmax((keys*q[0,h,0]).sum(-1)*0.5,dim=0)
            refs.append((prob[:,None]*keys).sum(0))
        torch.testing.assert_close(out,torch.stack(refs).reshape(1,1,4,2))
        for bad in (-1,8,999):
            ids=valid.clone();ids[0,0,1]=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                m.paged_attention_selected(layer,q,(ids,count),0.5)
        for bad in (-1,0,3):
            counts=count.clone();counts[0,0]=bad
            with self.subTest(count=bad),self.assertRaises(ValueError):
                m.paged_attention_selected(layer,q,(valid,counts),0.5)

    def test_cpu_wrapper(self):
        self._check_device("cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available()
                         and importlib.util.find_spec("triton") is not None,
                         "requires CUDA and Triton")
    def test_cuda_wrapper(self):
        self._check_device("cuda")


if __name__ == "__main__":
    unittest.main()
