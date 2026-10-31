"""Actual-source host facade audit, NOT native Torch/CUDA performance evidence."""
import ast
import heapq
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / 'sparse_attn/ops'
def extract(file, names, ns):
    source = Path(os.environ['TLI_PAGED_RECLAIM_SOURCE']) if file == 'paged_region_cache.py' and 'TLI_PAGED_RECLAIM_SOURCE' in os.environ else ROOT/file
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), ns)

class Tensor:
    clones = copies = 0
    fail_after_copy = None
    def __init__(self, a):
        self.a = a
        self.device = SimpleNamespace(type='cpu')
        self.dtype = a.dtype
        self.requires_grad = False
    @property
    def shape(self): return self.a.shape
    @property
    def ndim(self): return self.a.ndim
    def __getitem__(self, x): return Tensor(self.a[x])
    def clone(self):
        Tensor.clones += 1
        return Tensor(self.a.copy())
    def copy_(self, src):
        self.a[...] = src.a
        Tensor.copies += 1
        if Tensor.fail_after_copy == Tensor.copies:
            Tensor.fail_after_copy = None
            raise RuntimeError('injected copy failure')
        return self
    def element_size(self): return self.a.itemsize
    def numel(self): return self.a.size

fake_torch = SimpleNamespace(empty=lambda shape, dtype, device: Tensor(np.full(shape, -999, dtype=dtype)))
ns = {'dataclass': dataclass, 'torch': fake_torch, 'heapq': heapq, 'CacheLayerMixin': object}
extract('frozen_region_cache.py', {'RegionGeometry'}, ns)
extract('paged_region_cache.py', {'PagedRegionLayer'}, ns)
Layer, Geometry = ns['PagedRegionLayer'], ns['RegionGeometry']

class CountingDict(dict):
    def __iter__(self):
        self.visits += len(self)
        return super().__iter__()
    visits = 0

def check(condition, text):
    if not condition: raise AssertionError(text)

def run_paged():
    results=[]
    for page in (4,16,64):
        for near, far in ((False,False),(True,False),(False,True),(True,True)):
            for sink in (0,2):
                g=Geometry(sink=sink,near=page*2,window=page+1,page=page,max_new_tokens=page*3)
                layer=Layer(g)
                initial=page*5+1
                x=Tensor(np.arange(initial*8,dtype=np.float32).reshape(1,2,initial,4))
                layer.update(x,x);layer.freeze(near,far)
                layer.pages=CountingDict(layer.pages)
                Tensor.clones=Tensor.copies=0
                history=x.a.copy();expected_backups=0
                before_reclaimed=layer.reclaimed_pages
                for step in range(page*3):
                    length=layer.logical_length
                    if length%page: expected_backups+=2
                    incoming=Tensor(np.full((1,2,1,4),step+1000,dtype=np.float32))
                    layer.update(incoming,incoming)
                    history=np.concatenate((history,incoming.a),axis=2)
                    for lp,(_,k,v) in layer.pages.items():
                        end=min(page,layer.logical_length-lp*page)
                        check(np.array_equal(k.a[:,:,:end],history[:,:,lp*page:lp*page+end]),'K history')
                        check(np.array_equal(v.a[:,:,:end],k.a[:,:,:end]),'V history')
                    if not far:
                        check(len(layer.pages)==(layer.logical_length+page-1)//page,'all pages resident')
                check(Tensor.clones==expected_backups,'clone frequency')
                check(Tensor.copies==page*3*2,'successful append copy count')
                if not far:check(layer.reclaimed_pages==before_reclaimed,'no-op reclaim for far kept')
                results.append(dict(page=page,near=near,far=far,sink=sink,steps=page*3,
                    clones=Tensor.clones,copies=Tensor.copies,page_visits=layer.pages.visits,
                    reclaimed=layer.reclaimed_pages-before_reclaimed))
    # Existing bytewise rollback contract includes unused tail slots. Removing
    # backups changes it even though no previously committed token is overwritten.
    for failure in (1,2):
        g=Geometry(near=8,window=4,page=8)
        layer=Layer(g);x=Tensor(np.ones((1,2,13,4),dtype=np.float32))
        layer.update(x,x);layer.freeze()
        old={lp:(k.a.copy(),v.a.copy()) for lp,(_,k,v) in layer.pages.items()}
        Tensor.copies=0;Tensor.fail_after_copy=failure
        try:layer.update(Tensor(np.full((1,2,1,4),55,dtype=np.float32)),x[:,:,:1])
        except RuntimeError:pass
        else:raise AssertionError('failure not raised')
        check(layer.logical_length==13,'rollback length')
        for lp,(_,k,v) in layer.pages.items():
            check(np.array_equal(k.a,old[lp][0]) and np.array_equal(v.a,old[lp][1]),'full-page rollback')
        layer.update(x[:,:,:1],x[:,:,:1]);check(layer.logical_length==14,'retry')
    return results

if __name__=='__main__':
    cases=run_paged()
    print(json.dumps(dict(scope='actual-source host facades, not native Torch/CUDA',paged_cases=len(cases),decode_steps=sum(x['steps'] for x in cases),rollback_cases=2,paged=cases),indent=2))
