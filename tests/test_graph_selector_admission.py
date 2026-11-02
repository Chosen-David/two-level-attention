"""Execute the production coordinator with a selector stub, not model inference."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

PATH = Path(os.environ.get('NFA_ADMISSION_SOURCE', Path(__file__).resolve().parents[1] / 'sparse_attn/cache/paged_region_cache.py'))
class T(np.ndarray):
    def __new__(cls, x): return np.asarray(x).view(cls)
    @property
    def device(self): return 'cpu'

class Host:
    long=np.int64
    int32=np.int32
    metadata_calls=0
    def tensor(self,x,**kw):
        self.metadata_calls+=1
        return T(np.array(x,dtype=kw.get('dtype')))
    def arange(self,n,**kw):
        self.metadata_calls+=1
        return T(np.arange(n,dtype=kw.get('dtype')))
    def _assert_async(self,x):
        # PyTorch TensorCompare.cu explicitly requires exactly one element.
        if np.size(x)!=1: raise RuntimeError('Boolean value of Tensor with more than one value is ambiguous')
        if not bool(x): raise RuntimeError('assert failed')

def coordinator(backend, calls):
    tree=ast.parse(PATH.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_selected_ids_by_layer')
    fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    def select(cache,li,rid,indexer,q,scale,length):
        calls.append((li,rid,length,tuple(q.shape)))
        return ([rid,length],1)
    env={'torch':backend,'select_graph_ready_ids':select,'_protected_selection_batch':lambda *a:([0],1)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(PATH),'exec'),env)
    return env[fn.name]

def fixture(backend,device,batch,invalid=False,both=False):
    capacity=70;layers=2;heads=4
    ids=list(range(1,2*batch+1,2))[::-1]
    states={r:NS(logical_length=11+r,first_decode_seeded=True,
                 skip_near_by_layer=[both]*layers,skip_far_by_layer=[both]*layers) for r in ids}
    pending={r:19+r for r in ids}
    if invalid: pending[ids[-1]]=0
    q=T(np.zeros((layers,capacity,heads,8))) if isinstance(backend,Host) else backend.zeros((layers,capacity,heads,8),device=device)
    cache=NS(states=states,pending_lengths=pending,_last_q_buf=q,num_query_heads=heads,
             _store_gather_slots_batch=lambda *a:None)
    model=NS(model=NS(layers=[NS(self_attn=NS(indexer=NS(),scaling=1.0)) for _ in range(layers)]))
    return cache,model,[-1]+ids[:batch//2]+[-1]+ids[batch//2:]+[68,-1],ids

class AdmissionTests(unittest.TestCase):
    def verify(self,backend,device):
        for b in [1,2,3,8,32]:
            calls=[];fn=coordinator(backend,calls);cache,model,req,ids=fixture(backend,device,b)
            result=fn(cache,model,req)
            self.assertEqual(calls,[(li,r,19+r,(1,4,1,8)) for li in range(2) for r in ids])
            self.assertEqual(list(result[0]),ids)
        calls=[];fn=coordinator(backend,calls);cache,model,req,ids=fixture(backend,device,2,invalid=True)
        with self.assertRaisesRegex(ValueError,'positive lengths'):fn(cache,model,req)
        self.assertEqual(calls,[])
        cache,model,req,ids=fixture(backend,device,3,both=True)
        self.assertEqual(fn(cache,model,req),{0:'__prepared__',1:'__prepared__'})
        self.assertEqual(calls,[])
        self.assertEqual(fn(cache,model,[-1,68]),{0:{},1:{}})
    def test_host_coordinator(self):
        backend=Host();self.verify(backend,'cpu');self.assertEqual(backend.metadata_calls,0)
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable')
    def test_native_cpu_coordinator(self):
        import torch
        self.verify(torch,'cpu')
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable')
    def test_native_cuda_coordinator(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        self.verify(torch,'cuda');torch.cuda.synchronize()

if __name__=='__main__':unittest.main()
