"""Host AST regression; does not validate Triton lowering or GPU performance.
Run: python tests/test_nfa_union_wide_aux_host.py [selector-source]
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
class A(np.ndarray):
    def to(self,dtype):return np.asarray(self,dtype=dtype).view(A)
def ar(x):return np.asarray(x).view(A)
class P:
    def __init__(self,a,off=0):self.a=a;self.off=off
    def __add__(self,i):return P(self.a,self.off+i)
def load(p,mask=True,other=0):
    ix,mask=np.broadcast_arrays(p.off,mask);out=np.full(ix.shape,other,p.a.dtype)
    if np.any(ix[mask]<0) or np.any(ix[mask]>=len(p.a)):raise IndexError('pointer bounds')
    out[mask]=p.a[ix[mask]];return ar(out)
def store(p,v,mask=True):
    ix,mask=np.broadcast_arrays(p.off,mask)
    if np.any(ix[mask]<0) or np.any(ix[mask]>=len(p.a)):raise IndexError('store bounds')
    p.a[ix[mask]]=np.broadcast_to(v,ix.shape)[mask]
def gather(x,ix,axis=0):
    if np.any(ix<0) or np.any(ix>=len(x)):raise IndexError('gather index outside run')
    return ar(np.take(x,ix,axis=axis))
def build(path):
    tree=ast.parse(Path(path).read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_final_union_physical_kernel')
    f.decorator_list=[]
    for a in f.args.args:a.annotation=None
    tl=NS(program_id=lambda i:0,load=load,store=store,arange=lambda a,b:ar(np.arange(a,b)),minimum=lambda a,b:ar(np.minimum(a,b)),maximum=lambda a,b:ar(np.maximum(a,b)),where=lambda *x:ar(np.where(*x)),gather=gather,sort=lambda x,descending=False:ar(np.sort(x)),static_range=range,cumsum=lambda x,axis:ar(np.cumsum(x,axis=axis)),sum=lambda x,axis:ar(np.sum(x,axis=axis)),int32=np.int32)
    env={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(path),'exec'),env);return env[f.name]
def pow2(n):return 1<<(max(1,n)-1).bit_length()
def native_build(path):
    import importlib.util, tempfile
    import torch, triton
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no silent host fallback')
    tree=ast.parse(Path(path).read_text())
    f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_final_union_physical_kernel')
    tmp=tempfile.TemporaryDirectory();file=Path(tmp.name)/'union_kernel.py'
    file.write_text('import triton\nimport triton.language as tl\n'+ast.unparse(f)+'\n')
    spec=importlib.util.spec_from_file_location('union_kernel',file);mod=importlib.util.module_from_spec(spec);sys.modules['union_kernel']=mod;spec.loader.exec_module(mod)
    kernel=mod._final_union_physical_kernel
    def invoke(*ptrs,**kw):
        tensors=[torch.from_numpy(p.a.copy()).cuda() for p in ptrs]
        kernel[(1,)](*tensors,**kw,num_warps=8)
        torch.cuda.synchronize()
        eager=[x.cpu().numpy().copy() for x in tensors[-2:]]
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):kernel[(1,)](*tensors,**kw,num_warps=8)
        tensors[-2].fill_(-9);tensors[-1].zero_();graph.replay();torch.cuda.synchronize()
        for p,t,e in zip(ptrs[-2:],tensors[-2:],eager):
            got=t.cpu().numpy();np.testing.assert_array_equal(got,e);p.a[:]=got
    invoke._tmp=tmp
    return invoke

def run(path):
    fn=native_build(path) if '--cuda' in sys.argv else build(path);rng=np.random.default_rng(731);cases=0
    for swa in (0,128,896,1024):
      for seedcap in (0,128,892,1024,2048):
       for seeded in (0,1):
        for active in (0,-1):
         cap=16384;length=8192;w=1024;sink=4;prot=sink+swa;n=w+prot+seedcap;npad=pow2(n);apad=pow2(prot+seedcap)
         mid=np.sort(rng.choice(np.arange(4,length-swa),size=w,replace=False)).astype(np.int32)
         seeds=rng.integers(4,length,size=seedcap,dtype=np.int32)
         count=seedcap;table=rng.permutation(cap//64).astype(np.int32)
         out=np.full(n,-9,np.int32);vis=np.zeros(1,np.int32)
         args=[P(x) for x in (mid,np.array([w],np.int32),np.array([length],np.int32),np.array([active],np.int32),np.array([0],np.int32),np.array([seeded],np.int32),seeds,np.array([count],np.int32),table,out,vis)]
         fn(*args,W=w,PROT=prot,MAX_SEED=seedcap,N=n,N_PAD=npad,A_PAD=apad,MERGE_BITS=npad.bit_length()-1,SINK=sink,SWA=swa,CAP=cap,MAX_BLOCKS=cap//64,PAGE=64,MAX_REQ=1,MAX_VIS=n,INVALID=-1,HKV=1,B=1,LAYER_OFFSET=0)
         want=sorted(set(mid)|set(range(sink))|set(range(length-swa,length))|(set(seeds) if seeded==0 and active>=0 else set()))
         physical=[int(table[x//64])*64+x%64 for x in want]
         np.testing.assert_array_equal(out[:len(want)],physical)
         np.testing.assert_array_equal(out[len(want):],-1)
         if int(vis[0])!=len(want):raise AssertionError('count')
         cases+=1
    print(f'{cases} cases pass; '+('native eager and CUDA Graph' if '--cuda' in sys.argv else 'actual AST host integer logic only'))
if __name__=='__main__':
    path=sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'sparse_attn/indexer/nfa_layer_batch_selector.py'
    run(path)
