"""Execute actual scatter AST fragments; NumPy by default, optional CUDA/Graph."""
import ast, json, sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
paths=[Path(x) for x in sys.argv[1:3]] if len(sys.argv)>2 else [root/'baseline.py',root/'candidate.py']
native=len(sys.argv)>3 and sys.argv[3]=='cuda'
if native:
 import torch
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 make=lambda x: torch.tensor(x,device='cuda')
 asarray=lambda x:x.detach().cpu().numpy()
else:
 class Tensor(np.ndarray):
  def scatter_(self,dim,idx,src):
   if dim!=1: raise RuntimeError('unsupported axis')
   for h in range(self.shape[0]):
    for r in range(idx.shape[1]):self[h,int(idx[h,r])]=src[h,r]
   return self
 class Torch:
  where=staticmethod(lambda c,a,b:np.where(c,a,b).view(Tensor))
  full_like=staticmethod(lambda x,v:np.full_like(x,v).view(Tensor))
  full=staticmethod(lambda shape,v,dtype,device:np.full(shape,v,dtype=dtype).view(Tensor))
 torch=Torch();make=lambda x:np.asarray(x).view(Tensor);asarray=np.asarray

def fragment(path):
 tree=ast.parse(path.read_text())
 fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='compute_mask_graph')
 body=[];active=False
 for x in fn.body:
  if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cand_h' for t in x.targets):active=True;continue
  if active:
   body.append(x)
   if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='score_full' for t in x.targets):break
 if not body or 'scatter_' not in ast.unparse(ast.Module(body=body,type_ignores=[])):raise RuntimeError('source drift')
 return compile(ast.Module(body=body,type_ignores=[]),str(path),'exec')
codes=list(map(fragment,paths));count=0;negative=False
for h in (1,4,32):
 for cap in (1,17,129):
  for width in (1,8,132):
   for mode in ('none','all','mixed'):
    ids=np.broadcast_to(np.arange(width)%cap,(h,width)).copy()
    valid=np.ones((h,width),dtype=bool) if mode=='all' else np.zeros((h,width),dtype=bool) if mode=='none' else ids%2==0
    # Duplicate valid tokens have identical scores, matching producer contract.
    vals=(ids.astype(np.float32)/7).copy();vals[~valid]=np.resize([np.nan,np.inf,-np.inf],np.count_nonzero(~valid))
    outs=[]
    for code in codes:
     ns=dict(torch=torch,keep_h=make(valid),cand_h=make(ids),sfc=make(vals),H=h,S_cap=cap,dev='cuda' if native else 'cpu')
     exec(code,ns)
     out=asarray(ns['score_full']).copy()
     if native:
      g=torch.cuda.CUDAGraph()
      with torch.cuda.graph(g):exec(code,ns)
      g.replay();torch.cuda.synchronize()
      if not np.array_equal(out,asarray(ns['score_full']),equal_nan=True):raise RuntimeError('replay mismatch')
     outs.append(out)
    if outs[0].tobytes()!=outs[1].tobytes():raise RuntimeError('output mismatch')
    count+=1
# Invalid candidates must NOT be redirected to a real token; demonstrate guard necessity.
ns=dict(torch=torch,keep_h=make(np.array([[False]])),cand_h=make(np.array([[0]])),sfc=make(np.array([[np.nan]],np.float32)),H=1,S_cap=1,dev='cuda' if native else 'cpu')
exec(codes[1],ns)
correct=asarray(ns['score_full']).copy()
wrong=np.array([[np.nan]],np.float32)
negative=not np.array_equal(correct,wrong,equal_nan=True)
if not negative:raise RuntimeError('negative control failed')
print(json.dumps({'pairs':count,'bitwise_equal':True,'real_token_sentinel_negative':negative,'backend':'cuda' if native else 'numpy_facade','graph_replay_tested':native},indent=2))
if native and '--bench' in sys.argv:
 import statistics
 bench=[]
 for mode in ('eager','graph'):
  timings=[[],[]]
  for trial in range(6):
   for arm in ((0,1) if trial%2==0 else (1,0)):
    ids=torch.arange(8448,device='cuda').expand(32,-1)
    ns=dict(torch=torch,keep_h=ids%3!=0,cand_h=ids,sfc=(ids.float()/8192).to(torch.bfloat16),H=32,S_cap=131072,dev='cuda')
    def run():exec(codes[arm],ns)
    for _ in range(5):run()
    if mode=='graph':
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):run()
     run=g.replay
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):run()
    end.record();end.synchronize();timings[arm].append(start.elapsed_time(end)/100)
  bench.append({'mode':mode,'baseline_ms':timings[0],'candidate_ms':timings[1],'median_ratio':statistics.median(timings[0])/statistics.median(timings[1])})
 print(json.dumps({'scatter_fragment_only':bench},indent=2))
