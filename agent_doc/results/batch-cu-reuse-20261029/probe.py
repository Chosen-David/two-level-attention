"""Execute the actual wrapper AST with metadata doubles; not native Torch/CUDA."""
import ast, json, sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
paths=[Path(x) for x in sys.argv[1:]] or [root/'baseline.py',root/'candidate.py']
name='_eager_decoding_attn_gather_batch'
class Tensor:
 def __init__(self,x): self.x=np.asarray(x); self.shape=self.x.shape; self.device='test-device'
 def __getitem__(self,k): return Tensor(self.x[k])
 def numel(self): return self.x.size
class Torch:
 int32='int32'
 def __init__(self):self.created=[]
 def is_tensor(self,x):return isinstance(x,Tensor)
 def tensor(self,x,**kw):
  if kw!={'device':'test-device','dtype':'int32'}:raise RuntimeError('metadata changed')
  t=Tensor(x);self.created.append(t);return t
 def cat(self,x,dim):return np.concatenate(x,axis=dim)
def build(path):
 tree=ast.parse(path.read_text()); f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
 mod=ast.Module(body=[f],type_ignores=[]);return compile(mod,str(path),'exec'),tree
compiled=[build(p) for p in paths]
# All other production AST nodes must stay unchanged.
other=lambda tr:ast.dump(ast.Module(body=[n for n in tr.body if not(isinstance(n,ast.FunctionDef) and n.name==name)],type_ignores=[]))
if other(compiled[0][1])!=other(compiled[1][1]):raise RuntimeError('unrelated change')
cases=0
for b in (0,1,2,10,32):
 for t in (1,8192,16384,32768):
  for kind in ('none','scalar','list','tuple','tensor'):
   lengths=[max(0,t-i) for i in range(b)]
   vl={'none':None,'scalar':t,'list':lengths,'tuple':tuple(lengths),'tensor':Tensor(lengths)}[kind]
   observations=[]
   for version,(code,_) in enumerate(compiled):
    torch=Torch();seen=[]
    def gather(q,k,v,mask,block,cu,scale,**kw):
     row=int(q.x[0,0]); length=kw['valid_length'];length=length.x.tolist() if isinstance(length,Tensor) else length
     if cu.x.tolist()!=[0,t] or kw['allow_kv_hetero'] is not False:raise RuntimeError('bounds/contract changed')
     seen.append((row,length,id(cu)))
     return np.array([[row,t,scale]])
    ns={'torch':torch,'_eager_decoding_attn_gather':gather};exec(code,ns)
    args=(Tensor(np.arange(b).reshape(b,1)),Tensor(np.empty((b,t,0))),Tensor(np.empty((b,t,0))),Tensor(np.empty((b,1,0))),1,None,0.125,vl)
    try:out=ns[name](*args);error=None
    except ValueError as exc:out=None;error=type(exc).__name__
    expected=(b if version==0 else int(b>0))
    if len(torch.created)!=expected:raise RuntimeError('allocation count')
    observations.append((None if out is None else out.tolist(),error,[(r,l) for r,l,_ in seen]))
    if b:
     identities={id(x) for x in torch.created}
     ns[name](*args)
     if identities & {id(x) for x in torch.created[expected:]}:raise RuntimeError('cross-call reuse')
   if observations[0]!=observations[1]:raise RuntimeError('behavior mismatch')
   cases+=1
print(json.dumps({'paired_cases':cases,'allocation_calls_before':'B','allocation_calls_after':'1 if B>0 else 0','cross_call_isolation':True,'native_torch':False,'gpu':False},indent=2))
