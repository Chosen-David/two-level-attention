"""Actual geometry/masking AST fragments. Default NumPy facade; optional native Torch."""
import ast,json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
paths=[Path(x) for x in sys.argv[1:3]] if len(sys.argv)>2 else [root/'baseline.py',root/'candidate.py']
device=sys.argv[3] if len(sys.argv)>3 else None
if device:
 import torch
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
 make=lambda a:torch.tensor(a,device=device)
 array=lambda a:a.detach().cpu().numpy()
else:
 class T(np.ndarray):
  def view(self,*shape):return self.reshape(*shape)
  def masked_fill(self,mask,value):return np.where(mask,value,self).view(T)
  def masked_fill_(self,mask,value):self[np.broadcast_to(mask,self.shape)]=value;return self
 make=lambda a:np.asarray(a).view(T)
 array=np.asarray
def read(path):
 tree=ast.parse(path.read_text()); funcs={x.name:x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef)}
 geo=[x for x in funcs['_build_graph_geometry'].body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('mid_mask','mid_mask_inv','swa_force') for t in x.targets)]
 body=[]
 for x in funcs['compute_mask_graph'].body:
  if isinstance(x,ast.Expr) and ast.unparse(x).startswith('p.masked_fill_('):body.append(x)
  if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mid_scored' for t in x.targets):body.append(x)
 if len(geo)!=3 or not body:raise RuntimeError('source changed')
 return [compile(ast.Module(body=v,type_ignores=[]),str(path),'exec') for v in (geo,body)]
base,new=map(read,paths);cases=0;negative=False
for cap in (1,129,8192):
 for length in (0,1,cap-1,cap):
  for sink in (0,4,128):
   for swa in (0,1,128,cap+1):
    for rows in (1,8):
     values=np.resize(np.array([.1,.5,np.nan,np.inf,-np.inf],dtype=np.float32),(rows,cap))
     outs=[]
     for geo,body in (base,new):
      ns={'ar_tok':make(np.arange(cap)),'far_tok_lo':sink,'swa_lo_tok':make(np.array(max(0,length-swa))),'S_real':make(np.array(length)),'p':make(values.copy())}
      exec(geo,ns)
      if np.any(array(ns['mid_mask'])&array(ns['swa_force'])):raise RuntimeError('overlap')
      exec(body,ns);outs.append(array(ns['mid_scored']))
     if outs[0].tobytes()!=outs[1].tobytes():raise RuntimeError('mask output differs')
     cases+=1
# Mutation control: if mid did include SWA, the optimization would not be valid.
for geo,body in (base,new):
 ns={'p':make(np.array([[.2,.3]],dtype=np.float32)),'swa_force':make(np.array([False,True])),'mid_mask_inv':make(np.array([False,False]))}
 exec(body,ns)
 if body is base[1]:old=array(ns['mid_scored']).copy()
 else:negative=not np.array_equal(old,array(ns['mid_scored']))
if not negative:raise RuntimeError('negative control missing')
print(json.dumps({'cases':cases,'bitwise_masked_outputs_equal':True,'overlap_negative_detected':negative,'backend':device or 'numpy_facade','gpu':device=='cuda'},indent=2))
