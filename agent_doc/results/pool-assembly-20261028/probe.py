"""Actual selector assembly AST; NumPy adapter, not Torch/GPU acceptance."""
import ast,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P=Path(__file__).parent
class Tensor(np.ndarray):
 def expand(self,*shape):return np.broadcast_to(self,tuple(((1,)*(len(shape)-self.ndim)+self.shape)[i] if n==-1 else n for i,n in enumerate(shape))).view(Tensor)
def t(x):return np.asarray(x).view(Tensor)
class WS:
 def __init__(self):self.calls=[]
 def cat(self,name,parts):
  self.calls.append(name)
  return np.concatenate(parts,axis=-1).view(Tensor)
def parts(path):
 tree=ast.parse(path.read_text())
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FarSkipGraphWorkspace')
 sel=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='select')
 start=next(i for i,x in enumerate(sel.body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='left')
 end=next(i for i,x in enumerate(sel.body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='keep')
 tail=next(i for i,x in enumerate(sel.body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='positions')
 def code(nodes):return compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec')
 return code(sel.body[start:end+1]),code(sel.body[end+1].body[:2]),code(sel.body[tail:tail+2])
def run(codes,h,bs,length,sink,window,budget,npages):
 prefix=min(length,sink);swa=max(0,length-window);middle=max(prefix,(length-window-bs*4)//bs*bs)
 first=(middle+bs-1)//bs;stop=max(first,swa//bs)
 pages=t(np.tile(np.arange(first,first+npages),(h,1)))
 ws=WS();obj=SimpleNamespace(ws=ws,ids=lambda n:t(np.arange(n,dtype=np.int64)))
 e=dict(self=obj,cfg=SimpleNamespace(sink_tokens=sink,window_tokens=window,token_budget=budget),
        bs=bs,h=h,middle=middle,first=first,stop=stop,swa=swa,prefix=prefix,length=length,
        chosen_pages=pages,chosen_valid=pages<stop,torch=SimpleNamespace(maximum=np.maximum))
 exec(codes[0],e)
 pruning=0<e['keep']<e['candidates'].shape[-1]
 if pruning:exec(codes[1],e)
 before={k:e[k].copy() for k in ['candidates','valid']+(['pool','pool_valid'] if pruning else [])}
 # Fixed ordered stand-in for top-k output; scoring/selection itself unchanged.
 if pruning or e['keep']==0:
  e['candidates']=e['candidates'][:,:e['keep']]
  e['valid']=e['valid'][:,:e['keep']]
 exec(codes[2],e)
 before.update({k:e[k] for k in ['positions','mask']})
 return before,ws.calls,pruning
baseline=parts(P/'baseline.py');candidate=parts(P/'candidate.py')
cases=pruned=0
for h in [1,8]:
 for bs in [2,64]:
  for length in [1,3,128,1025,4291,32768]:
   for sink,window in [(0,0),(4,0),(0,128),(4,128)]:
    for npages in [0,2]:
     for extra in [0,1,1024]:
      args=(h,bs,length,sink,window,sink+window+extra,npages)
      a,ac,prune=run(baseline,*args);b,bc,_=run(candidate,*args)
      if a.keys()!=b.keys() or any(not np.array_equal(a[k],b[k]) for k in a):
       raise AssertionError(args)
      if len(ac)-len(bc)!=2 or any('protected' in x for x in bc):raise AssertionError('copy count')
      cases+=1;pruned+=int(prune)
print(json.dumps(dict(cases=cases,pruning_cases=pruned,nonpruning_cases=cases-pruned,
 removed_cat_calls_per_selection=2,evidence='selector assembly AST + NumPy; not Torch, GPU, top-k, or latency'),indent=2))

