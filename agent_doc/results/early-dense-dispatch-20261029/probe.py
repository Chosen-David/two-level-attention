"""Execute dispatch prefixes only; tensor metadata doubles, no Torch/GPU."""
import ast
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

class GatherReached(Exception): pass
class Shape:
    def __init__(self,shape,label): self.shape=tuple(shape);self.label=label;self.device='device'
    def squeeze(self,d): return Shape(self.shape[:d]+self.shape[d+1:],self.label)
    def unsqueeze(self,d): return Shape(self.shape[:d]+(1,)+self.shape[d:],self.label)
    def __getitem__(self,i): return Shape(self.shape[1:],self.label)
    def reshape(self,*shape):
        import math
        total=math.prod(self.shape);known=math.prod(x for x in shape if x!=-1)
        return Shape(tuple(total//known if x==-1 else x for x in shape),self.label)
    def __floordiv__(self,x): return self
    def sum(self,**kw): raise GatherReached()

def run(path,length,block,h,g,per,valid,hetero):
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_eager_decoding_attn_gather')
    ops=dict(repeat=0,arange=0,validate=0)
    def repeat(m,pattern,**kw):
        ops['repeat']+=1
        return (Shape((m.shape[0]//kw['g'],kw['g'],m.shape[1]*kw['bs']),m.label)
                if 'g' in kw else Shape((m.shape[0],m.shape[1]*kw['bs']),m.label))
    def arange(n,**kw): ops['arange']+=1;return Shape((n,),'heads')
    def validate(*args): ops['validate']+=1
    def dense(*args,**kw):
        return [((x.shape,x.label) if isinstance(x,Shape) else x) for x in args],kw
    ns=dict(torch=SimpleNamespace(Tensor=Shape,arange=arange),repeat=repeat,
            _validate_gqa_shape=validate,_eager_decoding_attn_dense=dense,_HAS_TRITON=False)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    args=[Shape((1,1,h*g,128),'q'),Shape((1,length,h,128),'k'),
          Shape((1,length,h,64 if hetero else 128),'v'),
          Shape((1,1,h*g if per else h,length//block),'mask'),block,Shape((2,),'cu')]
    try: result=ns[fn.name](*args,softmax_scale=0.125,valid_length=valid,allow_kv_hetero=hetero)
    except GatherReached: result='gather'
    return result,ops

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--baseline',type=Path,default=root/'baseline.py')
parser.add_argument('--candidate',type=Path,default=root/'candidate.py')
options=parser.parse_args()
cases=short=long=0
for length in (4096,8192,16384,32768):
 for block in (1,4,64):
  for h,g in ((1,1),(8,4)):
   for per in (False,True):
    for valid in (None,length-7):
     for hetero in (False,True):
      args=(length,block,h,g,per,valid,hetero)
      before,old=run(options.baseline,*args);after,new=run(options.candidate,*args)
      if before!=after or old['validate']!=1 or new['validate']!=1:raise RuntimeError('dispatch contract changed')
      if length<16384:
       if old['repeat']!=1 or new['repeat']!=0 or new['arange']!=0:raise RuntimeError('dead work remains')
       short+=1
      else:
       if old!=new or after!='gather':raise RuntimeError('long route changed')
       long+=1
      cases+=1
print(json.dumps(dict(cases=cases,short_dense=short,long_gather=long,
                     identical_dense_arguments=True,torch_tested=False,gpu_tested=False),indent=2))
