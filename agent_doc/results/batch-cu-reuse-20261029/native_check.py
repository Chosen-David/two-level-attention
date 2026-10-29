"""Run in the repo's CUDA environment. Exact-output paired wrapper check/timing."""
import ast, importlib, json, sys
from pathlib import Path
import torch
m=importlib.import_module('sparse_attn.ops.eager_decoding')
name='_eager_decoding_attn_gather_batch'
def load(path):
 f=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 ns=vars(m).copy();exec(compile(ast.Module(body=[f],type_ignores=[]),path,'exec'),ns);return ns[name]
base=load(sys.argv[1]);new=getattr(m,name)
if not torch.cuda.is_available():raise RuntimeError('CUDA required')
torch.manual_seed(32)
results=[]
for b in (2,10):
 for t in (256,16384):
  for rows in (2,8):
   q=torch.randn(b,1,8,32,device='cuda',dtype=torch.bfloat16)
   k=torch.randn(b,t,2,32,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
   mask=torch.zeros(b,1,rows,t,device='cuda',dtype=torch.bool);mask[...,:128]=True
   args=(q,k,v,mask,1,None,32**-0.5,[t-i for i in range(b)])
   a=base(*args);z=new(*args)
   torch.testing.assert_close(z,a,rtol=0,atol=0)
   times=[[],[]]
   for _ in range(3):base(*args);new(*args)
   for i in range(20):
    for version in ((0,1) if i%2==0 else (1,0)):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
     start.record();(base,new)[version](*args);end.record();end.synchronize();times[version].append(start.elapsed_time(end))
   results.append({'B':b,'T':t,'R':rows,'exact_output':True,'baseline_ms':times[0],'candidate_ms':times[1]})
print(json.dumps({'scope':'CUDA eager wrapper only; Graph/end-to-end acceptance separate','results':results},indent=2))
