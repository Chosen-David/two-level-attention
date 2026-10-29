"""Native G=1 dtype preservation, eager/capture. Run from repository root."""
import ast,inspect,json
from types import SimpleNamespace as NS
import torch
from sparse_attn.indexer import tli_indexer as mod
if not torch.cuda.is_available():raise RuntimeError('CUDA required')
tree=ast.parse(inspect.getsource(mod));guards=[]
for n in ast.walk(tree):
 if isinstance(n,ast.If) and '_HAS_TRITON' in ast.unparse(n.test) and any(isinstance(x,ast.Expr) and '_ako_gqa_mean_kernel[' in ast.unparse(x) for x in n.body):guards.append(n.test)
if len(guards)!=2:raise RuntimeError('caller count changed')
results=[]
for dtype in (torch.float32,torch.float16,torch.bfloat16):
 x=torch.tensor([[.2501,.2502,.4997]],device='cuda',dtype=dtype).contiguous()
 for guard in guards:
  ns={'torch':torch,'self':NS(per_q_head=False),'_HAS_TRITON':True,'_G_':1,'group_size':1,'_sm':x.view(1,1,1,3),'sm':x}
  use=eval(compile(ast.Expression(guard),'guard','eval'),ns)
  def run():
   if use:
    out=torch.empty_like(x)
    mod._ako_gqa_mean_kernel[(1,1)](x,out,3,0,0,G=1,BLOCK=1024,num_warps=4)
    return out
   return x.nan_to_num(nan=0).view(1,1,3).mean(1)
  expected=x.clone();actual=run();torch.testing.assert_close(actual,expected,rtol=0,atol=0)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):run()
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=run()
  graph.replay();torch.cuda.synchronize();torch.testing.assert_close(out,expected,rtol=0,atol=0)
  results.append({'dtype':str(dtype),'fused':use,'exact':True,'capture':True})
print(json.dumps(results,indent=2))
