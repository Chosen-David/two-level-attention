"""Actual caller AST routing and IEEE rounding evidence; no native CUDA."""
import ast,json,struct,sys
from pathlib import Path
from types import SimpleNamespace as NS
root=Path(__file__).resolve().parent
paths=list(map(Path,sys.argv[1:])) or [root/'baseline.py',root/'candidate.py']
def conditions(path):
 tree=ast.parse(path.read_text());out=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.If) and any('_ako_gqa_mean_kernel[' in ast.unparse(x) for x in n.body) and '_HAS_TRITON' in ast.unparse(n.test):
   # Choose direct launch blocks, not outer method/shape guards.
   if any(isinstance(x,ast.Expr) and '_ako_gqa_mean_kernel[' in ast.unparse(x) for x in n.body):out.append(n.test)
 if len(out)!=2:raise RuntimeError('caller count changed')
 return out
old,new=map(conditions,paths)
class Tensor:
 shape=(1,1,12,32)
 def __init__(self,dtype,contiguous):self.dtype=dtype;self.contiguous=contiguous
 def is_contiguous(self):return self.contiguous
count=changes=0
for dtype in ('bf16','fp16','fp32'):
 for g in (1,2,3,4):
  for tri in (False,True):
   for contig in (False,True):
    for per in (False,True):
     ns={'self':NS(per_q_head=per),'torch':NS(bfloat16='bf16'),'_HAS_TRITON':tri,'_G_':g,'group_size':g,'_sm':Tensor(dtype,contig),'sm':Tensor(dtype,contig)}
     for a,b in zip(old,new):
      evaluate=lambda x:eval(compile(ast.Expression(x),'guard','eval'),ns)
      before,after=evaluate(a),evaluate(b)
      if after != (before and dtype=='bf16'):raise RuntimeError('wrong routing')
      count+=1;changes+=before!=after
# Existing kernel casts acc/G to bf16 irrespective of OUT element type.
text=paths[0].read_text()
if 'val = (acc / G).to(tl.bfloat16)' not in text:raise RuntimeError('kernel changed')
def bf16(x):
 bits=struct.unpack('I',struct.pack('f',x))[0]
 bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
 return struct.unpack('f',struct.pack('I',bits))[0]
counter={}
for dtype,x,y in [('fp32',.2501,.2502),('fp16',.25,.25048828125)]:
 if not(x<y and bf16(x)==bf16(y)):raise RuntimeError('no tie counterexample')
 counter[dtype]={'before':[x,y],'after_bf16':[bf16(x),bf16(y)]}
print(json.dumps({'caller_cases':count,'non_bf16_routes_corrected':changes,'rounding_tie_counterexamples':counter,'native_torch':False,'gpu':False},indent=2))
