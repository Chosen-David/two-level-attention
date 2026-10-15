"""CPU control-flow probe of a separately fetched, pinned upstream wrapper.
No FlashAttention/MoBA kernel is imported or executed. Not numerical validation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Tuple, Optional
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
args=parser.parse_args()
source=args.source.read_text()
if hashlib.sha256(source.encode()).hexdigest() != "fcbdb3da5edb69fa4a5960cc766c754dca296c592f1c5764f1578e2c1d0374ca":
    raise ValueError("source differs from pinned MoBA b5d5836 wrapper")
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
routes=[]
def dense(q,k,v,*args,**kw):
    routes.append('native_dense_decode')
    return torch.zeros_like(q)
def sparse(**kw):
    routes.append('sparse_prefill')
    return torch.zeros_like(kw['q'])
ns=dict(torch=torch,Callable=Callable,Tuple=Tuple,Optional=Optional,MoBAConfig=object,flash_attn_func=dense)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(args.source),'exec'),ns)
for qlen,klen in [(8,8),(1,9)]:
    q=torch.randn(1,4,qlen,8);k=torch.randn(1,2,klen,8);v=torch.randn_like(k)
    ns['moba_layer'](sparse,SimpleNamespace(moba_chunk_size=4,moba_topk=1),SimpleNamespace(is_causal=True),q,k,v)
if routes!=['sparse_prefill','native_dense_decode']:raise RuntimeError('upstream phase mismatch')
print(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),routes=routes,
                     cpu_control_flow_only=True,kernel_executed=False)))
