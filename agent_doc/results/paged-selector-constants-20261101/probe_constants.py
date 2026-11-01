"""Execute the actual old constant block and new helper with NumPy storage."""
import ast
import argparse
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

parser=argparse.ArgumentParser()
parser.add_argument('--baseline-source', type=Path, required=True)
parser.add_argument('--candidate-root', type=Path, default=Path(__file__).resolve().parents[3])
args=parser.parse_args()
ROOT=args.candidate_root
spec=importlib.util.spec_from_file_location('lifetime_test',ROOT/'tests/test_paged_selector_constant_lifetime.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
source=args.baseline_source
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TLIIndexer')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_mask_graph')
def assigns(n,name):
    return isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets)
start=next(i for i,n in enumerate(fn.body) if assigns(n,'_cst_key'))
end=next(i for i,n in enumerate(fn.body) if assigns(n,'ar_blk'))
old=compile(ast.Module(body=fn.body[start:end],type_ignores=[]),str(source),'exec')
torch=t.HostTorch();device=t.Device();before=SimpleNamespace();after=SimpleNamespace()
new=t.load_method('_selection_constants',torch)
equal=0
for sparse in (False,True):
    for length in range(131073,131137):
        env=dict(self=before,dev=device,q=SimpleNamespace(dtype=t.np.float32),kt_cap=(length+63)//64,
                 S_cap=length,bs=64,sink_pool_lo=0,sink_tok=4,_sparse_selection=sparse,torch=torch)
        exec(old,env)
        current=new(after,device,t.np.float32,(length+63)//64,length,64,0,4,sparse,retain_for_graph=False)
        for k,a in env['_cst'][1].items():
            b=current[1][k]
            if a is None:assert b is None
            else:t.np.testing.assert_array_equal(a,b)
        equal+=1
del env,current,a,b
gc.collect()
def retained(owner):
    entries=getattr(owner,'_graph_const_entries',{})
    return dict(entries=len(entries),payload_bytes=sum(a.nbytes for e in entries.values() for a in e[1].values() if a is not None))
result=dict(kind='host NumPy, not GPU memory or speed',equal_cases=equal,
            before=retained(before),after=retained(after))
# Only the default dense mode's ar_tok payload: one indexer, 64 distinct lengths.
result['dense_ar_tok_payload_64_lengths_one_layer']=sum(n*8 for n in range(131073,131137))
print(json.dumps(result,indent=2))
