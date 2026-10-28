"""Regression guards for the iter3 merge; CPU tests do not validate CUDA speed."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.indexer import tli_indexer as mod

@pytest.mark.parametrize('r',[17,1024,2048,4224])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_invalid_softmax_kernel_is_not_dispatched(r,dtype):
    scores=torch.randn(4,r,dtype=dtype)
    keep=torch.ones(1,r,dtype=torch.bool)
    assert mod._iter3_fused_softmax(scores,keep,keep,r,4,1) is None


def bypass_function(source):
    tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_mask_graph')
    node=next(n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='_l2_bypass')
    # Execute the actual full bypass body, including its metrics conditional.
    f=ast.parse('def run(cand_idx2, keep2, S_cap, Hkv, R_cap, dev, mid_mask, sink_tok, swa_force, ar_tok, S_real, q, self, valid_len):\n pass').body[0]
    f.body=node.body
    ns={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'bypass','exec'),ns)
    return ns['run']

@pytest.mark.parametrize('capacity',[65,129])
@pytest.mark.parametrize('bad_id',[9,999])
def test_bypass_filters_inactive_candidates(capacity,bad_id):
    f=bypass_function(Path(mod.__file__).read_text())
    ids=torch.tensor([[4,5,bad_id]])
    keep=torch.tensor([[True,True,False]])
    mid=torch.ones(capacity,dtype=torch.bool);mid[:2]=False;mid[-4:]=False
    swa=torch.zeros_like(mid);swa[-4:]=True
    metrics=SimpleNamespace(add_select_result=lambda *args:None)
    got=f(ids,keep,capacity,1,3,'cpu',mid,2,swa,torch.arange(capacity),torch.tensor(capacity),torch.empty(1),SimpleNamespace(metrics=metrics),capacity)
    expected=swa.clone();expected[:2]=True;expected[4:6]=True
    assert torch.equal(got.flatten(),expected)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('r',[17,2049,4224])
def test_compact_cuda_and_graph(r):
    pytest.importorskip('triton')
    device=torch.device('cuda');s=65;h=8
    ids=torch.randint(s,(h,r),device=device)
    keep=torch.rand(h,r,device=device)>.4
    ids=torch.where(keep,ids,s+7)
    mid=torch.arange(s,device=device)%3==0
    safe=torch.where(keep,ids,s-1)
    order=torch.argsort(~keep,dim=-1,stable=True)
    ref_ids=safe.gather(-1,order)
    expected=(ref_ids,keep.gather(-1,order),mid[ref_ids],keep.sum(-1))
    for _ in range(3):
        actual=mod._iter3_compact(ids,keep,mid,r,h,s,device)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual=mod._iter3_compact(ids,keep,mid,r,h,s,device)
    graph.replay()
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
