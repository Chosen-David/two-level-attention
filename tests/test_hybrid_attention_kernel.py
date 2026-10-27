import os
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops.far_skip import FarSkipLayout
from sparse_attn.ops.hybrid_attention import attend_rows


@pytest.mark.parametrize('batch',[1,3,8])
@pytest.mark.parametrize('dim',[16,128])
def test_kernel_mixed_rows_and_dead_row(batch,dim):
    interpreter=os.environ.get('TRITON_INTERPRET')=='1'
    if not interpreter and not torch.cuda.is_available():
        pytest.skip('CUDA required (or explicit Triton CPU interpreter)')
    pytest.importorskip('triton')
    from sparse_attn.ops.hybrid_attention_triton import RaggedAttentionPlan, _attend
    dev='cpu' if interpreter else 'cuda'
    torch.manual_seed(46)
    q=torch.randn(batch,1,4,dim,device=dev)
    rows=[]; selections=[]
    for b in range(batch):
        size=17 if b%2 else 63
        row=SimpleNamespace(keys=torch.randn(1,2,size,dim,device=dev),
                            values=torch.randn(1,2,size,dim,device=dev))
        if b%2: row.layout=FarSkipLayout(2,15)
        ids=torch.randint(0,63,(2,19),device=dev)
        valid=torch.rand(2,19,device=dev)>.2
        rows.append(row);selections.append((ids,valid))
    alive=torch.ones(batch,device=dev,dtype=torch.bool)
    if batch>1: alive[-1]=False
    plan=RaggedAttentionPlan(rows)
    ids=torch.stack([i for i,v in selections]);valid=torch.stack([v for i,v in selections])
    out=torch.empty_like(q)
    _attend[(batch*4,)](q,plan.meta,ids,valid,alive,out,q.stride(0),q.stride(2),
        2,2,dim,19,dim**-.5,32,num_warps=4,enable_fp_fusion=False)
    expected=attend_rows(q,rows,selections,dim**-.5)
    expected[~alive]=0
    torch.testing.assert_close(out,expected,rtol=3e-5,atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_wrapper_capture_replay_and_pointer_guard():
    from sparse_attn.ops.hybrid_attention_triton import RaggedAttentionPlan
    row=SimpleNamespace(keys=torch.randn(1,1,17,128,device='cuda'),
                        values=torch.randn(1,1,17,128,device='cuda'),layout=FarSkipLayout(2,15))
    q=torch.randn(1,1,2,128,device='cuda')
    ids=torch.tensor([[0,1,16,17,18]],device='cuda')
    valid=torch.ones_like(ids,dtype=torch.bool);alive=torch.ones(1,device='cuda',dtype=torch.bool)
    plan=RaggedAttentionPlan([row])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): plan(q,[row],[(ids,valid)],.1,alive)
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=plan(q,[row],[(ids,valid)],.1,alive)
    for step in range(38):
        ids[:,2:].add_(1)
        g.replay()
        torch.testing.assert_close(out,attend_rows(q,[row],[(ids,valid)],.1),rtol=3e-5,atol=3e-6)
    row.keys=row.keys.clone()
    with pytest.raises(ValueError,match='storage changed'):
        plan(q,[row],[(ids,valid)],.1,alive)
