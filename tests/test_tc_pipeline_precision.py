"""Multi-tile BF16 TC pipeline output vs independent FP64 attention."""
import pytest
import torch
from sparse_attn.ops.partial_attention import partial_attention,merge_partials

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('stages',[1,2,3])
@pytest.mark.parametrize('splits',[1,4])
@pytest.mark.parametrize('round_query',[False,True])
def test_multitile_pipeline_error_budget(stages,splits,round_query):
    torch.manual_seed(32032)
    # Non-power-of-two GQA group exercises padded TC query rows.
    dtype=torch.bfloat16;g,h,d,n=3,2,128,1536
    q=torch.randn(1,1,h*g,d,device='cuda',dtype=dtype)
    k=torch.randn(1,n,h,d,device='cuda',dtype=dtype)
    v=torch.randn_like(k)
    ids=torch.arange(1024,device='cuda',dtype=torch.int32)[None,None].expand(1,h,-1).contiguous()
    counts=torch.zeros(1,h,device='cuda',dtype=torch.int32)
    lengths=torch.tensor([n],device='cuda',dtype=torch.int64)
    scale=d**-.5
    def run():
        kw=dict(sink=4,swa=128,scale=scale,backend='tc',splits=splits,
                pipeline_stages=stages,round_scaled_query=round_query)
        a=partial_attention(q,k,v,(ids,counts),lengths,protected=True,**kw)
        b=partial_attention(q,k,v,(ids,counts),lengths,protected=False,**kw)
        return merge_partials(a,b,dtype)
    run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    eps=torch.finfo(torch.float32).eps;unit=torch.finfo(dtype).eps/2
    eta=torch.finfo(dtype).tiny*torch.finfo(dtype).eps/2
    gamma_d=d*eps/(1-d*eps)
    for count in (0,37,700,1024,0):
        active_counts=[count,max(0,count-9)]
        counts.copy_(torch.tensor([active_counts],device='cuda',dtype=torch.int32))
        graph.replay()
        for head,active in enumerate(active_counts):
            chosen=list(range(4))+list(range(4,active))+list(range(n-128,n))
            query=q[0,0,head*g:(head+1)*g]
            query=(query*scale).double() if round_query else query.double()
            factor=1. if round_query else scale
            kk=k[0,chosen,head].double();vv=v[0,chosen,head].double()
            prob=((query@kk.T)*factor).softmax(-1);truth=prob@vv
            weighted_abs=prob@vv.abs()
            delta=gamma_d*(query.abs()@kk.abs().T).amax(-1,keepdim=True)*abs(factor)
            terms=len(chosen);gamma_n=terms*eps/(1-terms*eps)
            # Same conservative finite-input engineering allowance as ring
            # precision tests. Includes original single-cast TC term; this is
            # an acceptance bound, not proof of residual accuracy improvement.
            pre=(2*delta*torch.exp(2*delta)+8*gamma_n+64*eps+unit)*weighted_abs
            pre=pre+eta*vv.abs().sum(0)[None]
            budget=pre+unit*(truth.abs()+pre)+eta
            error=(actual[0,0,head*g:(head+1)*g].double()-truth).abs()
            assert bool((error<=budget+1e-12).all()),dict(stages=stages,splits=splits,
                round_query=round_query,count=count,max_budget_ratio=(error/(budget+1e-12)).max().item())
