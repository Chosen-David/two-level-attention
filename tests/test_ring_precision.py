"""Fixed-input ring postscale error against independently packed logical KV.

A finite-input engineering budget is fixed before measurement, with explicit
rounding terms and conservative exp/reduction/merge allowances. It is not
fitted to observed SDPA differences or a complete forward-error proof. Covers finite seeded inputs; not a general error proof.
"""
import json
import math
import pytest
import torch
from sparse_attn.ops.far_skip import FarSkipLayout
from sparse_attn.ops.partial_attention import partial_attention,merge_partials

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('backend',['simt','tc'])
@pytest.mark.parametrize('scale',[0.,128**-.5,2.])
@pytest.mark.parametrize('splits',[1,4])
def test_postscale_ring_error_budget(dtype,backend,scale,splits):
    torch.manual_seed(0x6A12)
    sink,tail,length,window=3,23,97,7
    q=torch.randn(1,1,8,128,device='cuda',dtype=dtype)
    logical_k=torch.randn(1,length,2,128,device='cuda',dtype=dtype)
    logical_v=torch.randn_like(logical_k)
    positions=torch.cat((torch.arange(sink,device='cuda'),torch.arange(length-tail,length,device='cuda')))
    physical=torch.where(positions<sink,positions,sink+(positions-sink)%tail)
    k=torch.empty(1,sink+tail,2,128,device='cuda',dtype=dtype);v=torch.empty_like(k)
    k[:,physical]=logical_k[:,positions];v[:,physical]=logical_v[:,positions]
    ids=torch.tensor([[[78,80,84,85,86],[77,79,81,83,85]]],device='cuda',dtype=torch.int32)
    counts=torch.tensor([[5,0]],device='cuda',dtype=torch.int32)
    lengths=torch.tensor([length],device='cuda',dtype=torch.int64)
    cfg=dict(sink=sink,swa=window,scale=scale,splits=splits,backend=backend,
             layout=FarSkipLayout(sink,tail),round_scaled_query=False)
    a=partial_attention(q,k,v,(ids,counts),lengths,protected=True,**cfg)
    b=partial_attention(q,k,v,(ids,counts),lengths,protected=False,**cfg)
    actual=merge_partials(a,b,dtype).double()[0,0].reshape(2,4,128)
    max_ratio,max_abs,max_ulp=0.,0.,0.
    eps=torch.finfo(torch.float32).eps;unit=torch.finfo(dtype).eps/2
    eta=torch.finfo(dtype).tiny*torch.finfo(dtype).eps/2
    for h in range(2):
        chosen=list(range(sink))+ids[0,h,:counts[0,h]].tolist()+list(range(length-window,length))
        query=q[0,0,h*4:(h+1)*4].double()
        kk=logical_k[0,chosen,h].double();vv=logical_v[0,chosen,h].double()
        logits=(query@kk.T)*scale;prob=logits.softmax(-1);truth=prob@vv
        rounded=truth.to(dtype)
        weighted_abs=prob@vv.abs()
        gamma_d=128*eps/(1-128*eps)
        delta=gamma_d*(query.abs()@kk.abs().T).amax(-1,keepdim=True)*abs(scale)
        n=len(chosen);gamma_n=n*eps/(1-n*eps)
        budget=(2*delta*torch.exp(2*delta)+8*gamma_n+64*eps)*weighted_abs
        if backend=='tc':
            # Probability cast: normal relative rounding + absolute subnormal
            # allowance. Every partial maximum <= global max, denominator>=1.
            budget=budget+unit*weighted_abs+eta*vv.abs().sum(0)[None]
        # Final cast acts on a perturbed result, not exactly on truth.
        budget=budget+unit*(truth.abs()+budget)+eta
        error=(actual[h]-truth).abs()
        assert torch.all(error<=budget+1e-12),dict(backend=backend,dtype=str(dtype),scale=scale,
            splits=splits,max_ratio=(error/(budget+1e-12)).max().item())
        ulp=(torch.nextafter(rounded,torch.full_like(rounded,math.inf)).double()-rounded.double()).abs()
        max_ratio=max(max_ratio,(error/(budget+1e-12)).max().item())
        max_abs=max(max_abs,error.max().item())
        max_ulp=max(max_ulp,((actual[h]-rounded.double()).abs()/ulp.clamp_min(1e-300)).max().item())
    print(json.dumps(dict(dtype=str(dtype),backend=backend,scale=scale,splits=splits,
                         max_budget_ratio=max_ratio,max_absolute_error=max_abs,max_ulp_from_rounded=max_ulp)))
