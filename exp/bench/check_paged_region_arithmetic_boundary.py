"""Correctness-only CUDA arithmetic checks across ring's 16K dispatch boundary."""
import argparse,hashlib,json,os,subprocess
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
import torch
from sparse_attn.ops import eager_decoding as eager
from sparse_attn.ops.paged_region_cache import PagedRegionLayer
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.paged_region_kernels import paged_attention,ring_rounds_probabilities

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.manual_seed(81);torch.set_num_threads(4);rows=[]
    destination=Path(a.out);destination.parent.mkdir(parents=True,exist_ok=True)
    cases=[(16383,128,128,'1','2048',True),(16384,128,128,'1','2048',True),
           (16448,128,128,'1','2048',True),(16384,128,128,'0','2048',True),
           (16384,128,128,'1','4096',True),(16384,128,128,'1','2048',False),
           (16384,192,128,'1','2048',True)]
    original=eager._EAGER_TRITON_OK['ok']
    try:
      for length,d,dv,fast,upper,healthy in cases:
       for dtype in [torch.float32,torch.bfloat16]:
        os.environ['TLI_E127_FAST']=fast;os.environ['TLI_E127_W_UPPER']=upper
        eager._EAGER_TRITON_OK['ok']=healthy
        k=torch.randn(1,2,length,d,device='cuda',dtype=dtype);v=torch.randn(1,2,length,dv,device='cuda',dtype=dtype)
        q=torch.randn(1,8,1,d,device='cuda',dtype=dtype)
        layer=PagedRegionLayer(RegionGeometry());layer.update(k,v);layer.freeze(False,False)
        mask=torch.zeros(1,1,2,length,device='cuda',dtype=torch.bool)
        mask[:,:,0,torch.randperm(length,device='cuda')[:1024]]=True
        mask[:,:,1,torch.randperm(length,device='cuda')[:1003]]=True
        cu=torch.tensor([0,length],device='cuda',dtype=torch.int32)
        calls={}
        def tracked(name,fn):
            def call(*args,**kwargs):
                calls[name+':entered']=calls.get(name+':entered',0)+1
                output=fn(*args,**kwargs)
                calls[name+':returned']=calls.get(name+':returned',0)+1
                return output
            return call
        with eager.eager_decoding_request_guard(),ExitStack() as stack:
            for name in ['_eager_gather_triton','_eager_decoding_attn_dense']:
                stack.enter_context(patch.object(eager,name,tracked(name,getattr(eager,name))))
            reference=eager.eager_decoding_attn(q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),mask,1,cu,softmax_scale=d**-.5,allow_kv_hetero=d!=dv)
            # This evaluates the reference's actual post-call fallback flag.
            actual=paged_attention(layer,q,mask,d**-.5,ring_arithmetic=True)
        finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
        tol=.003 if dtype==torch.bfloat16 else 2e-6
        error=float((actual.float()-reference.float()).abs().max())
        close_pass=bool(torch.allclose(actual,reference,atol=tol,rtol=tol))
        tensor_path=destination.parent/f'case_{len(rows):02d}.pt'
        torch.save(dict(actual=actual.cpu(),reference=reference.cpu()),tensor_path)
        actual_route=('triton_gather' if calls.get('_eager_gather_triton:returned',0) else
                      'dense' if calls.get('_eager_decoding_attn_dense:returned',0) else 'torch_gather')
        row=dict(length=length,d=d,dv=dv,dtype=str(dtype),fast=fast,upper=upper,
                 healthy_before=healthy,healthy_after=eager._EAGER_TRITON_OK['ok'],
                 round_prob=ring_rounds_probabilities(length,d,dv,cuda=True),max_error=error,
                 finite=finite,tolerance=tol,close_pass=close_pass,actual_reference_route=actual_route,reference_branch_calls=calls,
                 has_triton=eager._HAS_TRITON,tensor_file=tensor_path.name,tensor_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest())
        rows.append(row);Path(a.out).write_text(json.dumps(dict(status='pending_independent_validation',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),rows=rows),indent=2)+'\n')
        print(row,flush=True)
        torch.testing.assert_close(actual,reference,atol=tol,rtol=tol)
    finally:eager._EAGER_TRITON_OK['ok']=original
if __name__=='__main__':main()
