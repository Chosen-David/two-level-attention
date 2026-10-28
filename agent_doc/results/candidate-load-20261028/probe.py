"""Frozen kernel pairs. --bench requires CUDA; interpreter is correctness only."""
import argparse
import json
from pathlib import Path
import torch
import triton
from baseline_kernel import _score_fine_cand_kernel as baseline
from sparse_attn.indexer.tli_indexer import _score_fine_cand_kernel as candidate

p=argparse.ArgumentParser(); p.add_argument('--bench',action='store_true'); args=p.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
if args.bench and device!='cuda': raise RuntimeError('CUDA timing required')
torch.set_num_threads(1)
rows=[]
for dtype in ((torch.float32,torch.float16,torch.bfloat16) if device=='cuda' else (torch.float32,torch.float16)):
  for d in (32,128):
    for g in (1,4):
      for r in (65,129,4224):
        for layout in ('token','head'):
          torch.manual_seed(281028)
          t,h=257,3
          k=torch.randn((t,h,d) if layout=='token' else (h,t,d),device=device,dtype=dtype)
          if layout=='head': k=k.transpose(0,1)
          q=torch.randn(h*g,d,device=device)
          ids=torch.randint(t,(h,r),device=device)
          lengths=torch.tensor([0,r//2,r],device=device)
          # Production redirects inactive slots to a valid sentinel.
          valid=torch.arange(r,device=device)[None]<lengths[:,None]
          ids=torch.where(valid,ids,t-1)
          old=torch.empty(h*g,r,device=device,dtype=dtype); new=torch.empty_like(old)
          def launch(kernel,out):
            kernel[(triton.cdiv(r,64),h)](k,q,ids,lengths,out,r,k.stride(0),k.stride(1),q.stride(0),ids.stride(0),ids.stride(1),out.stride(0),out.stride(1),Hkv=h,G=g,D=d,BT=64,num_warps=4)
          launch(baseline,old); launch(candidate,new)
          torch.testing.assert_close(new,old,rtol=0,atol=0)
          assert torch.isneginf(new[~valid.repeat_interleave(g,0)]).all()
          rec=dict(dtype=str(dtype),d=d,g=g,r=r,layout=layout,exact=True,old_k_lanes=h*r*d,new_k_lanes=int(lengths.sum())*d)
          if args.bench:
            from triton.testing import do_bench,do_bench_cudagraph
            for mode,bench in [('eager',do_bench),('graph',do_bench_cudagraph)]:
              rec[mode+'_ms']=[bench(lambda:launch(baseline,old)),bench(lambda:launch(candidate,new))]
          rows.append(rec)
result=dict(scope='CUDA pairs/timing' if args.bench else 'correctness only; interpreter on CPU is not GPU evidence',device=device,cases=rows)
Path(__file__).with_name('gpu.json' if args.bench else 'pairs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(device=device,pairs=len(rows),exact=True,bench=args.bench)))
