"""CUDA steady mask-maintenance microbenchmark; inject runs outside replay."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batches',type=int,nargs='+',default=[1,8,32])
    p.add_argument('--capacities',type=int,nargs='+',default=[32768,131072])
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required; no CPU timing substitute')
    if min(a.batches)<1 or min(a.capacities)<2 or a.steps<1 or a.repeats<2: p.error('invalid benchmark sizes')
    root=Path(__file__).resolve().parents[2]
    files=['sparse_attn/ops/hybrid_batch_runner.py','exp/bench/hybrid_mask_compare.py']
    result=dict(scope='steady mask update outside model/Graph replay, fixed positions; not TPOT or full inject',gpu=torch.cuda.get_device_name(),torch=torch.__version__,hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},cases=[])
    for b in a.batches:
        for cap in a.capacities:
            lengths=torch.full((b,),cap//2,device='cuda',dtype=torch.long)
            pos=lengths-1
            positions=torch.arange(cap,device='cuda')
            before=torch.zeros(b,1,1,cap,device='cuda',dtype=torch.bfloat16)
            before.masked_fill_(positions[None,None,None,:]>= (lengths-1)[:,None,None,None],torch.finfo(before.dtype).min)
            masks={m:before.clone() for m in ('full','incremental')}
            def run(mode):
                mask=masks[mode]
                if mode=='full': mask.zero_().masked_fill_(positions[None,None,None,:]>=lengths[:,None,None,None],torch.finfo(mask.dtype).min)
                else: mask[:,0,0].scatter_(1,pos[:,None],0)
            run('full');run('incremental')
            torch.testing.assert_close(masks['full'],masks['incremental'],rtol=0,atol=0)
            for _ in range(10):run('full');run('incremental')
            samples={m:[] for m in masks}
            for repeat in range(a.repeats):
                for m in (('full','incremental') if repeat%2==0 else ('incremental','full')):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();wall=time.perf_counter();start.record()
                    for _ in range(a.steps):run(m)
                    end.record();end.synchronize()
                    samples[m].append(dict(device_us=start.elapsed_time(end)*1000/a.steps,wall_us=(time.perf_counter()-wall)*1e6/a.steps))
            result['cases'].append(dict(batch=b,capacity=cap,samples=samples,medians={m:{k:statistics.median(x[k] for x in samples[m]) for k in ('device_us','wall_us')} for m in samples}))
            Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
            print(result['cases'][-1]['medians'],flush=True)
if __name__=='__main__':main()
