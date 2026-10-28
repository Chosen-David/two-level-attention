"""Same-input indexer benchmark; timing excludes KV writes and attention.

--reference serial-eager compares the actual old clear/prepare_mask loop.
--reference serial-parallel isolates batching from algorithm/cache changes.
--capture is supported only for the latter; it is not an old-Graph comparison.
CPU timing/profile records are diagnostics, never CUDA speedup evidence.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.indexer.tli_batch import prepare_mask_batched
from sparse_attn.metrics import Metrics


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--length',type=int,default=4096)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--heads',type=int,default=32)
    p.add_argument('--kv-heads',type=int,default=8)
    p.add_argument('--score',choices=['torch','triton'],default='torch')
    p.add_argument('--reference',choices=['serial-eager','serial-parallel'],default='serial-eager')
    p.add_argument('--capture',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.steps<1 or a.repeats<1 or a.batch<1 or a.length<512:
        p.error('positive steps/repeats/batch and length>=512 required')
    if (a.capture or a.score=='triton') and not a.device.startswith('cuda'):
        p.error('capture and Triton timing require real CUDA')
    if a.capture and a.reference!='serial-parallel':
        p.error('capture comparison isolates batching: use serial-parallel')
    os.environ['TLI_BATCH_SCORE']=a.score
    torch.set_num_threads(1)
    cfg=dict(tia_block_size=64,tia_level1_topk=64,tia_level2_topk=1024,
             tia_level2_cmp_ratio=4,tia_enable_async_topk=False,
             tli_enable_layer_skip=False,tli_subspace='tail',
             tli_sink_tokens=4,tli_swa_size=128,tli_near_pages=32,
             tli_near_tokens='off',tli_near_len=4096)
    def make():
        idx=TLIIndexer(SimpleNamespace(**cfg)); idx.metrics=Metrics(); return idx
    device=torch.device(a.device); dtype=getattr(torch,a.dtype)
    is_cuda=device.type=='cuda'
    def sync():
        if is_cuda: torch.cuda.synchronize(device)
    result=dict(args=vars(a),config=cfg,torch=torch.__version__,cuda_executed=is_cuda,
                gpu=torch.cuda.get_device_name(device) if is_cuda else None,
                timing_scope='indexer only; KV append, Q generation, attention and length upload excluded',
                baseline=a.reference,capture=a.capture,
                source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                    [Path(__file__),Path('sparse_attn/indexer/tli_batch.py'),Path('sparse_attn/indexer/tli_batch_kernels.py'),Path('sparse_attn/indexer/tli_indexer.py'),Path('sparse_attn/metrics.py')]},
                repeats=[],mask_mismatch_bits=0,attention_max_abs=0.0,profile=None)
    # Original code calls CUDA capture introspection even on CPU. This shim
    # permits its Torch fallback; it does not emulate GPU or graph execution.
    cpu_shim=nullcontext() if is_cuda else patch('torch.cuda.is_current_stream_capturing',return_value=False)
    with torch.no_grad(),cpu_shim:
        for rep in range(a.repeats):
            torch.manual_seed(642+rep)
            S=a.length+a.steps+a.warmup+64
            k=torch.randn(a.batch,a.kv_heads,S,128,device=device,dtype=dtype).transpose(1,2)
            v=torch.randn_like(k)
            q=torch.randn(a.batch,1,a.heads,128,device=device,dtype=dtype)
            lens_host=[a.length-b*17 for b in range(a.batch)]
            if min(lens_host)<1: raise ValueError('batch too large for length')
            lengths=torch.tensor(lens_host,device=device)
            rows=torch.arange(a.batch,device=device)
            candidate=make(); refs=[make() for _ in range(a.batch)]
            cu=torch.tensor([0,S],device=device,dtype=torch.int32)
            pos=torch.empty(a.batch,device=device,dtype=torch.long)
            def serial():
                masks=[]
                for b,idx in enumerate(refs):
                    if a.reference=='serial-eager':
                        idx.clear()
                        m,_=idx.prepare_mask(q[b:b+1],pos[b:b+1],k[b:b+1],cu,
                                             softmax_scale=128**-.5,valid_length=lens_host[b])
                    else:
                        m,_=prepare_mask_batched(idx,q[b:b+1],k[b:b+1],lengths[b:b+1],128**-.5)
                    masks.append(m)
                return torch.cat(masks)
            def batch():
                return prepare_mask_batched(candidate,q,k,lengths,128**-.5)[0]
            funcs=[serial,batch]; samples=[[],[]]; events=[[],[]]
            pos.copy_(lengths-1)
            cold=[]
            for fn in funcs:
                sync(); t=time.perf_counter(); fn(); sync(); cold.append((time.perf_counter()-t)*1e3)
            for _ in range(a.warmup):
                serial();batch()
            if a.capture:
                stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    serial();batch()
                torch.cuda.current_stream().wait_stream(stream)
                graphs=[]; outputs=[]
                for fn in funcs:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): out=fn()
                    graphs.append(graph); outputs.append(out)
                funcs=[lambda j=j:(graphs[j].replay(),outputs[j])[1] for j in range(2)]
            for step in range(a.steps):
                k[rows,lengths]=torch.randn(a.batch,a.kv_heads,128,device=device,dtype=dtype)
                lengths.add_(1); lens_host=[x+1 for x in lens_host];pos.copy_(lengths-1);q.normal_()
                out=[None,None]
                for arm in ([0,1] if (rep+step)%2==0 else [1,0]):
                    sync()
                    if is_cuda:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                    t=time.perf_counter(); out[arm]=funcs[arm]()
                    if is_cuda: end.record()
                    sync();samples[arm].append((time.perf_counter()-t)*1e3)
                    if is_cuda:events[arm].append(start.elapsed_time(end))
                result['mask_mismatch_bits']+=int((out[0]!=out[1]).sum())
                def attend(m):
                    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=m[:,0].repeat_interleave(a.heads//a.kv_heads,dim=1)[:,:,None],enable_gqa=True)
                error=(attend(out[0])-attend(out[1])).abs().max().item()
                result['attention_max_abs']=max(result['attention_max_abs'],error)
            result['repeats'].append(dict(cold_ms=cold,wall_ms=samples,cuda_event_ms=events))
            if a.profile and rep==0:
                activities=[torch.profiler.ProfilerActivity.CPU]
                if is_cuda: activities.append(torch.profiler.ProfilerActivity.CUDA)
                profiles=[]
                for fn in funcs:
                    with torch.profiler.profile(activities=activities) as prof: fn();sync()
                    profiles.append([dict(name=e.key,count=e.count,cpu_us=e.cpu_time_total,device_us=e.device_time_total) for e in prof.key_averages()])
                result['profile']=profiles
        result['median_wall_ms']=[statistics.median([x for r in result['repeats'] for x in r['wall_ms'][j]]) for j in range(2)]
        result['gpu_acceptance']='pending' # microbench alone never validates full-model quality
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['median_wall_ms','mask_mismatch_bits','attention_max_abs','cuda_executed']}))

if __name__=='__main__': main()
