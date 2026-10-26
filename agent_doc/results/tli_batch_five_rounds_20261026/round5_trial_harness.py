"""Compare source snapshots of the same batched indexer, not different batching.

Timing covers prepare_mask_batched only. Capture is real CUDA or rejected.
Mask/attention checks, input updates, profiling and cold initialization are
outside steady-state timing. CPU results do not imply GPU latency improvements.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types
from types import SimpleNamespace
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.metrics import Metrics


def load(root, name):
    pkg=types.ModuleType(name); pkg.__path__=[str(root)]; sys.modules[name]=pkg
    spec=importlib.util.spec_from_file_location(name+'.tli_batch',root/'tli_batch.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def storage_bytes(state):
    unique={}
    for x in vars(state).values():
        if isinstance(x,torch.Tensor):
            st=x.untyped_storage(); unique[st.data_ptr()]=st.nbytes()
    return sum(unique.values())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline-root',required=True)
    p.add_argument('--candidate-root',default='sparse_attn/indexer')
    p.add_argument('--device',default='cpu');p.add_argument('--dtype',default='float32',choices=['float32','bfloat16'])
    p.add_argument('--score',default='torch',choices=['torch','triton'])
    p.add_argument('--method',default='mixed',choices=['avg','minmax','mixed'])
    p.add_argument('--batch',type=int,default=4);p.add_argument('--length',type=int,default=4096)
    p.add_argument('--steps',type=int,default=16);p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--capture',action='store_true')
    p.add_argument('--profile',action='store_true');p.add_argument('--out',required=True)
    p.add_argument('--fusions',action='store_true')
    a=p.parse_args()
    if min(a.batch,a.steps,a.repeats)<1 or a.length-a.batch*17<1:p.error('invalid dimensions')
    cuda=a.device.startswith('cuda')
    if (a.capture or a.score=='triton') and not cuda:p.error('CUDA is required for capture/Triton timing')
    os.environ['TLI_BATCH_SCORE']=a.score
    os.environ['TLI_BATCH_FUSIONS']='1' if a.fusions else '0'
    if a.fusions and a.score!='triton':p.error('--fusions requires --score triton')
    torch.set_num_threads(1)
    roots=[Path(a.baseline_root).resolve(),Path(a.candidate_root).resolve()]
    mods=[load(r,'batch_round_'+str(i)) for i,r in enumerate(roots)]
    cfg=dict(tia_block_size=64,tia_level1_topk=64,tia_level2_topk=1024,tia_level2_cmp_ratio=4,
             tia_enable_async_topk=False,tli_enable_layer_skip=False,tli_subspace='tail',
             tli_sink_tokens=4,tli_swa_size=128,tli_near_pages=32,tli_near_tokens='off',tli_near_len=4096,
             tli_far_method='minmax' if a.method=='mixed' else a.method,
             tli_near_method='avg' if a.method=='mixed' else a.method)
    def make():
        idx=TLIIndexer(SimpleNamespace(**cfg));idx.metrics=Metrics();return idx
    def sync():
        if cuda:torch.cuda.synchronize()
    result=dict(args=vars(a),config=cfg,torch=torch.__version__,cuda_executed=cuda,
                gpu=torch.cuda.get_device_name() if cuda else None,
                source_sha256=[{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in r.glob('tli_batch*.py')} for r in roots],
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                shared_indexer_sha256=hashlib.sha256(Path('sparse_attn/indexer/tli_indexer.py').read_bytes()).hexdigest(),
                timing_scope='indexer only',repeats=[],mask_mismatch_bits=0,attention_max_abs=0.,profile=None)
    with torch.no_grad():
        for rep in range(a.repeats):
            torch.manual_seed(719+rep);S=a.length+a.steps+64
            k=torch.randn(a.batch,8,S,128,device=a.device,dtype=getattr(torch,a.dtype)).transpose(1,2)
            q=torch.randn(a.batch,1,32,128,device=a.device,dtype=k.dtype);v=torch.randn_like(k)
            lengths=torch.tensor([a.length-b*17 for b in range(a.batch)],device=a.device)
            row=torch.arange(a.batch,device=a.device);idxs=[make(),make()]
            fns=[lambda j=j:mods[j].prepare_mask_batched(idxs[j],q,k,lengths,128**-.5)[0] for j in range(2)]
            cold=[]
            for fn in fns:
                sync();t=time.perf_counter();fn();sync();cold.append((time.perf_counter()-t)*1e3)
            for _ in range(a.warmup):
                for fn in fns:fn()
            if a.capture:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for fn in fns:fn()
                torch.cuda.current_stream().wait_stream(stream)
                graphs=[];buffers=[]
                for fn in fns:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=fn()
                    graphs.append(graph);buffers.append(out)
                fns=[lambda j=j:(graphs[j].replay(),buffers[j])[1] for j in range(2)]
            samples=[[],[]];events=[[],[]]
            for step in range(a.steps):
                k[row,lengths]=torch.randn(a.batch,8,128,device=a.device,dtype=k.dtype)
                lengths.add_(1);q.normal_();out=[None,None]
                for j in ([0,1] if (rep+step)%2==0 else [1,0]):
                    sync()
                    if cuda:
                        e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record()
                    t=time.perf_counter();out[j]=fns[j]()
                    if cuda:e1.record()
                    sync();samples[j].append((time.perf_counter()-t)*1e3)
                    if cuda:events[j].append(e0.elapsed_time(e1))
                result['mask_mismatch_bits']+=int((out[0]!=out[1]).sum())
                def attend(m):
                    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=m[:,0].repeat_interleave(4,dim=1)[:,:,None],enable_gqa=True)
                result['attention_max_abs']=max(result['attention_max_abs'],(attend(out[0])-attend(out[1])).abs().max().item())
            result['repeats'].append(dict(cold_ms=cold,wall_ms=samples,cuda_event_ms=events,side_cache_storage_bytes=[storage_bytes(i._batch_tli) for i in idxs]))
            if a.profile and rep==0:
                acts=[torch.profiler.ProfilerActivity.CPU]
                if cuda:acts.append(torch.profiler.ProfilerActivity.CUDA)
                result['profile']=[]
                for fn in fns:
                    with torch.profiler.profile(activities=acts) as prof:fn();sync()
                    result['profile'].append([dict(name=e.key,count=e.count,cpu_us=e.cpu_time_total,device_us=e.device_time_total) for e in prof.key_averages()])
        result['median_wall_ms']=[statistics.median([x for r in result['repeats'] for x in r['wall_ms'][j]]) for j in range(2)]
        result['row_outputs_checked']=a.batch*a.steps*a.repeats
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['median_wall_ms','mask_mismatch_bits','attention_max_abs','cuda_executed']}))
    if result['mask_mismatch_bits'] or result['attention_max_abs']:
        raise SystemExit('comparison failed; raw differences saved')
if __name__=='__main__':main()
