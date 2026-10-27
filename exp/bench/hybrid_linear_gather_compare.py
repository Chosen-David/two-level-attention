"""CUDA paired linear attention microbenchmark; not model TPOT."""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import torch
from sparse_attn.ops.hybrid_batch_cache import _attend_linear
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',type=int,nargs='+',default=[32768,131072])
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required; no CPU timing substitute')
    if min(a.lengths)<1 or a.steps<1 or a.repeats<2: p.error('positive lengths/steps and repeats>=2 required')
    os.environ['TLI_FAR_SKIP_KERNELS']='triton'
    root=Path(__file__).resolve().parents[2]
    files=['sparse_attn/ops/hybrid_batch_cache.py','sparse_attn/ops/far_skip_accel.py','sparse_attn/ops/far_skip_kernels.py']
    result={'scope':'single linear row mask/topk/gather/SDPA; not full-model TPOT',
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'hashes':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'cases':[]}
    for length in a.lengths:
        q=torch.randn(1,1,32,128,device='cuda',dtype=torch.bfloat16)
        k,v=[torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16).transpose(1,2) for _ in range(2)]
        mask=torch.zeros(1,1,8,length,device='cuda',dtype=torch.bool)
        mask[..., :min(1024,length)]=True
        for execution in ('eager','graph'):
            ws=FarSkipWorkspace(fixed_shapes=True); graphs={}; funcs={}
            def call(mode): return _attend_linear(q,k,v,mask,1156,128**-.5,ws if mode=='fused' else None)
            torch.testing.assert_close(call('fused'),call('torch'),rtol=0,atol=0)
            for mode in ('torch','fused'):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):call(mode)
                torch.cuda.current_stream().wait_stream(stream)
                if execution=='graph':
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): out=call(mode)
                    graphs[mode]=(g,out); funcs[mode]=g.replay
                else:funcs[mode]=lambda mode=mode:call(mode)
            samples={'torch':[],'fused':[]}
            for repeat in range(a.repeats):
                for mode in (('torch','fused') if repeat%2==0 else ('fused','torch')):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(); wall=time.perf_counter();start.record()
                    for _ in range(a.steps):funcs[mode]()
                    end.record();end.synchronize()
                    samples[mode].append({'device_us':start.elapsed_time(end)*1000/a.steps,'wall_us':(time.perf_counter()-wall)*1e6/a.steps})
            case={'length':length,'execution':execution,'samples':samples,
                  'medians':{m:{key:statistics.median(x[key] for x in samples[m]) for key in ('device_us','wall_us')} for m in samples}}
            result['cases'].append(case);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(case['medians'],flush=True)
if __name__=='__main__':main()
