"""Real-GPU validation/stress entry; fail closed on CPU or interpreter.

Runs existing repository gates, new CUDA correctness/capture tests, then paired
indexer measurements. Full-model quality/TPOT is a separate acceptance gate.
Each case gets a fresh process so graph pools/JIT history cannot cross cases.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--outdir',required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[4])
    p.add_argument('--lengths',type=int,nargs='+',default=[32768])
    p.add_argument('--layouts',nargs='+',choices=['head','token'],default=['head'])
    p.add_argument('--l2-subspaces',nargs='+',choices=['same','full'],default=['same'])
    p.add_argument('--methods',nargs='+',choices=['mixed','avg','minmax'],default=['mixed'])
    p.add_argument('--skip-modes',nargs='+',choices=['on','off'],default=['off','on'])
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='bfloat16')
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--sanitizer',action='store_true',help='also run new kernel gates under compute-sanitizer memcheck')
    a=p.parse_args()
    if os.environ.get('TRITON_INTERPRET')=='1' or not torch.cuda.is_available():
        p.error('real CUDA required; CPU/interpreter runs cannot produce GPU receipts')
    if min(a.batches+a.lengths+[a.steps,a.repeats])<1:p.error('positive dimensions required')
    import triton
    out=Path(a.outdir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():p.error('use a fresh outdir; never overwrite previous receipts')
    root=Path(__file__).resolve().parents[2]
    env=os.environ.copy()
    env.update(TLI_BATCH_GPU_PIPELINE='0',TLI_BATCH_FUSIONS='0',TLI_BATCH_SCORE='torch',
               TLI_BATCH_INDEXER='legacy')
    sources=[root/'exp/bench/tli_batch_round_compare.py',Path(__file__),
             root/'tests/test_tli_gpu_pipeline.py',root/'sparse_attn/indexer/tli_indexer.py',
             root/'sparse_attn/metrics.py']+list((root/'sparse_attn/indexer').glob('tli_batch*.py'))
    manifest=dict(status='running',args=vars(a),torch=torch.__version__,triton=triton.__version__,
        cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        runs=[],limitations=['indexer timing excludes attention, KV append and metadata upload',
                            'not full-model TPOT/throughput or RULER acceptance'])
    try:
        manifest['driver']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    except (FileNotFoundError,subprocess.CalledProcessError):manifest['driver']='query unavailable'
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def run(label,cmd):
        begin=time.time()
        with (out/(label+'.log')).open('w') as log:
            try:
                proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                code=proc.returncode
            except OSError as error:
                log.write(str(error));code=127
        manifest['runs'].append(dict(label=label,command=cmd,returncode=code,seconds=time.time()-begin))
        save();print(f'{label}: exit {code}',flush=True)
        return code==0
    save()
    gates=[('e127',[sys.executable,'exp/bench/test_e127_gates.py']),
           ('batch-gates',[sys.executable,'exp/bench/test_tli_batch_gates.py']),
           ('pipeline-gates',[sys.executable,'-m','pytest','-q','tests/test_tli_gpu_pipeline.py',
                              'tests/test_tli_batch_parallel.py','tests/test_tli_batch_fusion_kernels.py','tests/test_far_skip_residency.py','--tb=short'])]
    if a.sanitizer:
        gates.append(('memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','86',
            sys.executable,'-m','pytest','-q','tests/test_tli_gpu_pipeline.py','--tb=short']))
    for label,cmd in gates:
        if not run(label,cmd):
            manifest['status']='gate_failed';save();raise SystemExit('gate failed; logs retained')
    failed=False
    matrix=itertools.product(a.batches,a.lengths,a.layouts,a.l2_subspaces,a.methods,
                             (False,True),('score','movement'),a.skip_modes)
    for b,length,layout,subspace,method,capture,baseline,skip_mode in matrix:
        label=f'b{b}-s{length}-{layout}-{subspace}-{method}-'+('graph' if capture else 'eager')+'-'+baseline+'-skip-'+skip_mode
        receipt=out/(label+'.json')
        cmd=[sys.executable,'-m','exp.bench.tli_batch_round_compare',
             '--baseline-root','sparse_attn/indexer','--device','cuda','--dtype',a.dtype,
             '--score','triton','--fusions','--gpu-pipeline','--method',method,
             '--batch',str(b),'--length',str(length),'--layout',layout,'--l2-subspace',subspace,
             '--steps',str(a.steps),'--repeats',str(a.repeats),'--out',str(receipt)]
        if skip_mode=='on':cmd.append('--skip-far')
        if baseline=='movement':cmd.append('--baseline-fusions')
        if capture:cmd.append('--capture')
        if a.profile:cmd.append('--profile')
        if not run(label,cmd):failed=True
    manifest['status']='comparison_failed' if failed else 'gpu_indexer_checks_passed_model_gates_pending'
    save()
    if failed:raise SystemExit('one or more comparisons failed; inspect per-case logs/receipts')


if __name__=='__main__':main()
