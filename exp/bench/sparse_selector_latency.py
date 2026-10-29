"""Paired CUDA Graph selector timing; reports raw data pending review."""
import argparse
import hashlib
import subprocess
import importlib.util
import json
import os
from pathlib import Path
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    args=parser.parse_args()
    from operator_model_paired import require_lease, snapshot
    require_lease()
    idle=snapshot()
    if idle['processes']:
        raise RuntimeError('GPU1 occupied before measurement')
    root=Path(__file__).resolve().parents[2]
    sources=[p for p in (root/'sparse_attn').rglob('*') if p.suffix in ('.py','.cpp','.cu','.cuh')]
    sources += [Path(__file__).resolve(),Path(__file__).with_name('test_tli_graph_gates.py'),Path(__file__).with_name('operator_model_paired.py')]
    receipt=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch=torch.__version__,cuda=torch.version.cuda,initial_gpu=idle)
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    telemetry=[]
    observed_pid=None
    spec=importlib.util.spec_from_file_location('prod',Path(__file__).with_name('test_tli_graph_gates.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.manual_seed(887)
    capacity=32840
    q=torch.randn(1,1,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,capacity,8,128,device='cuda',dtype=q.dtype)
    vl=torch.tensor([32768],device='cuda');cu=torch.tensor([0,capacity],device='cuda',dtype=torch.int32)
    pinned=[];graphs={};outputs={}
    for arm in ('dense','compact','sparse'):
        os.environ['TLI_COMPACT_NORMALIZATION']=str(int(arm=='compact'))
        os.environ['TLI_SPARSE_SELECTION']=str(int(arm=='sparse'))
        idx=TLIIndexer(mod._make_prod_args());idx.layer_idx=2;idx.clear()
        index=idx.prepare_index(k,cu)
        for _ in range(3):idx.compute_selection_graph(q,index,128**-.5,vl)
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=idx.compute_selection_graph(q,index,128**-.5,vl)
        pinned.append((idx,index));graphs[arm]=g;outputs[arm]=out
    for g in graphs.values():
        for _ in range(50):g.replay()
    torch.cuda.synchronize()
    for arm,out in outputs.items():
        assert torch.equal(out.indices,outputs['dense'].indices),arm
        assert torch.equal(out.counts,outputs['dense'].counts),arm
    rows=[]
    for cycle in range(5):
        for arm in ('dense','compact','sparse','sparse','compact','dense'):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):graphs[arm].replay()
            end.record();end.synchronize()
            rows.append(dict(cycle=cycle,arm=arm,us=start.elapsed_time(end)*10))
            sample=snapshot();telemetry.append(sample)
            pids={int(r.split(',')[1]) for r in sample['processes']}
            if len(pids)!=1:
                raise RuntimeError('GPU process contention or missing ownership')
            current=next(iter(pids))
            if observed_pid is None:observed_pid=current
            if current!=observed_pid:raise RuntimeError('GPU process changed')
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):graphs['sparse'].replay()
            torch.cuda.synchronize()
        args.profile.write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
    args.out.write_text(json.dumps(dict(status='pending-independent-review',scope='selector only, not model TPOT; process sampling between timing batches',telemetry=telemetry,capacity=capacity,valid_len=32768,rows=rows),indent=2))

if __name__=='__main__':main()
