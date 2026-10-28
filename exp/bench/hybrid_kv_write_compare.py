"""Paired CUDA write-only benchmark. Not attention/TPOT or allocator evidence.

Reference is the exact index_select/where/index_copy sequence in main b0a378c.
Changing only TLI_FAR_SKIP_KERNELS selects the new fused row write. Both paths
still launch per row; this does not claim whole-batch index/attention fusion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from sparse_attn.ops.far_skip import FarSkipLayout
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.hybrid_batch_cache import write_hybrid_row


def rows(batch, capacity, mode):
    result = []
    for b in range(batch):
        ring = mode == 'ring' or (mode == 'mixed' and b % 2 == 0)
        row = object.__new__(FarSkipStaticLayer) if ring else SimpleNamespace()
        if ring:
            row.layout = FarSkipLayout(4, 4287)
        size = row.layout.capacity if ring else capacity
        row.keys, row.values = [torch.zeros(1, 8, size, 128, device='cuda', dtype=torch.bfloat16)
                                for _ in range(2)]
        result.append(row)
    return result


def timed(fn, steps):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    begin.record()
    for _ in range(steps):
        fn()
    end.record(); end.synchronize()
    return dict(device_us=begin.elapsed_time(end)*1000/steps,
                wall_us=(time.perf_counter()-wall)*1e6/steps)


@torch.inference_mode()
def run(batch, capacity, layout, dead_fraction, execution, steps, repeats):
    states = {mode: rows(batch, capacity, layout) for mode in ('torch', 'triton')}
    k, v = [torch.randn(batch, 8, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    pos = (torch.arange(batch, device='cuda') % 31 + capacity//2).long()
    alive = torch.arange(batch, device='cuda') >= int(batch*dead_fraction)
    def launch(mode):
        os.environ['TLI_FAR_SKIP_KERNELS'] = mode
        for b, row in enumerate(states[mode]):
            write_hybrid_row(row, k[b:b+1], v[b:b+1], pos[b:b+1], alive[b:b+1])
    for mode in states:
        for _ in range(5):
            launch(mode)
    for a, b in zip(states['torch'], states['triton']):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
    graphs, functions = {}, {}
    for mode in states:
        if execution == 'graph':
            # Side-stream warmup precedes capture; no inputs/allocations change.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                launch(mode); launch(mode)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(mode)
            graphs[mode] = graph
            functions[mode] = graph.replay
        else:
            functions[mode] = lambda mode=mode: launch(mode)
    samples = {mode: [] for mode in states}
    for repeat in range(repeats):
        for mode in (('torch', 'triton') if repeat % 2 == 0 else ('triton', 'torch')):
            samples[mode].append(timed(functions[mode], steps))
    medians = {mode: {key: statistics.median(s[key] for s in samples[mode])
                      for key in ('device_us', 'wall_us')} for mode in states}
    return dict(batch=batch, capacity=capacity, layout=layout, dead_fraction=dead_fraction,
                execution=execution, medians=medians, samples=samples,
                bitwise_write_parity=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8])
    p.add_argument('--capacity', type=int, default=32768)
    p.add_argument('--layouts', nargs='+', choices=['ring', 'linear', 'mixed'], default=['ring', 'linear', 'mixed'])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required: no CPU/interpreter performance substitute')
    if min(args.batches) < 1 or args.capacity < 128 or args.steps < 1 or args.repeats < 2:
        p.error('positive batches/steps, capacity >= 128 and repeats >= 2 required')
    root = Path(__file__).resolve().parents[2]
    paths = ['sparse_attn/ops/hybrid_batch_cache.py', 'sparse_attn/ops/far_skip_accel.py',
             'sparse_attn/ops/far_skip_kernels.py', 'exp/bench/hybrid_kv_write_compare.py']
    result = dict(scope='CUDA row KV writes only; no model TPOT or memory claim',
                  baseline_main='b0a378c304cebcba702a1454d0fe8bdf70033926',
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  sources={path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths},
                  settings=vars(args), cases=[])
    for batch in args.batches:
        for layout in args.layouts:
            for dead in (0.0, 0.5, 1.0):
                for execution in ('eager', 'graph'):
                    case = run(batch, args.capacity, layout, dead, execution, args.steps, args.repeats)
                    result['cases'].append(case)
                    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
                    print(batch, layout, dead, execution, case['medians'], flush=True)


if __name__ == '__main__':
    main()
