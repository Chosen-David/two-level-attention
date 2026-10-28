"""CUDA-only paired projection maintenance microbenchmark, not model TPOT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from sparse_attn.ops.far_skip import FarSkipConfig, FarSkipLayout
from sparse_attn.ops.far_skip_subspace import FarSkipFineCache


def reference(fine, k, position):
    # Frozen main 17e56e8 baseline. No dispatch through the candidate.
    if isinstance(position, torch.Tensor):
        physical = fine.layout.map(position)
        values = k[0][physical[None, :, None], fine.heads[:, None, None], fine.dims]
        fine.keys.index_copy_(1, physical, values)
    else:
        length = position+1
        if fine.last_length == length:
            return
        if fine.last_length != length-1:
            fine.initialize(k, length)
            return
        slot = fine.layout.slot(position)
        fine.keys[:, slot].copy_(k[0, slot].index_select(-1, fine.dims))
        fine.last_length = length


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required; no CPU timing substitute')
    if min(a.batches) < 1 or a.steps < 1 or a.repeats < 2: p.error('positive batch/steps and repeats>=2 required')
    os.environ['TLI_FAR_SKIP_KERNELS'] = 'triton'
    torch.manual_seed(20261027)
    root = Path(__file__).resolve().parents[2]
    files = ['sparse_attn/ops/far_skip_subspace.py', 'sparse_attn/ops/far_skip_accel.py', 'sparse_attn/ops/far_skip_kernels.py', 'exp/bench/far_skip_projection_compare.py']
    result = dict(scope='projection update only, fixed current token, sequential per-row dispatch; not TPOT',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  hashes={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}, cases=[])
    for batch in a.batches:
        cfg = FarSkipConfig(64, 4096, 32, 1024, 4, 128, quantize=False)
        layout = FarSkipLayout(4, 4287)
        dims = torch.arange(96, 128, device='cuda')
        rows = []
        for b in range(batch):
            k = torch.randn(1, 8, layout.capacity, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
            ref, fused = [FarSkipFineCache(cfg, layout, dims) for _ in range(2)]
            for fine in (ref, fused): fine.initialize(k, 32768+b)
            rows.append((k, ref, fused))
        for execution in ('staged_eager', 'tensor_eager', 'graph'):
            positions = [32768+b if execution == 'staged_eager' else torch.tensor([32768+b], device='cuda') for b in range(batch)]
            def call(mode):
                for (k, ref, fused), pos in zip(rows, positions):
                    if execution == 'staged_eager':
                        # Both arms model the same next-token host metadata.
                        (ref if mode == 'torch' else fused).last_length = pos
                    if mode == 'torch': reference(ref, k, pos)
                    elif execution == 'staged_eager':
                        fused.append(k, pos+1)
                    else: fused.update_tensor(k, pos)
            call('torch'); call('fused')
            for _, ref, fused in rows: torch.testing.assert_close(fused.keys, ref.keys, rtol=0, atol=0)
            graphs, funcs = {}, {}
            for mode in ('torch', 'fused'):
                stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5): call(mode)
                torch.cuda.current_stream().wait_stream(stream)
                if execution == 'graph':
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): call(mode)
                    graphs[mode], funcs[mode] = g, g.replay
                else: funcs[mode] = lambda mode=mode: call(mode)
            samples = {'torch': [], 'fused': []}
            for repeat in range(a.repeats):
                for mode in (('torch', 'fused') if repeat%2 == 0 else ('fused', 'torch')):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(); wall = time.perf_counter(); start.record()
                    for _ in range(a.steps): funcs[mode]()
                    end.record(); end.synchronize()
                    samples[mode].append(dict(device_us=start.elapsed_time(end)*1000/a.steps, wall_us=(time.perf_counter()-wall)*1e6/a.steps))
            for _, ref, fused in rows: torch.testing.assert_close(fused.keys, ref.keys, rtol=0, atol=0)
            case = dict(batch=batch, execution=execution, samples=samples,
                        medians={m: {key: statistics.median(x[key] for x in samples[m]) for key in ('device_us', 'wall_us')} for m in samples})
            result['cases'].append(case)
            Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
            print(case['medians'], flush=True)

if __name__ == '__main__': main()
