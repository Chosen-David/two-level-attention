#!/usr/bin/env python3
"""B1 GQA gather-attention 的 heads/CTA 微基准与逐位对拍。"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_attn.ops.paged_region_kernels import _gather_attend_grouped_kernel


def run_kernel(q, k, v, gather, lens, query_to_req, heads_per_program, warps):
    total_q, hq, dim = q.shape
    hk = k.shape[2]
    out = q.new_empty((total_q, hq, dim))
    _gather_attend_grouped_kernel[(total_q, hq // heads_per_program)](
        q, k, v, gather, lens, query_to_req, out, total_q,
        HQ=hq, HK=hk, D=dim, DV=dim, PAGE=128, LOG2_PAGE=7,
        MAX_VISIBLE=gather.shape[-1], MAX_BATCH=1, BT=128,
        SCALE=dim ** -0.5, BD=dim, BV=dim, HEAD_GROUP=hq // hk,
        HEAD_TILE=16, HEADS_PER_PROGRAM=heads_per_program,
        USE_BF16=True, num_warps=warps,
    )
    return out


def graph_time(fn, reps=7, replays=100):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / replays)
    return statistics.median(samples), static_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', default='1,2,4')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    generator = torch.Generator(device='cpu').manual_seed(100)
    hq, hk, dim, pages, page, max_visible = 32, 8, 128, 512, 128, 1280
    q = torch.randn((1, hq, dim), generator=generator).bfloat16().to(device)
    k = torch.randn((pages, 1, hk, page, dim), generator=generator).bfloat16().to(device)
    v = torch.randn((pages, 1, hk, page, dim), generator=generator).bfloat16().to(device)
    gather = torch.empty((1, hk, max_visible), dtype=torch.int32)
    for head in range(hk):
        gather[0, head] = torch.randperm(pages * page, generator=generator)[:max_visible]
    gather = gather.to(device)
    lens = torch.full((1, hk), 1156, dtype=torch.int32, device=device)
    query_to_req = torch.zeros((1,), dtype=torch.int32, device=device)

    reference = None
    results = []
    for heads in [int(item) for item in args.heads.split(',')]:
        fn = lambda: run_kernel(q, k, v, gather, lens, query_to_req, heads, 8)
        eager = fn()
        if reference is None:
            reference = eager
        bit_exact = torch.equal(eager.view(torch.int16), reference.view(torch.int16))
        elapsed, graph_out = graph_time(fn)
        graph_exact = torch.equal(graph_out.view(torch.int16), reference.view(torch.int16))
        results.append((heads, elapsed, bit_exact and graph_exact))
        print(f'heads/CTA={heads}: graph={elapsed:.6f} ms, '
              f'eager+graph bit-exact={bit_exact and graph_exact}')
    print(f'summary={results}')
    return 0 if all(item[2] for item in results) else 1


if __name__ == '__main__':
    sys.exit(main())
