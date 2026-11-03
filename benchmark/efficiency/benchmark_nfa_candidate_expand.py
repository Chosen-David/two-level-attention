"""NFA L1 Top-K/candidate 并行展开的逐位对拍与独立微基准。

只比较 candidate 阶段：旧单 CTA fused kernel 对拍新的 prepare + 并行 expand。
下游 fine score、packed key、partial/merge Top-K 与 finish 均不在本脚本中修改。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sparse_attn.indexer.nfa_layer_batch_selector import (
    _candidate_expand_batched_kernel,
    _topk_candidate_batched_kernel,
    _topk_candidate_prepare_kernel,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, choices=(1, 2, 4))
    parser.add_argument('--graph', action='store_true', help='计时使用 CUDA Graph replay')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--output')
    return parser.parse_args()


# Qwen3-8B RULER32K B1 生产几何。R 是 middle virtual slots，实际候选行宽
# 还包含 HEAD+TAIL=132，因此 fine/partial 看到的行宽为 8452、共 5 个 tile。
GEOMETRY = dict(
    HKV=8, KT=512, KT_PAD=512,
    K_F=96, K_N=32, K_F_PAD=128, K_N_PAD=32,
    R=8320, HEAD=4, TAIL=128,
    SINK=4, SWA=128, CAP=32768,
    SINK_POOL_LO=0, BS=64, PAGE=64,
    NEAR_LEN=4096, K2=1024,
    MAX_REQ=8, LAYER_OFFSET=0,
)
WIDTH = GEOMETRY['R'] + GEOMETRY['HEAD'] + GEOMETRY['TAIL']


def _make_state(batch, length, seed, generation, shared_scores=False, wrap=False):
    g = GEOMETRY
    hkv = g['HKV']
    gen = torch.Generator(device='cpu').manual_seed(seed)
    # 小整数 BF16 分数制造大量精确同分；独立/共享两路分别覆盖常规与重复 token。
    far_cpu = torch.randint(-16, 17, (batch, hkv, g['KT']),
                            generator=gen, dtype=torch.int16).to(torch.bfloat16)
    near_cpu = (far_cpu.clone() if shared_scores else
                torch.randint(-16, 17, (batch, hkv, g['KT']),
                              generator=gen, dtype=torch.int16).to(torch.bfloat16))
    # 保留足够有效块，同时让部分 Top-K 输入为 -inf。
    far_cpu[:, :, ::17] = float('-inf')
    near_cpu[:, :, 1::19] = float('-inf')
    lengths = torch.arange(batch, dtype=torch.int64) * 37 + length
    state = {
        'far': far_cpu.cuda(), 'near': near_cpu.cuda(),
        'i_f': torch.empty((batch, hkv, g['K_F']), dtype=torch.int64, device='cuda'),
        'i_n': torch.empty((batch, hkv, g['K_N']), dtype=torch.int64, device='cuda'),
        'lengths': lengths.cuda(),
        'skip_far': torch.zeros((1, g['MAX_REQ']), dtype=torch.int32, device='cuda'),
        'skip_near': torch.zeros((1, g['MAX_REQ']), dtype=torch.int32, device='cuda'),
        'valid_req': torch.arange(batch, dtype=torch.int64, device='cuda'),
        'out': torch.empty((batch, hkv, WIDTH), dtype=torch.int64, device='cuda'),
        'valid': torch.empty((batch, hkv, WIDTH), dtype=torch.bool, device='cuda'),
        'lower': torch.empty((batch,), dtype=torch.int64, device='cuda'),
        'upper': torch.empty((batch,), dtype=torch.int64, device='cuda'),
        'quota': torch.empty((batch,), dtype=torch.int64, device='cuda'),
        'owners': torch.zeros((batch, hkv, g['CAP']), dtype=torch.int32, device='cuda'),
        'generations': torch.full((batch, hkv), generation, dtype=torch.int32, device='cuda'),
    }
    if wrap:
        state['owners'].fill_(2147483647)
    return state


def _clone_state(state):
    return {name: value.clone() for name, value in state.items()}


def _reference(state):
    g = GEOMETRY
    batch = state['lengths'].numel()
    _topk_candidate_batched_kernel[(batch * g['HKV'],)](
        state['far'], state['near'], state['i_f'], state['i_n'], state['lengths'],
        state['skip_far'], state['skip_near'], state['valid_req'],
        state['out'], state['valid'], state['lower'], state['upper'],
        state['quota'], state['owners'], state['generations'],
        KT=g['KT'], KT_PAD=g['KT_PAD'], K_F=g['K_F'], K_N=g['K_N'],
        K_F_PAD=g['K_F_PAD'], K_N_PAD=g['K_N_PAD'],
        R=g['R'], HEAD=g['HEAD'], TAIL=g['TAIL'], SINK=g['SINK'],
        SWA=g['SWA'], CAP=g['CAP'], SINK_POOL_LO=g['SINK_POOL_LO'],
        BS=g['BS'], PAGE=g['PAGE'], NEAR_LEN=g['NEAR_LEN'], K2=g['K2'],
        HKV=g['HKV'], B=batch, MAX_REQ=g['MAX_REQ'],
        LAYER_OFFSET=g['LAYER_OFFSET'], num_warps=8)


def _candidate(state):
    g = GEOMETRY
    batch = state['lengths'].numel()
    _topk_candidate_prepare_kernel[(batch * g['HKV'],)](
        state['far'], state['near'], state['i_f'], state['i_n'], state['lengths'],
        state['skip_far'], state['skip_near'], state['valid_req'],
        state['lower'], state['upper'], state['quota'],
        state['owners'], state['generations'],
        KT=g['KT'], KT_PAD=g['KT_PAD'], K_F=g['K_F'], K_N=g['K_N'],
        K_F_PAD=g['K_F_PAD'], K_N_PAD=g['K_N_PAD'],
        R=g['R'], HEAD=g['HEAD'], TAIL=g['TAIL'], SINK=g['SINK'],
        SWA=g['SWA'], CAP=g['CAP'], SINK_POOL_LO=g['SINK_POOL_LO'],
        BS=g['BS'], PAGE=g['PAGE'], NEAR_LEN=g['NEAR_LEN'], K2=g['K2'],
        HKV=g['HKV'], B=batch, MAX_REQ=g['MAX_REQ'],
        LAYER_OFFSET=g['LAYER_OFFSET'], num_warps=8)
    _candidate_expand_batched_kernel[(batch * g['HKV'], (WIDTH + 1023) // 1024)](
        state['i_f'], state['i_n'], state['far'], state['near'], state['lengths'],
        state['out'], state['valid'], state['owners'], state['generations'],
        K_F=g['K_F'], K_N=g['K_N'], KT=g['KT'],
        R=g['R'], HEAD=g['HEAD'], TAIL=g['TAIL'], SINK=g['SINK'],
        SWA=g['SWA'], CAP=g['CAP'], SINK_POOL_LO=g['SINK_POOL_LO'],
        BS=g['BS'], HKV=g['HKV'], B=batch, num_warps=4)


def _assert_bit_exact(reference, candidate, scenario):
    for name in reference:
        if not torch.equal(reference[name], candidate[name]):
            ref = reference[name].cpu()
            cand = candidate[name].cpu()
            mismatch = (ref != cand).nonzero()
            first = mismatch[0].tolist() if mismatch.numel() else []
            raise AssertionError(
                f'{scenario}: {name} differs at {first}; '
                f'ref={ref[tuple(first)].item() if first else None}, '
                f'cand={cand[tuple(first)].item() if first else None}')


def _correctness(batch, seed, use_graph=False):
    scenarios = [
        dict(length=32612, generation=7, shared_scores=False, wrap=False),
        dict(length=32612, generation=11, shared_scores=True, wrap=False),
        dict(length=73, generation=3, shared_scores=True, wrap=False),
        dict(length=32612, generation=2147483647 // WIDTH,
             shared_scores=True, wrap=True),
    ]
    for index, scenario in enumerate(scenarios):
        initial = _make_state(batch, seed=seed + index, **scenario)
        reference = _clone_state(initial)
        candidate = _clone_state(initial)
        if use_graph:
            reference_graph = _capture(lambda: _reference(reference))
            candidate_graph = _capture(lambda: _candidate(candidate))
            reference_graph.replay()
            candidate_graph.replay()
        else:
            _reference(reference)
            _candidate(candidate)
        _assert_bit_exact(
            reference, candidate,
            f'{"graph" if use_graph else "eager"}-scenario-{index}')


def _capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _elapsed_ms(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _performance(batch, seed, use_graph, warmup, iters):
    initial = _make_state(batch, 32612, seed + 100, 23)
    states = {
        'reference': _clone_state(initial),
        'candidate': _clone_state(initial),
    }
    calls = {
        'reference': lambda: _reference(states['reference']),
        'candidate': lambda: _candidate(states['candidate']),
    }
    # 编译与初次执行必须在 Graph capture 外完成。
    for fn in calls.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    if use_graph:
        graphs = {name: _capture(fn) for name, fn in calls.items()}
        calls = {name: graph.replay for name, graph in graphs.items()}
    for fn in calls.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = {'reference': [], 'candidate': []}
    # ABBA 降低同进程时序与频率漂移；每次计时包含固定次数调用/replay。
    for name in ('reference', 'candidate', 'candidate', 'reference'):
        samples[name].append(_elapsed_ms(calls[name], iters))
    return {
        name: {
            'samples_ms': values,
            'median_ms': statistics.median(values),
        }
        for name, values in samples.items()
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    _correctness(args.batch_size, args.seed)
    if args.graph:
        # 上一行已先完成两臂 JIT；此处验证 capture/replay 本身不改变任何槽位。
        _correctness(args.batch_size, args.seed, use_graph=True)
    timing = _performance(
        args.batch_size, args.seed, args.graph, args.warmup, args.iters)
    reference_ms = timing['reference']['median_ms']
    candidate_ms = timing['candidate']['median_ms']
    result = {
        'scope': 'L1 Top-K + generation/metadata + candidate expansion only',
        'geometry': {**GEOMETRY, 'candidate_width': WIDTH,
                     'expand_chunks': (WIDTH + 1023) // 1024},
        'batch_size': args.batch_size,
        'execution': 'graph' if args.graph else 'eager',
        'correctness': {
            'all_scenarios_bit_exact': True,
            'graph_capture_replay_bit_exact': bool(args.graph),
            'covers': ['production', 'ties-and-duplicates', 'short-quota',
                       'owner-generation-wrap'],
        },
        'timing': timing,
        'reference_over_candidate': reference_ms / candidate_ms,
        'warmup': args.warmup,
        'iters_per_sample': args.iters,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')


if __name__ == '__main__':
    main()
