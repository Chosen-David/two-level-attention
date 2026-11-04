"""iter110：生产有序 merge 的 4/8/16 warp 逐位与性能扫描。"""

import argparse
import json

import torch
from triton.testing import do_bench

import sparse_attn.indexer.nfa_layer_batch_selector as ops
from benchmark.efficiency.benchmark_nfa_topk_merge import _graph_time, _make_partial


def _launch(partial, lower, upper, quota, out, count, capacity, warps):
    m, hkv, ntiles, width = partial.shape
    ops._nfa_filtered_topk_merge_kernel[(m * hkv,)](
        partial, lower, upper, quota, out, count,
        NTILES=ntiles, MERGE_PAD=1 << (ntiles * width - 1).bit_length(),
        LOG2_K=10, S=capacity, K=width, SINK=4,
        META_GROUP=hkv, TOKEN_BITS=16, num_warps=warps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    all_ok = True
    for seed in range(10):
        partial, lower, upper, quota, capacity = _make_partial(20261110 + seed)
        reference = torch.empty((3, 8, 1024), dtype=torch.int32, device='cuda')
        ref_count = torch.empty((3, 8), dtype=torch.int32, device='cuda')
        _launch(partial, lower, upper, quota, reference, ref_count, capacity, 8)
        for warps in (4, 16):
            candidate = torch.empty_like(reference)
            count = torch.empty_like(ref_count)
            _launch(partial, lower, upper, quota, candidate, count, capacity, warps)
            torch.cuda.synchronize()
            ok = (torch.equal(reference, candidate)
                  and torch.equal(ref_count, count))
            all_ok &= ok
            print(f'CASE seed={seed} warps={warps} bit_exact={ok}')
            if not ok:
                print('mismatched_indices', (reference != candidate).sum().item())
                print('mismatched_counts', (ref_count != count).sum().item())
                return 1

    if args.correctness_only:
        print(json.dumps({'all_ok': all_ok, 'correctness_only': True}))
        return 0 if all_ok else 1

    partial, lower, upper, quota, capacity = _make_partial(20261120, m=1)
    # _make_partial 的元数据构造保留 B3 语义；性能臂显式截取 B1。
    partial = partial[:1].contiguous()
    lower, upper, quota = lower[:1], upper[:1], quota[:1]
    results = {}
    for warps in (4, 8, 16):
        out = torch.empty((1, 8, 1024), dtype=torch.int32, device='cuda')
        count = torch.empty((1, 8), dtype=torch.int32, device='cuda')
        launch = lambda: _launch(
            partial, lower, upper, quota, out, count, capacity, warps)
        launch()
        torch.cuda.synchronize()
        eager = do_bench(launch, warmup=25, rep=100)
        graph = _graph_time(launch)
        results[str(warps)] = {
            'eager_ms': eager, 'graph_ms': graph,
            'out': out, 'count': count}
    reference = results['8']
    for warps in (4, 16):
        exact = (torch.equal(reference['out'], results[str(warps)]['out'])
                 and torch.equal(reference['count'], results[str(warps)]['count']))
        all_ok &= exact
        results[str(warps)]['graph_bit_exact'] = exact
    for value in results.values():
        del value['out'], value['count']
    print(json.dumps({
        'shape': [1, 8, 5, 1024], 'results': results,
        'eager_speedup_8_over_4': reference['eager_ms'] / results['4']['eager_ms'],
        'eager_speedup_8_over_16': reference['eager_ms'] / results['16']['eager_ms'],
        'graph_speedup_8_over_4': reference['graph_ms'] / results['4']['graph_ms'],
        'graph_speedup_8_over_16': reference['graph_ms'] / results['16']['graph_ms'],
        'all_ok': all_ok}))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
