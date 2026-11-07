"""iter109：16 warp 下 2048/4096-token partial Top-K 全链路对拍。"""

import argparse
import json

import torch
import triton
from triton.testing import do_bench

import sparse_attn.indexer.nfa_layer_batch_selector as ops
from benchmark.efficiency.benchmark_nfa_topk_merge import _graph_time
from benchmark.efficiency.benchmark_nfa_topk_partial import _make_logits


def _metadata(m):
    lower = torch.tensor([128, 0, 0], dtype=torch.int64, device='cuda')[:m]
    upper = torch.tensor([8000, 64, 4096], dtype=torch.int64, device='cuda')[:m]
    quota = torch.tensor([1024, 300, 0], dtype=torch.int64, device='cuda')[:m]
    return lower, upper, quota


def _launch(logits, partial, lower, upper, quota, out, count, capacity, tile):
    m, hkv, r = logits.shape
    ntiles = triton.cdiv(r, tile)
    tail = r - (ntiles - 1) * tile
    ops._nfa_filtered_topk_partial_kernel[(m * hkv, ntiles)](
        logits, partial, r, TILE=tile, NTILES=ntiles,
        TAIL=tail, TAIL_PAD=triton.next_power_of_2(tail),
        S=capacity, K=1024, META_GROUP=hkv, num_warps=16)
    ops._nfa_filtered_topk_merge_kernel[(m * hkv,)](
        partial, lower, upper, quota, out, count,
        NTILES=ntiles, MERGE_PAD=triton.next_power_of_2(ntiles * 1024),
        LOG2_K=10, S=capacity, K=1024, SINK=4,
        META_GROUP=hkv, TOKEN_BITS=16, num_warps=8)


def _buffers(logits, tile):
    m, hkv, r = logits.shape
    partial = torch.empty(
        (m, hkv, triton.cdiv(r, tile), 1024), dtype=torch.int32, device='cuda')
    out = torch.empty((m, hkv, 1024), dtype=torch.int32, device='cuda')
    count = torch.empty((m, hkv), dtype=torch.int32, device='cuda')
    return partial, out, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    all_ok = True
    for seed in range(10):
        logits, capacity = _make_logits(20261109 + seed)
        lower, upper, quota = _metadata(logits.shape[0])
        ref_partial, ref_out, ref_count = _buffers(logits, 2048)
        new_partial, new_out, new_count = _buffers(logits, 4096)
        _launch(logits, ref_partial, lower, upper, quota,
                ref_out, ref_count, capacity, 2048)
        _launch(logits, new_partial, lower, upper, quota,
                new_out, new_count, capacity, 4096)
        torch.cuda.synchronize()
        ok = (torch.equal(ref_out, new_out)
              and torch.equal(ref_count, new_count))
        all_ok &= ok
        print(f'CASE seed={seed} bit_exact={ok}')
        if not ok:
            print('mismatched_indices', (ref_out != new_out).sum().item())
            print('mismatched_counts', (ref_count != new_count).sum().item())
            return 1

    if args.correctness_only:
        print(json.dumps({'all_ok': all_ok, 'correctness_only': True}))
        return 0 if all_ok else 1

    logits, capacity = _make_logits(20261119, m=1)
    lower, upper, quota = _metadata(1)
    ref_partial, ref_out, ref_count = _buffers(logits, 2048)
    new_partial, new_out, new_count = _buffers(logits, 4096)
    ref = lambda: _launch(logits, ref_partial, lower, upper, quota,
                          ref_out, ref_count, capacity, 2048)
    new = lambda: _launch(logits, new_partial, lower, upper, quota,
                          new_out, new_count, capacity, 4096)
    ref()
    new()
    torch.cuda.synchronize()
    eager_ref = do_bench(ref, warmup=25, rep=100)
    eager_new = do_bench(new, warmup=25, rep=100)
    graph_ref = _graph_time(ref)
    graph_new = _graph_time(new)
    graph_exact = (torch.equal(ref_out, new_out)
                   and torch.equal(ref_count, new_count))
    all_ok &= graph_exact
    print(json.dumps({
        'shape': [1, 8, 8452], 'width': 1024,
        'reference_tile': 2048, 'candidate_tile': 4096,
        'eager_reference_ms': eager_ref, 'eager_candidate_ms': eager_new,
        'eager_speedup': eager_ref / eager_new,
        'graph_reference_ms': graph_ref, 'graph_candidate_ms': graph_new,
        'graph_speedup': graph_ref / graph_new,
        'graph_bit_exact': graph_exact, 'all_ok': all_ok}))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
