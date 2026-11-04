"""iter108：生产 fine-score kernel 的 Graph 1/2 warp 逐位与性能对拍。"""

import argparse
import json

import torch
import triton
from triton.testing import do_bench

import sparse_attn.indexer.nfa_layer_batch_selector as ops
from benchmark.efficiency.benchmark_nfa_fine_scores import _graph_time, _make_case


def _launch(case, out, warps):
    kernel = ops._nfa_fine_scores_paged_kernel
    kernel[(case['m'] * 8, triton.cdiv(case['r'], 128))](
        case['kv_ptrs'], case['block_table'], case['valid_req'], case['dims'],
        case['q'], case['cand'], case['keep'], case['owners'],
        case['generations'], out, case['lower'], case['upper'], case['quota'],
        case['kv'], case['r'], case['q'].stride(0), case['q'].stride(1),
        case['cand'].stride(1), case['cand'].stride(2),
        S=case['capacity'], MAX_BLOCKS=case['max_blocks'], PAGE=case['page'],
        HK=8, HEAD_DIM=128, DIM_NUMEL=32, D_BLOCK=32, HKV=8,
        B=case['m'], TILE=128, MAX_REQ=case['m'], LAYER_OFFSET=0,
        PACK_KEY=True, num_warps=warps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    all_ok = True
    for seed in range(5):
        case = _make_case(20261108 + seed)
        reference = torch.empty(case['cand'].shape, dtype=torch.int32, device='cuda')
        candidate = torch.empty_like(reference)
        _launch(case, reference, 1)
        _launch(case, candidate, 2)
        torch.cuda.synchronize()
        ok = torch.equal(reference, candidate)
        all_ok &= ok
        print(f'CASE seed={seed} bit_exact={ok}')
        if not ok:
            print('mismatched_keys', (reference != candidate).sum().item())
            return 1

    if args.correctness_only:
        print(json.dumps({'all_ok': all_ok, 'correctness_only': True}))
        return 0 if all_ok else 1

    case = _make_case(20261118, m=1)
    reference = torch.empty(case['cand'].shape, dtype=torch.int32, device='cuda')
    candidate = torch.empty_like(reference)
    ref = lambda: _launch(case, reference, 1)
    new = lambda: _launch(case, candidate, 2)
    ref()
    new()
    torch.cuda.synchronize()
    eager_ref = do_bench(ref, warmup=25, rep=100)
    eager_new = do_bench(new, warmup=25, rep=100)
    graph_ref = _graph_time(ref)
    graph_new = _graph_time(new)
    graph_exact = torch.equal(reference, candidate)
    all_ok &= graph_exact
    print(json.dumps({
        'shape': [1, 8, case['r']], 'dims': 32,
        'eager_reference_ms': eager_ref, 'eager_candidate_ms': eager_new,
        'eager_speedup': eager_ref / eager_new,
        'graph_reference_ms': graph_ref, 'graph_candidate_ms': graph_new,
        'graph_speedup': graph_ref / graph_new,
        'graph_bit_exact': graph_exact, 'all_ok': all_ok}))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
