"""iter107：2048-wide partial Top-K 的 8/16 warp 逐位与性能对拍。"""

import json

import torch
import triton
from triton.testing import do_bench

import sparse_attn.indexer.nfa_layer_batch_selector as ops
from benchmark.efficiency.benchmark_nfa_topk_merge import _graph_time


def _make_logits(seed, m=3, hkv=8, r=8452, capacity=32768):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    cand = torch.stack([
        torch.randperm(capacity, generator=generator, device='cuda')[:r]
        for _ in range(m * hkv)
    ]).view(m, hkv, r).to(torch.int64).contiguous()
    logits = torch.randn(
        m, hkv, r, generator=generator, dtype=torch.bfloat16,
        device='cuda').contiguous()
    logits[:, :, 128:512] = 0.25
    logits[:, :, 512:640] = float('-inf')
    cand[:, :, 640:768] = capacity - 1
    lower = torch.tensor([128, 0, 0], dtype=torch.int64, device='cuda')[:m]
    upper = torch.tensor([8000, 64, 4096], dtype=torch.int64, device='cuda')[:m]
    quota = torch.tensor([1024, 300, 0], dtype=torch.int64, device='cuda')[:m]
    bits = logits.view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    ordered = torch.where(negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
    score = logits.to(torch.float32)
    good = ((cand >= lower[:, None, None]) & (cand < upper[:, None, None])
            & (quota[:, None, None] > 0) & (score == score)
            & (score != float('-inf')))
    invalid = torch.full((), -32768 << 16, dtype=torch.int32, device='cuda')
    packed = torch.where(
        good, (ordered << 16) + capacity - cand.to(torch.int32), invalid)
    return packed.contiguous(), capacity


def _launch(logits, out, capacity, warps):
    m, hkv, r = logits.shape
    ntiles = triton.cdiv(r, 2048)
    ops._nfa_filtered_topk_partial_kernel[(m * hkv, ntiles)](
        logits, out, r, TILE=2048, NTILES=ntiles,
        S=capacity, K=1024, META_GROUP=hkv, num_warps=warps)


def main():
    torch.cuda.set_device(0)
    all_ok = True
    for seed in range(10):
        logits, capacity = _make_logits(20261107 + seed)
        shape = (logits.shape[0], logits.shape[1], 5, 1024)
        reference = torch.empty(shape, dtype=torch.int32, device='cuda')
        candidate = torch.empty_like(reference)
        _launch(logits, reference, capacity, 8)
        _launch(logits, candidate, capacity, 16)
        torch.cuda.synchronize()
        ok = torch.equal(reference, candidate)
        all_ok &= ok
        print(f'CASE seed={seed} bit_exact={ok}')
        if not ok:
            print('mismatched_keys', (reference != candidate).sum().item())
            return 1

    logits, capacity = _make_logits(20261117, m=1)
    reference = torch.empty((1, 8, 5, 1024), dtype=torch.int32, device='cuda')
    candidate = torch.empty_like(reference)
    ref = lambda: _launch(logits, reference, capacity, 8)
    new = lambda: _launch(logits, candidate, capacity, 16)
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
        'shape': [1, 8, 8452], 'width': 1024, 'ntiles': 5,
        'eager_reference_ms': eager_ref, 'eager_candidate_ms': eager_new,
        'eager_speedup': eager_ref / eager_new,
        'graph_reference_ms': graph_ref, 'graph_candidate_ms': graph_new,
        'graph_speedup': graph_ref / graph_new,
        'graph_bit_exact': graph_exact, 'all_ok': all_ok}))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
