"""iter104：有序 partial run 的 K-wide merge 与原 2K 网络逐位对拍。"""

import json
import statistics

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

import sparse_attn.indexer.nfa_layer_batch_selector as ops


@triton.jit
def _reference_merge_kernel(
    PARTIAL, LOW, HIGH, QUOTA, OUT, COUNT,
    NTILES: tl.constexpr, MERGE_PAD: tl.constexpr, LOG2_K: tl.constexpr,
    S: tl.constexpr, K: tl.constexpr, SINK: tl.constexpr,
    META_GROUP: tl.constexpr, TOKEN_BITS: tl.constexpr,
):
    """iter103 的 2K bitonic merge，仅作为独立参考，不进入生产。"""
    h = tl.program_id(0)
    i = tl.arange(0, K)
    pair = tl.arange(0, 2 * K)
    rank = tl.where(pair < K, pair, 2 * K - 1 - pair)
    key = tl.load(PARTIAL + h * NTILES * K + i)
    for group in tl.static_range(1, NTILES):
        incoming = tl.load(PARTIAL + (h * NTILES + group) * K + rank)
        retained = tl.gather(key, rank, axis=0)
        value = tl.where(pair < K, retained, incoming)
        peer = tl.gather(value, pair ^ K, axis=0)
        value = tl.where(
            pair < K, tl.maximum(value, peer), tl.minimum(value, peer))
        key = tl.gather(value, i, axis=0)
        for bit in tl.static_range(LOG2_K - 1, -1, -1):
            stride = 1 << bit
            peer = tl.gather(key, i ^ stride, axis=0)
            key = tl.where(
                (i & stride) == 0,
                tl.maximum(key, peer), tl.minimum(key, peer))
    ops._finish_sorted_logits_key(
        h, key, LOW, HIGH, QUOTA, OUT, COUNT,
        S=S, K=K, SINK=SINK, META_GROUP=META_GROUP, TOKEN_BITS=TOKEN_BITS)


def _launch(kernel, partial, lower, upper, quota, out, count, capacity):
    m, hkv, ntiles, width = partial.shape
    kernel[(m * hkv,)](
        partial, lower, upper, quota, out, count,
        NTILES=ntiles, MERGE_PAD=triton.next_power_of_2(ntiles * width),
        LOG2_K=width.bit_length() - 1,
        S=capacity, K=width, SINK=4, META_GROUP=hkv, TOKEN_BITS=16,
        num_warps=8)


def _make_partial(seed, m=3, hkv=8, r=8452, width=1024, capacity=32768):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    cand = torch.stack([
        torch.randperm(capacity, generator=generator, device='cuda')[:r]
        for _ in range(m * hkv)
    ]).view(m, hkv, r).to(torch.int64).contiguous()
    logits = torch.randn(
        m, hkv, r, generator=generator, dtype=torch.bfloat16,
        device='cuda').contiguous()
    # 同分、重复 token、显式无效项和尾块同时覆盖 quota/tie/补洞路径。
    logits[:, :, 128:512] = 0.25
    logits[:, :, 512:640] = float('-inf')
    cand[:, :, 640:768] = capacity - 1
    lower = torch.tensor([128, 0, 0], dtype=torch.int64, device='cuda')
    upper = torch.tensor([8000, 64, 4096], dtype=torch.int64, device='cuda')
    quota = torch.tensor([1024, 300, 0], dtype=torch.int64, device='cuda')
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
    packed = packed.contiguous()
    ntiles = triton.cdiv(r, 2048)
    partial = torch.empty(m, hkv, ntiles, width, dtype=torch.int32, device='cuda')
    ops._nfa_filtered_topk_partial_kernel[(m * hkv, ntiles)](
        packed, partial, r, TILE=2048, NTILES=ntiles,
        S=capacity, K=width, META_GROUP=hkv, num_warps=8)
    return partial, lower, upper, quota, capacity


def _graph_time(launch, inner=16, replays=100, reps=5):
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            launch()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / (inner * replays))
    return statistics.median(samples)


def main():
    torch.cuda.set_device(0)
    all_ok = True
    for seed in range(10):
        partial, lower, upper, quota, capacity = _make_partial(20261104 + seed)
        shape = (partial.shape[0], partial.shape[1], partial.shape[3])
        reference_out = torch.empty(shape, dtype=torch.int32, device='cuda')
        candidate_out = torch.empty_like(reference_out)
        reference_count = torch.empty(shape[:2], dtype=torch.int32, device='cuda')
        candidate_count = torch.empty_like(reference_count)
        _launch(_reference_merge_kernel, partial, lower, upper, quota,
                reference_out, reference_count, capacity)
        _launch(ops._nfa_filtered_topk_merge_kernel, partial, lower, upper, quota,
                candidate_out, candidate_count, capacity)
        torch.cuda.synchronize()
        ok = (torch.equal(reference_out, candidate_out)
              and torch.equal(reference_count, candidate_count))
        all_ok &= ok
        print(f'CASE seed={seed} bit_exact={ok}')
        if not ok:
            print('mismatched_indices', (reference_out != candidate_out).sum().item())
            print('mismatched_counts', (reference_count != candidate_count).sum().item())
            break

    if all_ok:
        # 正式性能只使用 B1 生产几何；正确性上面的 B3/quota 矩阵不计时。
        p1 = partial[:1].contiguous()
        lo1, hi1, q1 = lower[:1], upper[:1], quota[:1]
        ref_out = torch.empty(1, 8, 1024, dtype=torch.int32, device='cuda')
        new_out = torch.empty_like(ref_out)
        ref_count = torch.empty(1, 8, dtype=torch.int32, device='cuda')
        new_count = torch.empty_like(ref_count)
        reference = lambda: _launch(
            _reference_merge_kernel, p1, lo1, hi1, q1, ref_out, ref_count, capacity)
        candidate = lambda: _launch(
            ops._nfa_filtered_topk_merge_kernel, p1, lo1, hi1, q1,
            new_out, new_count, capacity)
        reference()
        candidate()
        torch.cuda.synchronize()
        eager_ref = do_bench(reference, warmup=25, rep=100)
        eager_new = do_bench(candidate, warmup=25, rep=100)
        graph_ref = _graph_time(reference)
        graph_new = _graph_time(candidate)
        graph_exact = (torch.equal(ref_out, new_out)
                       and torch.equal(ref_count, new_count))
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
