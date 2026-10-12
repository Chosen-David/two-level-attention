#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131-S2：cclustersim compute_mask 段 kernel 分解 profile（GPU0）。

目的：阶段 1 定位 cclustersim mask 段 +0.257ms vs mavg（.251 口径），
确认段内时间构成（near_sc 组装 / far_tok_score einsum+gather / topk /
cat·scatter 胶水），指导融合方向。
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "exp", "bench"))

import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer_fast import TLIIndexerFast  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', type=str, default='cclustersim')
    p.add_argument('--S', type=int, default=65536)
    p.add_argument('--steps', type=int, default=12)
    args = p.parse_args()

    torch.manual_seed(20261011 + 2)
    idx = TLIIndexerFast(make_args(args.arm))
    idx.layer_idx = 2
    idx.clear()
    dev = 'cuda:0'
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    k = torch.randn(1, args.S, Hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, args.S, Hkv, D, device=dev, dtype=torch.bfloat16)

    from torch.profiler import profile, ProfilerActivity

    def step_once(k, v):
        q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1
        index_dict = idx.prepare_index(k, cu)
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        mask = idx.compute_mask(q_ids, score_dict)
        idx.metrics.add_select_result(q_ids, mask, 1)
        new_k = torch.randn(1, 1, Hkv, D, device=dev, dtype=torch.bfloat16)
        new_v = torch.randn(1, 1, Hkv, D, device=dev, dtype=torch.bfloat16)
        k2 = torch.cat([k, new_k], dim=1)
        v2 = torch.cat([v, new_v], dim=1)
        return k2, v2

    WARMUP = 10   # 预热在 profile 窗口外（含首步冷建簇）
    for _ in range(WARMUP):
        k, v = step_once(k, v)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for step in range(args.steps):
            k, v = step_once(k, v)
            torch.cuda.synchronize()

    # 聚合 CUDA kernel 时间
    agg = {}
    for e in prof.key_averages():
        if e.device_type == torch.profiler.DeviceType.CUDA or e.self_device_time_total > 0:
            agg[e.key] = (e.self_device_time_total, e.count)
    tot = sum(t for t, _ in agg.values())
    print(f"\n=== {args.arm} 全管线 CUDA kernel 总 {tot/1e3:.3f} ms/步"
          f"（{args.steps} 步平均）===")
    for key, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:24]:
        print(f"  {t/1e3:8.4f} ms/步  ×{c:3d}  {key[:100]}")


if __name__ == '__main__':
    main()
