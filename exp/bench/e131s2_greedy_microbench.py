#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131-S2：greedy cluster pass 瓶颈分解 microbench（GPU0 独占开发卡）。

目的：确认阶段 1 报告冷建 8.4s（cclustersim）/ 11.2s（cavgsim）的真实构成——
E113b Triton kernel 已存在（grid=(H,) 单 program 一头），怀疑瓶颈是：
  a) randn 最坏数据下 k_live 涨满 → kernel 内 O(T·k_live) 簇状态扫描；
  b) 8 program / 78 SM 低占用。
分解口径：
  1) far 侧全量贪心（T≈64K，cavgsim/cclustersim 冷建主项）
  2) near 侧全量贪心（Tn≈8K，cclustersim 重建 spike 主项）
  3) greedy kernel 单独计时（Triton 路径 vs SGLANG_TLI_GREEDY_KERNEL=0 Python 路径）
  4) BT/num_warps 调参扫描（大 k_live 最坏口径，E113 扫描只扫过 K≈2K structured）
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402


def timed(fn, n=1):
    """cuda Event 计时，n 次取中位（GPU 时间线口径）。"""
    ts = []
    for _ in range(n):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--S', type=int, default=65536)
    p.add_argument('--seed', type=int, default=20261011)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = 'cuda:0'
    Hkv, D = 8, 128
    k = torch.randn(1, args.S, Hkv, D, device=dev, dtype=torch.bfloat16)

    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.indexer.greedy_triton import greedy_pass_triton, greedy_build_triton

    # cclustersim 口径：sim_dims=subspace → tail32（cmp_ratio=4 → dd=32）
    bs = 64
    sink_tok = 2 * bs
    swa_tok = 2 * bs
    alpha = 0.125
    mid_len = args.S - sink_tok - swa_tok
    near_len_dyn = max(bs, int(alpha * mid_len))
    near_base = args.S - swa_tok
    far_hi_blk = max(3, (near_base - near_len_dyn) // bs)
    far_hi = far_hi_blk * bs
    near_hi = max(far_hi, (near_base) // bs * bs)
    Tfar, Tn = far_hi - sink_tok, near_hi - far_hi
    print(f"[shape] S={args.S} far区=({sink_tok},{far_hi}) Tfar={Tfar} "
          f"near区=({far_hi},{near_hi}) Tn={Tn}", flush=True)

    dd = 32
    x_far = k[0, sink_tok:far_hi].float()[..., 96:128].contiguous()   # [Tfar,Hkv,32]
    x_near = k[0, far_hi:near_hi].float()[..., 96:128].contiguous()   # [Tn,Hkv,32]
    print(f"[data] x_far {tuple(x_far.shape)} dd={dd}", flush=True)

    # ---- 1) far 全量贪心（chunk 16K 分段，生产路径同款）----
    t = timed(lambda: greedy_build_triton(x_far, sim=0.9), n=1)
    print(f"[greedy_build_triton far] T={Tfar} → {t:.1f} ms "
          f"({t / Tfar * 1e3:.2f} us/token)", flush=True)

    # 活簇数统计（randn 最坏 → 应接近 T）
    sums, cnt, sq, kl, assign = greedy_build_triton(x_far, sim=0.9)
    print(f"[live_clusters far] k_live per head = {kl.tolist()}", flush=True)

    # ---- 2) near 全量贪心（重建 spike 主项）----
    t = timed(lambda: greedy_build_triton(x_near, sim=0.9), n=1)
    print(f"[greedy_build_triton near] T={Tn} → {t:.1f} ms "
          f"({t / Tn * 1e3:.2f} us/token)", flush=True)

    # ---- 3) BT/num_warps 调参扫描（大 k_live 最坏口径）----
    # 先建一个 k_live≈Tfar/2 的状态（固定大扫描量），再测 8192 token 续跑
    st = greedy_build_triton(x_far[:Tfar // 2], 0.9)  # (sums,cnt,sq,kl,assign)
    for BT in (512, 1024, 2048, 4096):
        for nw in (4, 8, 16):
            try:
                def run():
                    # 深拷贝状态防原地污染跨次计时（sums/cnt/sq 原地更新）
                    s2, c2, q2 = st[0].clone(), st[1].clone(), st[2].clone()
                    kl2 = st[3].clone()
                    greedy_pass_triton(x_far[Tfar // 2:Tfar // 2 + 8192], 0.9,
                                       s2, c2, q2, kl2, BT=BT, num_warps=nw)
                t = timed(run, n=1)
                print(f"[BT scan] BT={BT} nw={nw}: {t:.1f} ms "
                      f"({t / 8192 * 1e3:.2f} us/token, k_live≈{Tfar // 2})", flush=True)
            except Exception as e:
                print(f"[BT scan] BT={BT} nw={nw}: FAIL {type(e).__name__}: {e}",
                      flush=True)

    # ---- 4) Python 路径对照（确认 Triton 路径确实生效且快）----
    os.environ["SGLANG_TLI_GREEDY_KERNEL"] = "0"
    x_small = x_far[:2048]
    t0 = time.perf_counter()
    TLIIndexer._greedy_cluster_pass_python(
        x_small, 0.9,
        *([torch.zeros(8, 2048, 32, device=dev)] * 0 or
          [x_small.new_zeros(8, 2048, 32), x_small.new_zeros(8, 2048),
           x_small.new_zeros(8, 2048), torch.zeros(8, dtype=torch.long, device=dev)]))
    torch.cuda.synchronize()
    print(f"[python ref] T=2048 → {(time.perf_counter() - t0) * 1e3:.1f} ms "
          f"({(time.perf_counter() - t0) * 1e3 / 2048 * 1e3:.2f} us/token)",
          flush=True)


if __name__ == '__main__':
    main()
