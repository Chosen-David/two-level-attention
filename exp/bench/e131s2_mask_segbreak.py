#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131-S2：cclustersim compute_mask 段子表达式分解计时（GPU0）。

profutor 的 profiler 有串行化放大，改用 cuda Event 对锚点 compute_mask
源码逐子表达式计时（固定稳态输入重放 N 次取中位）。cclustersim 的
use_partition 尾段（tli_indexer.py L1149-1372）关键子表达式：
  A) _far_token_score（einsum+gather）
  B) _near_token_score（einsum+gather）
  C) sf_raw→sf_g 全宽 fp32 转换 + group-mean（near_sc 底稿）
  D) near_sc 组装（clone + 簇分段回填）
  E) far topk + near topk + scatter
  F) softmax(p) + where 掩蔽（mask 段公共路径，mavg 亦有）
"""
import argparse
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "exp", "bench"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer_fast import TLIIndexerFast  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402


def med_time(fn, n=30):
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', type=str, default='cclustersim')
    p.add_argument('--S', type=int, default=65536)
    args = p.parse_args()

    torch.manual_seed(20261011 + 2)
    idx = TLIIndexerFast(make_args(args.arm))
    idx.layer_idx = 2
    idx.clear()
    dev = 'cuda:0'
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    k = torch.randn(1, args.S, Hkv, D, device=dev, dtype=torch.bfloat16)

    # 推进 20 步到稳态（含建簇）
    for _ in range(20):
        q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1
        index_dict = idx.prepare_index(k, cu)
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        mask = idx.compute_mask(q_ids, score_dict)
        k = torch.cat([k, torch.randn(1, 1, Hkv, D, device=dev, dtype=torch.bfloat16)], dim=1)
    torch.cuda.synchronize()

    # 固定一份稳态输入
    q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
    cu = prepare_cu_seqlens(k)
    q_ids = prepare_seqlens(cu) - 1
    index_dict = idx.prepare_index(k, cu)
    score_dict = idx.compute_score(q, q_ids, index_dict, scale)
    _ = idx.compute_mask(q_ids, score_dict)   # 预热一遍（内核选择/缓存）
    torch.cuda.synchronize()

    print(f"\n=== {args.arm} S={k.shape[1]} 稳态 compute_mask 子段（median of 30）===")
    t_total = med_time(lambda: idx.compute_mask(q_ids, score_dict))
    print(f"[total compute_mask]      {t_total:.4f} ms")

    q_last = idx._last_q
    # A) far token score
    t = med_time(lambda: idx._far_token_score(q_last))
    print(f"[A _far_token_score]      {t:.4f} ms  einsum[8,{idx._km_centroids.shape[1]}] + gather")
    # B) near token score
    t = med_time(lambda: idx._near_token_score(q_last))
    print(f"[B _near_token_score]     {t:.4f} ms  einsum[8,{idx._km_near_centroids.shape[1]}] + gather")

    # 消费端关键维度（照 compute_mask 锚点口径重算）
    bs = 64
    kt = index_dict["k_min"].shape[1]
    S_real = score_dict["valid_length"]
    swa_tok = idx.sliding_window_size
    far_lo_blk = idx.sink_blocks
    mid_len = max(0, S_real - far_lo_blk * bs - swa_tok)
    near_len_dyn = max(bs, int(idx.alpha * mid_len))
    near_blks = max(far_lo_blk, (S_real - swa_tok - near_len_dyn) // bs)
    far_tok_lo = far_lo_blk * bs
    far_tok_hi = min(near_blks * bs, S_real)
    swa_lo_tok = max(0, S_real - swa_tok)
    print(f"[dims] far_tok=({far_tok_lo},{far_tok_hi}) swa_lo_tok={swa_lo_tok} "
          f"kt={kt} S_real={S_real}")

    # C) sf_raw → sf_g 全宽转换 + group-mean（near_sc 底稿）
    p2 = None
    def make_sf_g(wide):
        sf_raw = score_dict["score_fine"][..., : (p2 or score_dict["score_fine"].shape[-1])].to(torch.float32) \
            if wide else score_dict["score_fine"][..., far_tok_hi:swa_lo_tok].to(torch.float32)
        return rearrange(sf_raw, "b qt (h g) kt -> b qt h g kt", g=idx.group_size).mean(dim=-2)
    t = med_time(lambda: make_sf_g(True))
    print(f"[C sf_g 全宽 fp32+mean]   {t:.4f} ms  [1,1,32,{score_dict['score_fine'].shape[-1]}]")
    t = med_time(lambda: make_sf_g(False))
    print(f"[C' sf_g 窄段版（改后）]   {t:.4f} ms  宽 {swa_lo_tok - far_tok_hi}")

    # D) near_sc 组装（窄段版）
    def near_sc_narrow():
        sf_g = make_sf_g(False)
        near_sc = sf_g.clone()
        cs_lo = max(idx._km_near_lo, far_tok_hi)
        cs_hi = min(idx._km_near_hi, swa_lo_tok)
        if cs_hi > cs_lo:
            near_tok_score = idx._near_token_score(q_last)
            seg = near_tok_score[..., cs_lo - idx._km_near_lo: cs_hi - idx._km_near_lo]
            near_sc[..., cs_lo - far_tok_hi: cs_hi - far_tok_hi] = seg.unsqueeze(0).unsqueeze(0)
        return near_sc
    t = med_time(lambda: near_sc_narrow())
    print(f"[D near_sc 窄段组装]      {t:.4f} ms")

    # E) topk far + topk near + scatter（照锚点预算口径）
    near_sc = near_sc_narrow()
    far_tok_score = idx._far_token_score(q_last)
    K2 = 1024
    K2_mid = max(0, K2 - far_tok_lo - (S_real - swa_lo_tok))
    k2_far = min(512, far_tok_score.shape[-1], K2_mid)
    k2_near = max(0, K2_mid - k2_far)
    def topk_both():
        i_f = (torch.topk(far_tok_score, k2_far, dim=-1).indices + idx._km_far_lo).unsqueeze(0).unsqueeze(0)
        i_n = torch.topk(near_sc, min(k2_near, near_sc.shape[-1]), dim=-1).indices + far_tok_hi
        m = torch.zeros(1, 1, Hkv, S_real, dtype=torch.bool, device=dev)
        m.scatter_(-1, i_f, True)
        m.scatter_(-1, i_n, True)
        return m
    t = med_time(lambda: topk_both())
    print(f"[E topk far+near+scatter] {t:.4f} ms  k2_far={k2_far} k2_near={k2_near}")


if __name__ == '__main__':
    main()
