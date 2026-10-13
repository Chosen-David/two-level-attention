#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131-S2 终验：优化前后 e2e mask 逐位对拍（同 seed 同输入，双环境跑同进程）。

两臂（cclustersim / cavgsim）各跑 N=24 步（覆盖一次块边界 far 续跑 +
一次 near 重建），每步 mask 落袋后逐位对比。正确性门：mask 逐位一致
（正确性门 abs 1e-9/rel 1e-12 的 mask 等价形式——mask 是 bool 无数值容差），
top-k 重叠率 = 一致位比例。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "exp", "bench"))

import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
# 【E131-S2 移植（2026-10-12）】inplace 合并后 fast.py 已退役，基类默认即
# fast path（TLI_E127_FAST env 反转：默认开，=0 归档逃生口）。
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402


def run_arm(arm, steps=24):
    """跑 N 步，返回每步 mask 的 clone 列表（GPU 张量转 CPU）。"""
    os.environ["TLI_E127_FAST"] = "1"
    torch.manual_seed(20261012)
    idx = TLIIndexer(make_args(arm))
    idx.layer_idx = 2
    idx.clear()
    dev = "cuda:0"
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    S = 65536
    # KV 一次性生成（两大臂省内存：k 直接在 GPU 上逐步 cat）
    g = torch.Generator(device=dev).manual_seed(777001)
    k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16, generator=g)
    qs = torch.randn(steps, 1, 1, H, D, device=dev, dtype=torch.bfloat16, generator=g)
    newks = torch.randn(steps, 1, 1, Hkv, D, device=dev, dtype=torch.bfloat16, generator=g)

    masks = []
    for i in range(steps):
        q = qs[i]
        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1
        index_dict = idx.prepare_index(k, cu)
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        mask = idx.compute_mask(q_ids, score_dict)
        masks.append(mask.cpu().clone())
        k = torch.cat([k, newks[i]], dim=1)
    torch.cuda.synchronize()
    return masks


def main():
    for arm in ("cclustersim", "cavgsim"):
        # 优化前
        os.environ["TLI_E131_GREEDY_V2"] = "0"
        os.environ["TLI_E131_TOKSCORE_TRITON"] = "0"
        m_pre = run_arm(arm)
        # 优化后
        os.environ["TLI_E131_GREEDY_V2"] = "1"
        os.environ["TLI_E131_TOKSCORE_TRITON"] = "1"
        m_post = run_arm(arm)
        tot = agree = 0
        for a, b in zip(m_pre, m_post):
            assert a.shape == b.shape, f"{arm}: shape {a.shape} vs {b.shape}"
            tot += a.numel()
            agree += int((a == b).sum())
        rate = agree / tot * 100
        step_bad = [i for i, (a, b) in enumerate(zip(m_pre, m_post)) if not torch.equal(a, b)]
        print(f"[{arm}] mask 逐位一致率 {agree}/{tot} = {rate:.4f}%  "
              f"不一致步: {step_bad if step_bad else '无'}", flush=True)


if __name__ == "__main__":
    main()
