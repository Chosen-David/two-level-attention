#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E133：mavg 冠军 + tail32 子空间 mask 逐位对拍（用户 10-12 指令）。

E129 C2 已证 tail32 在 cclustersim base 上五任务 sha256 逐位 no-op（根因：
cluster+sim_greedy 双池选择不消费 L1 块池分数）。但论文主方法 mavg
(.25,.125,.625)（minmax+4bit）的 L1 块分数直接驱动 α/β/γ 分区预算——
tail32 是否无损未知，本脚本在同一进程内对拍两配置 24 步 mask：
  A = mavg_e127 原样（subspace='full', cmp_ratio=2，E127 锚点配置）
  B = mavg + tail32（subspace='tail', cmp_ratio=4，E129 C2 同款口径）
逐位一致 → 子空间采纳的质量零损失证明成立（mask 级，等价于 e2e 输出
一致——selection mask 是 indexer 对模型的全部作用面）。
"""
import os
import sys

REPO = "/home/wangyuanshuo02/two-level-attention-e111"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "exp", "bench"))

import argparse  # noqa: E402
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402


def mavg_e2e_anchor_args():
    """e2e 锚点口径（E123/E125 mavg）：cmp_ratio=4 + subspace=full。

    口径勘误（E133 v2）：cmp_ratio 与 subspace 独立生效——
      - subspace 只控制 L1 粗筛维度（full=128 / tail=32）
      - cmp_ratio 控制 L2 细筛 k_qat 宽度（64//r 尾段，full 下同样生效，
        _prepare_index_base L500-507 fallback）
    v1 的 A 臂（mavg_e127 = cmp2+full）L2=64 维，与 e2e 锚点（cmp4+full，
    L2=32 维）不同——v1 的 98.14% 差异混入 L2 宽度差，作废重跑。
    """
    ns = make_args("mavg_e127")
    ns.tia_level2_cmp_ratio = 4
    return ns


def mavg_tail_args():
    """E133 采纳臂：e2e 锚点口径 + tail32（单键差 = L1 粗筛 128→32 维）。"""
    ns = mavg_e2e_anchor_args()
    ns.tli_subspace = "tail"
    return ns


def run_arm(args, steps=24):
    os.environ["TLI_E127_FAST"] = "1"
    torch.manual_seed(20261012)
    idx = TLIIndexer(args)
    idx.layer_idx = 2
    idx.clear()
    dev = "cuda:0"
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    S = 65536
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
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    # v2：A = e2e 锚点口径（cmp4+full，L2=32）；B = +tail32（单键差 L1 128→32）
    mA = run_arm(mavg_e2e_anchor_args(), steps)
    mB = run_arm(mavg_tail_args(), steps)
    tot = agree = 0
    step_bad = []
    for i, (a, b) in enumerate(zip(mA, mB)):
        assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
        tot += a.numel()
        eq = int((a == b).sum())
        agree += eq
        if eq != a.numel():
            step_bad.append((i, a.numel() - eq))
    rate = agree / tot * 100
    print(f"[mavg cmp4+full vs cmp4+tail32] mask 逐位一致率 {agree}/{tot} = {rate:.4f}%  "
          f"不一致步: {step_bad if step_bad else '无'}")
    if rate == 100.0:
        print("[VERDICT] 逐位一致——tail32 对 mavg 冠军 mask 零信息量，质量零损失成立")
    else:
        print("[VERDICT] 存在差异步——须 e2e 5 任务质量验证后再定采纳")


if __name__ == "__main__":
    main()
