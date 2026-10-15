# -*- coding: utf-8 -*-
"""E140 #253 Step2 tie 探针 v2：修正初版 k2 负数切片 bug，fail-closed。

TL-E139-TOPK-PROBE-IDENTITY-20261015 修复：初版 `k2 = min(k + C//2, W)`
在内置形状下恒 k2 < C → `[: k2-C]` 负切片取到「尾部」补位（应为前
k2-C 个位置序）；且 randn 负值 + 二进制幂缩放不构成受控 tie 密度。
修正版（本文件）：
  - pad 独立参数：k2 = C + pad，显式断言 0 < C < k2 <= W
  - 候选值严格正值：randint(1, tie_levels+1) 整数级数（tie 密度 =
    1/tie_levels，真正受控）
  - 零值档 / 全零档：候选 p 恰 0（bf16 下溢仿真）——已知边界（候选零
    与补位零位置序混合），预期 mismatch，如实报告不判失败（生产该档
    由 bitdiff 硬门把关，_zero_pad_pool docstring 已声明）
  - fail-closed：tie/补位两档任一 mismatch 非零 → sys.exit(1)
torch.topk 并列序非 API 契约（pytorch 2.8 文档明示）——本探针结论
限定当前派生/版本/shape，生产等价性以 e139_step2_cand_bitdiff
（mask 0/1008）硬门为准。
"""
import sys

import torch

DEV = "cuda:0"
torch.manual_seed(20261015)

FAIL = []


def probe(W, C, pad, tie_levels, rows=64, reps=5):
    """W 池宽 / C 候选数 / pad 补位数（k2 = C + pad）/ tie_levels 正值级数。"""
    assert 0 < C < C + pad <= W, f"几何非法: W={W} C={C} pad={pad}"
    k = C // 2                       # tie 档：k < C（候选内 topk）
    k2 = C + pad                     # 补位档：k2 > C
    tie_mis = pad_mis = zero_mis = 0
    total = 0
    mask = torch.ones(W, dtype=torch.bool, device=DEV)
    for _ in range(reps):
        pos = torch.rand(rows, W, device=DEV).argsort(dim=-1)[:, :C]
        pos, _ = pos.sort(dim=-1)
        vals = torch.randint(1, tie_levels + 1, (rows, C),
                             device=DEV).to(torch.bfloat16)   # 严格正值离散
        full = torch.zeros(rows, W, device=DEV, dtype=torch.bfloat16)
        full.scatter_(1, pos, vals)
        # ① tie 档：全宽 topk(k) vs 候选列 topk(k) 集合等价
        sel_full = torch.topk(full, k, dim=-1).indices
        sel_cand = pos.gather(1, torch.topk(vals, k, dim=-1).indices)
        # ② 补位档：全宽 topk(k2) vs 候选全选 + 非候选前 pad 个（位置序）
        sel_full2 = torch.topk(full, k2, dim=-1).indices
        # ③ 零值档（已知边界，预期 mismatch，如实报告）：候选含恰 0
        vals0 = vals.clone()
        vals0[:, : C // 4] = 0
        full0 = torch.zeros(rows, W, device=DEV, dtype=torch.bfloat16)
        full0.scatter_(1, pos, vals0)
        sel0 = torch.topk(full0, k2, dim=-1).indices
        for i in range(rows):
            total += 1
            if set(sel_full[i].tolist()) != set(sel_cand[i].tolist()):
                tie_mis += 1
            m1 = mask.clone()
            m1[pos[i]] = False
            nc = m1.nonzero().squeeze(-1)
            sc = set(pos[i].tolist()) | set(nc[:pad].tolist())
            if set(sel_full2[i].tolist()) != sc:
                pad_mis += 1
            # 候选零与非候选零位置序混合 → 全宽按全局位置序、候选版先
            # 候选零后补位——预期集合差（生产由 bitdiff 硬门把关）
            if set(sel0[i].tolist()) != sc:
                zero_mis += 1
    return tie_mis, pad_mis, zero_mis, total


print("=== Step2 tie 探针 v2（修正版，fail-closed）===")
print(f"torch {torch.__version__} / {torch.cuda.get_device_name(0)} "
      f"（topk 并列序非 API 契约，结论限定本派生/版本/shape）")
for W, C, pad in [(4096, 1024, 512), (32768, 8192, 1024), (65536, 8192, 1024)]:
    for tl in (2, 8, 64):      # tie 密度递减
        tm, pm, zm, tot = probe(W, C, pad, tl)
        print(f"W={W:6d} C={C:5d} pad={pad:5d} tie_lv={tl:3d}: "
              f"tie不匹配 {tm}/{tot} | 补位不匹配 {pm}/{tot} | "
              f"零值边界不匹配 {zm}/{tot}（已知边界，如实报告）")
        if tm != 0:
            FAIL.append(f"tie-{W}-{tl}")
        if pm != 0:
            FAIL.append(f"pad-{W}-{tl}")
if FAIL:
    print(f"=== FAIL: {FAIL} ===")
    sys.exit(1)
print("=== ALL PASS（tie/补位两档全零不匹配；零值档为已声明边界） ===")
