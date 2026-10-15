# -*- coding: utf-8 -*-
"""E139 #253 Step2 可行性探针：torch.topk tie 序 + 零补位序的等价性实证。

问题背景：选择链 L2 双池 topk 现为全宽位置切片 O(W) radix select（111.5ms
新瓶颈）。候选列化 = gather 候选列的 p 值（bit 完全相同，无 ULP）后对
C 列 topk（O(C)）。数学等价条件：
  ① p 值无 tie 交叉 k 边界时集合恒等（值相同直接得）；
  ② tie 时全宽 topk 与「位置排序候选」topk 的 tie 序须一致（都按低位置）；
  ③ k > 候选数时全宽选零值位置须按位置序——候选版需补位仿真。
本探针用受控 tie 密度数据实证 ②③。"""
import torch

DEV = "cuda:0"
torch.manual_seed(0)

def probe(W, C, k, tie_levels, rows=64, reps=5):
    """W 池宽 / C 候选数 / topk k / tie_levels 离散值级数（越小 tie 越密）"""
    set_mismatch = 0
    zero_mismatch = 0
    total = 0
    for r in range(reps):
        # 每行随机候选位置（sorted）+ 受控离散值（bf16 量化天然造 tie）
        pos = torch.rand(rows, W, device=DEV).argsort(dim=-1)[:, :C]
        pos, _ = pos.sort(dim=-1)                        # 位置序候选
        vals = (torch.randn(rows, C, device=DEV) * 8).round() / tie_levels
        vals = vals.to(torch.bfloat16)                   # bf16 量化 → 密集 tie
        # 全宽张量：非候选位置全 0（softmax 非候选 p≡0 语义）
        full = torch.zeros(rows, W, device=DEV, dtype=torch.bfloat16)
        full.scatter_(1, pos, vals)
        sel_full = torch.topk(full, k, dim=-1).indices
        # 候选列版：gather 后对 C 列 topk → 映射回位置
        cand_sel = torch.topk(vals, k, dim=-1).indices   # k ≤ C 档
        sel_cand = pos.gather(1, cand_sel)
        for i in range(rows):
            total += 1
            if set(sel_full[i].tolist()) != set(sel_cand[i].tolist()):
                set_mismatch += 1
        # ③ 零补位档：k > C —— 全宽选零应按位置序；候选版补位 = 池内
        # 非候选位置的前 (k-C) 个（位置序）
        k2 = min(k + C // 2, W)
        sel_full2 = torch.topk(full, k2, dim=-1).indices
        for i in range(rows):
            sf = set(sel_full2[i].tolist())
            sc = set(pos[i].tolist())                    # 全部候选入选
            # 补位：池内非候选位置按位置序取 k2 - C 个
            mask = torch.ones(W, dtype=torch.bool, device=DEV)
            mask[pos[i]] = False
            comp = mask.nonzero().squeeze(-1)[: k2 - C]
            sc |= set(comp.tolist())
            if sf != sc:
                zero_mismatch += 1
    return set_mismatch, zero_mismatch, total

print("=== Step2 tie 序探针（值 gather 无 ULP，只测 tie/补位序） ===")
for W, C, k in [(4096, 1024, 256), (32768, 8192, 512), (65536, 8192, 512)]:
    for tl in (64, 8, 2):   # tie 密度递增
        sm, zm, tot = probe(W, C, k, tl)
        print(f"W={W:6d} C={C:5d} k={k:4d} tie_lv={tl:3d}: "
              f"tie集合不匹配 {sm}/{tot} | 零补位不匹配 {zm}/{tot}")
print("=== DONE ===")
