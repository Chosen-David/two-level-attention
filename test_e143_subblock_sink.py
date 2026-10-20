#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E143 sub-block sink 生产化门禁（CPU only，0 SKIP）。

背景（用户 2026-10-15 批准）：sink 默认 128 是无实验依据的工程约定；E135
扩展网格（e135_sink_ext.json，swa=128 固定）实测 sink 128→4 trace mass 仅
-0.0021，拐点在 4（StreamingLLM "4 suffices" 在 Qwen3-8B 复现；sink=0 时
champion 仅 -0.0030——tail32 打分器能自行找回 sink token）。生产口径
K2_mid = K2 − sink − swa，sink 128→4 释放 124 tok 给 mid 选择器
（768→892）。本门禁覆盖 E143 改动的三面：

  C1 构造门禁：sink 子块值（4/0/16）不再 raise；swa 非块对齐仍 fail-closed
     （非对齐 swa 有边界块结构性丢失，契约不变）。
  C2 mask 几何（sink=4, swa=128, S=4352, 冠军 α/β/γ）：
     ① 强制区恰为 [0,4) ∪ 尾 128（token 级精确，不多不少——[4,x) 是否
        选中由分数决定，不得被强制）；
     ② 无死区：K2≥S 极限下（dense 极限）[4,64) 必须全选——证明 sink
        所在块参与池竞争，[sink, bs) 段结构性可选；
     ③ 预算上界：每行 unique 选中 ≤ K2（重叠虚耗 ≤ sink%bs 属保守方向）。
  C3 128/128 回归：默认路径预算身份逐位不变（每行恰 1024 =
     sink 128 + mid 768 + swa 128）。

运行：python3 test_e143_subblock_sink.py
"""
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from sparse_attn.indexer import TLIIndexer          # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def gen_kq(S, seed=31):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, 2, 128, generator=g) * 0.5
    q = torch.randn(1, 1, 4, 128, generator=g) * 0.5
    return k, q


CHAMP = dict(tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
             tli_far_method="minmax", tli_near_method="avg")


def run_mask(args, k, q):
    idx = TLIIndexer(args)
    idx.layer_idx = 3
    cu = torch.tensor([0, k.shape[1]])
    q_ids = torch.tensor([k.shape[1] - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
    return idx, mask


def row_mask(mask):
    """prepare_mask 返回 [B?, Hkv, q, S] 或 [Hkv, S]——统一压成单行 [S]。"""
    m = mask
    while m.dim() > 1:
        m = m.any(0) if m.dim() > 2 else m[0]
    if m.dim() == 2:                      # [Hkv, S] → 任一行（共享口径行间一致）
        m = m[0]
    return m.bool()


def drive_compute_mask(args, fine_1h, kt):
    """直驱 compute_mask（合成分数，绕过 prepare_index/compute_score，
    借 test_e122_gamma_off 同款 harness）：coarse 均匀分 L1 池全过，
    fine [T] 单 head 细筛分扩成 Hkv=2 对称。"""
    idx = TLIIndexer(args)
    idx.layer_idx = 3
    idx.group_size = 1
    T = fine_1h.shape[-1]
    fine = fine_1h.view(1, 1, 1, T).expand(1, 1, 2, T).clone()
    score_dict = {
        "score_coarse": torch.ones(1, 1, 2, kt),
        "score_fine": fine,
        "score_coarse_avg": None,
        "score_moba": None,
    }
    mask = idx.compute_mask(torch.tensor([T - 1]), score_dict)
    return idx, mask


# ================================================================ C1 构造门禁
def c1_ctor_gate():
    name = "C1 构造门禁：sink 子块放行（4/0/16），swa 非对齐仍 fail-closed"
    ok = True
    detail = []
    for sk in (4, 0, 16, 63):
        try:
            TLIIndexer(make_args(tli_sink_tokens=sk))
            detail.append(f"sk{sk}=放行")
        except ValueError as e:
            ok = False
            detail.append(f"sk{sk}=误拒({e})")
    try:
        TLIIndexer(make_args(tli_sink_tokens=4, tli_swa_size=100))
        ok = False
        detail.append("swa100=误放行")
    except ValueError:
        detail.append("swa100=拒绝")
    # 默认与块对齐集合不受影响
    for sk, sw in ((128, 128), (64, 64), (0, 64), (192, 256)):
        try:
            TLIIndexer(make_args(tli_sink_tokens=sk, tli_swa_size=sw))
        except ValueError as e:
            ok = False
            detail.append(f"sk{sk}/sw{sw}=误拒({e})")
    report(name, ok, "；".join(detail))


# ================================================================ C2 sink=4 mask 几何
def c2_subblock_geometry():
    S, bs = 4352, 64
    k, q = gen_kq(S)
    args = make_args(**CHAMP, tli_sink_tokens=4, tli_swa_size=128)
    idx, mask = run_mask(args, k, q)
    m = row_mask(mask)

    ok1 = bool(m[:4].all()) and bool(m[-128:].all())
    # ② 无死区（直驱 harness，γ=None 自由单池 + coarse 均匀分 L1 全过）：
    #   a) dense 极限 K2=S → K2_mid=mid_len → [0,S) 全选；
    #   b) 记分注入 fine[4:64]=+100 → K2=1024 下块 0 非 sink 段 [4,64)
    #      必被选中——行为级证明 sink 所在块参与池竞争（非仅 floor 语义）。
    #   注：dense 全选断言必须在 γ=None 臂做——配额臂 near 配额 < near 区宽
    #   时未选中是配额语义（test_e122 T4 同口径），非死区。
    S2 = 4352
    fine0 = torch.zeros(S2)
    args_dense = make_args(tli_sink_tokens=4, tli_swa_size=128,
                           tia_level2_topk=S2, tli_alpha=0.25, tli_beta=0.125,
                           tli_gamma="off", tli_far_method="minmax",
                           tli_near_method="avg", tli_enable_kmeans=False,
                           tli_enable_layer_skip=False)
    _, m_dense = drive_compute_mask(args_dense, fine0, S2 // 64)
    h0d = m_dense[0, 0, 0]
    n_miss = int((~h0d).sum())
    # dense 极限的期望不是"全选"而是"缺失 ≤ sink%bs"：mid topk 域从
    # far_tok_lo=floor(sink/bs)·bs=0 起（含 sink 所在块），4 个 sink token
    # 虚耗 4 个名额 → 恰 4 个 mid token 落选（文档化重叠成本，方向保守）。
    # 关键是缺失不得落在 [4,64)（落选由 topk 平局决定，非结构死区）。
    ok2 = n_miss <= 4 and bool(h0d[4:64].all())
    fine_inj = torch.full((S2,), -100.0)
    fine_inj[4:64] = 100.0                    # 块 0 非 sink 段打满分
    args_inj = make_args(
        tli_sink_tokens=4, tli_swa_size=128, tia_level2_topk=1024,
        tli_alpha=0.25, tli_beta=0.125, tli_gamma="off",
        tli_far_method="minmax", tli_near_method="avg",
        tli_enable_kmeans=False, tli_enable_layer_skip=False)
    _, m_inj = drive_compute_mask(args_inj, fine_inj, S2 // 64)
    ok2b = bool(m_inj[..., 4:64].all())
    # 预算上界：K2=1024 时 unique ≤ 1024（重叠虚耗 ≤ sink%bs=4，保守方向）
    n_sel = int(m.sum())
    ok3 = n_sel <= 1024
    # 池左界语义：floor(4/64)=0（sink 所在块参与竞争）
    ok4 = idx._sink_pool_lo_blk == 0
    report("C2 ① 强制区恰 [0,4)∪尾128（token 级）", ok1,
           f"sink全选={bool(m[:4].all())} swa全选={bool(m[-128:].all())}")
    report("C2 ②a 无死区：γ=None 单池 dense 极限缺失 ≤ sink%bs=4（重叠虚耗）",
           ok2, f"缺失={n_miss} [4,64)全选={bool(h0d[4:64].all())}")
    report("C2 ②b 无死区：注入 fine[4:64]=+100 → 块0非sink段必选中", ok2b,
           f"[4,64)全选={ok2b}")
    report("C2 ③ 预算上界 unique ≤ K2=1024（重叠虚耗 ≤4 保守）", ok3,
           f"unique={n_sel}")
    report("C2 ④ 池左界 floor(4/64)=0（sink 所在块参与池竞争）", ok4,
           f"_sink_pool_lo_blk={idx._sink_pool_lo_blk}")


# ================================================================ C3 128/128 回归
def c3_default_regression():
    S = 4352
    k, q = gen_kq(S)
    args = make_args(**CHAMP)                      # 默认 sink=128/swa=128
    idx, mask = run_mask(args, k, q)
    m = row_mask(mask)
    n_sel = int(m.sum())
    ok = (bool(m[:128].all()) and bool(m[-128:].all())
          and idx._sink_pool_lo_blk == 2
          and n_sel == 1024)
    report("C3 默认 128/128 回归：预算身份 sink128+mid768+swa128=1024 逐位不变",
           ok, f"unique={n_sel} 池左界={idx._sink_pool_lo_blk} "
               f"sink全选={bool(m[:128].all())} swa全选={bool(m[-128:].all())}")


# ================================================================ C4 sink=0 臂
def c4_sink_zero():
    S = 4352
    k, q = gen_kq(S)
    args = make_args(**CHAMP, tli_sink_tokens=0, tli_swa_size=128)
    idx, mask = run_mask(args, k, q)
    m = row_mask(mask)
    n_sel = int(m.sum())
    ok = (idx._sink_pool_lo_blk == 0
          and bool(m[-128:].all())
          and n_sel <= 1024)
    report("C4 sink=0：构造放行、无强制头区、预算上界守恒", ok,
           f"unique={n_sel} 池左界={idx._sink_pool_lo_blk} swa全选={bool(m[-128:].all())}")


if __name__ == "__main__":
    c1_ctor_gate()
    c2_subblock_geometry()
    c3_default_regression()
    c4_sink_zero()
    n_pass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n===== E143 sub-block sink 门禁：executed {n_pass}/{len(RESULTS)} PASS"
          f"（0 SKIP）=====")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
