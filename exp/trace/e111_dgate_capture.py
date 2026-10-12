#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E111 D' gate 短路 patch 验证 harness（capture / compare 双模式 + GPT 0739
审计 2026-10-12 D' 三条验收扩展版）。

用法（各自从目标树根运行，sparse_attn 取 CWD 树）：
  python3 /tmp/e111_dgate_capture.py capture /tmp/e111_baseline.pt   # patch 前旧树
  python3 /tmp/e111_dgate_capture.py compare /tmp/e111_baseline.pt   # patch 后新树

验证分四类：
  A. 逐位类：patch 前后 mask 逐位相等——非 skip 全配置矩阵 +
     skip+(0,0) legacy 掉落（非 e64 组合 patch 不得改变 legacy 行为）。
  B. 结构断言类（skip+e64 = 设计内行为变更，用户 2026-10-12 拍板）：
     far 全 False / sink+swa 强制 / near 拿满 K2_mid / 总预算恒等。
     覆盖 GPT 要求：mavg、aavg、near-cluster（回退语义）、尾页
     （S%bs=1）、短序列、候选饥饿。
  C. 省算证据类（GPT 验收 1「仅改 mask 不算通过」）：compute_score 返回
     的 score_dict 在 compute_mask 之前 far 段已全 -inf——省算分支真实
     未计算 far 段（未省算时 far 段细筛分是实值）。含 fallback 捷径
     排除路径（mminmax+full 组合）。
  D. near-only 独立参考对拍（GPT 验收 2）：独立代码路径（不走
     use_partition 分支）从 score_dict 复算 sink + near topk + swa，
     与 indexer.compute_mask 输出逐位对比。
"""
import argparse
import importlib
import os
import sys
import types

import torch

sys.path.insert(0, os.getcwd())
IDX_PKG = importlib.import_module("sparse_attn.indexer")

BS = 64
SINK_TOK = 128
SWA_TOK = 128
K1 = 128
K2 = 1024

CONFIGS = {
    # ---------- A. 逐位类：非 skip 全路径矩阵 ----------
    "mavg_e64": dict(S=8192, skip=False, kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    "aavg_e64": dict(S=8192, skip=False, kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="avg",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    "ccluster_e64": dict(S=8192, skip=False, kw=dict(
        tli_far_select="cluster", tli_near_select="cluster",
        tli_enable_kmeans=True, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.125, tli_beta=0.25,
        tli_gamma=0.125)),
    "pool00_4bit": dict(S=4097, skip=False, kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.0, tli_beta=0.0)),
    "sigma_far_e64": dict(S=4097, skip=False, kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625, tli_sigma_select="far", tli_sigma=8.0)),
    "perqh_e64": dict(S=8192, skip=False, kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625, tli_per_q_head=True)),
    # ---------- A. 逐位类：skip + 非 e64（patch 后须逐位不变）----------
    "cluster00_skip": dict(S=4097, skip=True, kw=dict(
        tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.0, tli_beta=0.0)),
    "pool00_4bit_skip": dict(S=4097, skip=True, kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.0, tli_beta=0.0)),
    # ---------- B. 结构断言类：skip + e64（设计内行为变更）----------
    "mavg_e64_skip": dict(S=8192, skip=True, structural=(0.25, 0.125),
                          kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    "mavg_e64_skip_b1cfg": dict(S=4352, skip=True, structural=(0.125, 0.375),
                                kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.125, tli_beta=0.375,
        tli_gamma=0.625)),
    "aavg_e64_skip": dict(S=8192, skip=True, structural=(0.25, 0.125),
                          kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="avg",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    # near cluster 臂 + skip：簇构建被 E72 skip 门挡 → near_tok_score=None
    # → near 池回退细筛分（显式登记的回退语义，细筛分质量 ≥ 簇代表分）
    "ncluster_e64_skip": dict(S=8192, skip=True, structural=(0.25, 0.125),
                              kw=dict(
        tli_far_select="4bit", tli_near_select="cluster",
        tli_enable_kmeans=True, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    # fallback 捷径排除路径：full 子空间 + 双池不用 avg（mminmax 型）——
    # 旧版走父类捷径全量算；新版被 skip 条件排除 → 本类路径省算
    "mminmax_fallback_skip": dict(S=8192, skip=True, structural=(0.25, 0.125),
                                  kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="minmax", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    # 尾页（S%bs=1）
    "mavg_e64_skip_tail": dict(S=4097, skip=True, structural=(0.25, 0.125),
                              kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    # 短序列（mid 不足、near 区仅 2 块）
    "mavg_e64_skip_short": dict(S=640, skip=True, structural=(0.25, 0.125),
                                kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
        tli_gamma=0.625)),
    # 候选饥饿（α=1/16 → near 区 512 < K2_mid 768，池内截断不回补 far）
    "mavg_e64_skip_starve": dict(S=8192, skip=True, structural=(0.0625, 0.125),
                                 kw=dict(
        tli_far_select="4bit", tli_near_select="4bit",
        tli_enable_kmeans=False, tli_far_method="minmax",
        tli_near_method="avg", tli_alpha=0.0625, tli_beta=0.125,
        tli_gamma=0.625)),
}


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=BS, tia_level1_topk=K1, tia_level2_topk=K2,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def gen_kq(S, seed=31):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, 2, 128, generator=g) * 0.5
    q = torch.randn(1, 1, 4, 128, generator=g) * 0.5
    return k, q


def geom(S, alpha):
    mid_len = max(0, S - SINK_TOK - SWA_TOK)
    near_len_dyn = max(BS, int(alpha * mid_len))
    near_blks = max(2, (S - SWA_TOK - near_len_dyn) // BS)
    return near_blks, S - SWA_TOK   # near_blks, swa_lo_tok


def run_stages(kw, S, skip):
    """三阶段分离跑（拿 score_dict 做省算证据 + near-only 参考）。

    skip 开启走生产语义（081 snapshot skip_ids）：设 _skip_ids 后由
    prepare_index 内 _resolve_skip 解析——手动设 idx.skip_far 会被
    _resolve_skip 在 prepare_index 入口重置（曾致 harness 全程 non-skip
    的假 skip 验证，2026-10-12 排查）。
    """
    idx = IDX_PKG.TLIIndexer(make_args(**kw))
    idx.layer_idx = 3
    if skip:
        idx._skip_ids = {3}
    k, q = gen_kq(S)
    cu = torch.tensor([0, S])
    q_ids = torch.tensor([S - 1])
    scale = q.shape[-1] ** -0.5
    index_dict = idx.prepare_index(k, cu)
    score_dict = idx.compute_score(q, q_ids, index_dict, scale)
    mask = idx.compute_mask(q_ids, score_dict)
    return idx, k, q, cu, q_ids, scale, index_dict, score_dict, mask


def run_mask(kw, S, skip):
    *_, mask = run_stages(kw, S, skip)
    return mask


def structural_check(name, mask, S, alpha, beta):
    """skip+e64 新语义结构断言（B 类）+ far 不可补位（GPT 验收 3）。"""
    near_blks, swa_lo_tok = geom(S, alpha)
    far_tok_lo, far_tok_hi = SINK_TOK, near_blks * BS
    m = mask[0, 0]                       # [Hkv, S]（per_q_head 不在结构类）
    # sink / swa 强制
    assert bool(m[..., :SINK_TOK].all()), f"{name}: sink 强制区缺失"
    assert bool(m[..., swa_lo_tok:].all()), f"{name}: swa 强制区缺失"
    # far 全 False：零分补位防线——far token 结构性不可选（GPT 第 3 条）
    assert not bool(m[..., far_tok_lo:far_tok_hi].any()), \
        f"{name}: skip 层 far 段应全 False（p=0 补位防线失效）"
    # near 池 = L1 near 页（nb_near=β·K1）× BS，与 near 区宽取小；
    # 选中 = min(K2_mid, near_pool)——饥饿不回补 far（C-1 语义）
    K2_mid = K2 - SINK_TOK - SWA_TOK
    nb_near = max(1, int(round(K1 * beta)))
    near_pool = min(nb_near * BS, swa_lo_tok - far_tok_hi)
    expect_near = min(K2_mid, max(0, near_pool))
    per_head = m[..., far_tok_hi:swa_lo_tok].sum(dim=-1)
    assert bool((per_head == expect_near).all()), (
        f"{name}: near 选中应 = {expect_near}（K2_mid={K2_mid}, "
        f"pool={near_pool}），实际 {per_head.tolist()}")
    total = int(m.sum(dim=-1)[0])
    expect_total = SINK_TOK + expect_near + SWA_TOK
    assert total == expect_total, \
        f"{name}: 总选中应 = {expect_total}，实际 {total}"
    # far 段因 pad 展宽时（S=4097 → kt=65 块）padding 段不得混入
    if m.shape[-1] > S:
        assert not bool(m[..., S:].any()), f"{name}: padding 段混入选中"
    print(f"[e111-B] {name}: far=0 near={expect_near} "
          f"total={expect_total} ✓", flush=True)


def shortcircuit_evidence(name, score_dict, S, alpha):
    """C 类省算证据：score_dict 的 far 段在 compute_mask 之前已全 -inf。

    未省算时 far 段细筛分是量化点积实值（|值| > 0 概率 ~1，随机数据下
    全 -inf 不可能自然出现）→ far 段 -inf 即省算分支真实生效证据。
    """
    near_blks, _ = geom(S, alpha)
    far_tok_lo, far_tok_hi = SINK_TOK, near_blks * BS
    sc = score_dict["score_coarse"][0, 0]          # [H, kt]
    sf = score_dict["score_fine"][0, 0]           # [H, T]
    sc_far = sc[..., 2:near_blks]
    sf_far = sf[..., far_tok_lo:far_tok_hi]
    assert bool((sc_far == float("-inf")).all()), \
        f"{name}: score_coarse far 块段未省算（存在实值）"
    assert bool((sf_far == float("-inf")).all()), (
        f"{name}: score_fine far token 段未省算（存在实值）——仅改 mask "
        f"不算通过（GPT 验收 1）")
    # 保留段必须有实值（省算没有把 near/swa 段也算没）
    sf_keep = sf[..., far_tok_hi:]
    assert bool((sf_keep > float("-inf")).any()), \
        f"{name}: score_fine 保留段全 -inf（省算越界）"
    print(f"[e111-C] {name}: far 段 score_coarse/score_fine 全 -inf "
          f"（评分入口省算证据）✓", flush=True)


def near_only_reference(name, idx, k, q, cu, q_ids, scale, index_dict,
                        score_dict, mask, S, alpha, beta):
    """D 类：独立 near-only 参考实现（不复用 compute_mask 分区分支），
    从 score_dict 出发复算期望 mask，与 indexer 输出逐位对比。"""
    from einops import rearrange, repeat
    near_blks, swa_lo_tok = geom(S, alpha)
    far_tok_lo, far_tok_hi = SINK_TOK, near_blks * BS
    Hkv = mask.shape[2] if mask.dim() == 4 else None
    group = idx.group_size
    # L1 near 池：near 段块分数 topk nb_near 页（near_method 分数源）
    nb_near = max(1, int(round(K1 * beta)))
    sc = score_dict["score_coarse"]
    sc_avg = score_dict.get("score_coarse_avg")
    if sc_avg is not None:
        sc_avg = rearrange(sc_avg, "b qt (h g) kt -> b qt h g kt",
                           g=group).mean(dim=-2)
    sc = rearrange(sc, "b qt (h g) kt -> b qt h g kt", g=group).mean(dim=-2)
    sc_near_src = sc_avg if (idx.near_method == "avg"
                             and sc_avg is not None) else sc
    # L1 near 池段 = [near_blks, swa_lo_blk)：swa 块完全排除出双池
    # （indexer L1079 口径：纯靠 mask 强制，不占 near 池名额）
    kt = sc.shape[-1]
    swa_lo_blk = max(near_blks, kt - max(1, SWA_TOK // BS))
    near_blk_seg = sc_near_src[..., near_blks:swa_lo_blk]
    k1n = min(nb_near, near_blk_seg.shape[-1])
    i_n_blk = torch.topk(near_blk_seg, k1n, dim=-1).indices + near_blks
    # L1 门控 + 细筛 + 每 head softmax + 组内 mean（indexer L1151-1154
    # 顺序：先 per-head softmax 再 group-mean——mean 与 softmax 不可交换，
    # 参考必须同序）
    l1 = torch.zeros_like(sc, dtype=torch.bool).scatter_(
        -1, i_n_blk, torch.ones_like(i_n_blk, dtype=torch.bool))
    # 当前块强制（TIA 语义：最后一块总在池内——near 段含当前块当且仅当
    # near 区覆盖到序列尾前一块；swa 块被强制区兜住，参考不强加）
    sf = score_dict["score_fine"]
    l1_h = repeat(l1, "b qt h kt -> b qt (h g) kt", g=group)
    l1_tok = l1_h.repeat_interleave(BS, dim=-1)[..., :sf.shape[-1]]
    p_h = torch.softmax(
        torch.where(l1_tok, sf, torch.tensor(float("-inf"))), dim=-1
    ).nan_to_num(0)
    p_h[..., max(0, S - SWA_TOK):S] = 1.0
    p = rearrange(p_h, "b qt (h g) kt -> b qt h g kt", g=group).mean(dim=-2)
    # L2：near 段 [far_tok_hi, swa_lo_tok) topk min(K2_mid, 段宽)
    # ——参考用「段内全候选」（不受 L1 页数截断的宽上界）≠ indexer 实际
    # 候选（L1 门控后 p=0 的不可选）——topk 结果两者一致当且仅当
    # K2_mid ≤ 有效候选；饥饿时 indexer 池宽也截到段宽，一致。
    # 故参考取段宽上界，若与 indexer 有差异即 L1 门控差异证据。
    near_p = p[..., far_tok_hi:swa_lo_tok]
    K2_mid = K2 - SINK_TOK - SWA_TOK
    k2n = min(K2_mid, near_p.shape[-1])
    i_n = torch.topk(near_p, k2n, dim=-1).indices + far_tok_hi
    m_ref = torch.zeros_like(p, dtype=torch.bool)
    m_ref.scatter_(-1, i_n, True)
    m_ref[..., :SINK_TOK] = True
    m_ref[..., max(0, S - SWA_TOK):S] = True
    m_cur = mask[0]                  # [1, Hkv, T]
    ok = torch.equal(m_cur, m_ref[0])   # m_ref[0] 同为 [1, Hkv, T]
    if not ok:
        diff = int((m_cur != m_ref[0]).sum())
        raise AssertionError(
            f"{name}: near-only 参考对拍不一致 diff={diff}——"
            f"skip 分支输出 ≠ 显式 near-only 语义（GPT 验收 2）")
    print(f"[e111-D] {name}: near-only 参考逐位一致 ✓", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["capture", "compare"])
    p.add_argument("path")
    args = p.parse_args()

    results = {}
    for name, spec in CONFIGS.items():
        mask = run_mask(spec["kw"], spec["S"], spec["skip"])
        results[name] = mask.cpu()
        print(f"[e111] {name}: shape={list(mask.shape)} "
              f"sel/head={int(mask.sum(dim=-1).flatten()[0])}", flush=True)

    if args.mode == "capture":
        # B/C/D 类只在 compare（patch 后新树）跑——旧实现 skip+e64 落
        # legacy（far 段有 p=0 补位选中、score_fine far 段有实值），
        # 新语义断言在旧树必然失败。
        torch.save(results, args.path)
        print(f"[e111] 基线已存 {args.path}（结构/证据/参考类只存不比较）",
              flush=True)
        return

    # ---------- B/C/D：新树全量验证 ----------
    for name, spec in CONFIGS.items():
        if "structural" not in spec:
            continue
        alpha, beta = spec["structural"]
        (idx, k, q, cu, q_ids, scale, index_dict, score_dict, mask) = \
            run_stages(spec["kw"], spec["S"], True)
        structural_check(name, mask.cpu(), spec["S"], alpha, beta)
        shortcircuit_evidence(name, score_dict, spec["S"], alpha)
        near_only_reference(name, idx, k, q, cu, q_ids, scale, index_dict,
                            score_dict, mask, spec["S"], alpha, beta)
    print("[e111] skip+e64 新语义 B/C/D 全过（结构断言 / 省算证据 / "
          "near-only 参考）", flush=True)

    # ---------- A：跨版本逐位 ----------
    base = torch.load(args.path)
    n_fail = 0
    for name, cur in results.items():
        if "structural" in CONFIGS[name]:
            continue   # 设计内行为变更，不做跨版本逐位比较
        old = base[name]
        if not torch.equal(old, cur):
            n_fail += 1
            print(f"[FAIL] {name}: mask 逐位不一致 "
                  f"(diff={int((old != cur).sum())})", flush=True)
        else:
            print(f"[PASS] {name}: 逐位一致", flush=True)
    if n_fail:
        raise SystemExit(f"[GATE-FAIL] {n_fail} 配置逐位不一致——"
                         "非 skip 层 / skip+(0,0) legacy 不得有任何变化")
    print("[e111] compare 全过：逐位类逐位一致 + 结构/证据/参考类通过",
          flush=True)


if __name__ == "__main__":
    main()
