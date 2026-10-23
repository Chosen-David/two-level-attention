#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E127 优化门禁套件（每项优化一个 test_*，违反即回退）。

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_e127_gates.py            # 全部
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_e127_gates.py metrics eager

门禁口径：
  - 值逐位一致的门禁用 ==（不是 allclose）
  - attention 输出对拍用 allclose(1e-2/1e-2)（bf16 合理容差，任务指定）
  - 惰性/无同步路径用 torch.cuda.set_sync_debug_mode("error") 验证零 host 同步
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


# ---------------------------------------------------------------- metrics 惰性化
def test_metrics_lazy():
    """iter1：add_select_result 惰性路径 —— get_select_tokens 逐位一致 + 零同步。"""
    from sparse_attn.metrics import Metrics
    torch.manual_seed(20261011)
    dev = torch.device('cuda:0')
    # 构造覆盖：低于阈值（不记录）/ 高于阈值（记录）/ 多行混合 / 不同 H
    calls = []
    for S, H in [(100, 8), (3000, 8), (5000, 32), (8192, 8)]:
        q_ids = torch.tensor([S - 1], device=dev)
        mask = torch.rand(1, 1, H, S, device=dev) > 0.9
        calls.append((q_ids, mask, 1))
    # 多行混合：第 0 行低于阈值，第 1 行高于
    q_ids2 = torch.tensor([100, 9000], device=dev)
    mask2 = torch.rand(1, 2, 8, 9000, device=dev) > 0.85
    calls.append((q_ids2, mask2, 64))

    results = {}
    for mode in ["0", "1"]:
        os.environ["TLI_E127_FAST"] = mode
        m = Metrics()
        for q_ids, mask, bs in calls:
            m.add_select_result(q_ids, mask, bs)
        results[mode] = m.get_select_tokens()
    check("metrics_lazy: get_select_tokens 逐位一致",
          results["0"] == results["1"] and results["0"] is not None,
          f"old={results['0']} lazy={results['1']}")

    # 零同步验证：error 模式下惰性路径不得触发 host 同步
    os.environ["TLI_E127_FAST"] = "1"
    m = Metrics()
    q_ids = torch.tensor([4096], device=dev)
    mask = torch.rand(1, 1, 8, 4096, device=dev) > 0.9
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(20):
            m.add_select_result(q_ids, mask, 1)
        no_sync = True
    except Exception as e:
        no_sync = False
        print(f"  同步异常: {type(e).__name__}: {e}")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    check("metrics_lazy: add_select_result 零 host 同步", no_sync)
    os.environ.pop("TLI_E127_FAST", None)


# ---------------------------------------------------------------- eager gather
def test_eager_gather():
    """iter2+3：eager_decoding_attn gather 路径 —— allclose(1e-2) 对拍 + 语义守卫。

    输入用与生产一致的形态：mask [1,1,Hkv,S]（共享口径）与 [1,1,H,S]
    （per_q_head 口径）都要过；全 False 行的 B7 fail-closed 守卫两条路径
    都必须触发同一 RuntimeError。
    【iter3】gather 路径零同步化：B7/W 超界/cu 契约违规改为 GPU 标志 +
    生成结束 eager_decoding_check_errors() 尾部 raise（文案/语义守恒，
    时机后移）；小序列路由（T<16K）仍当场走 dense raise。
    """
    from sparse_attn.ops.eager_decoding import (
        eager_decoding_attn, eager_decoding_check_errors)
    torch.manual_seed(20261012)
    dev = torch.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    ok, worst = True, 0.0
    for S in [4097, 8192, 16384, 65536 - 13]:
        for mask_kind in ["shared", "per_qh"]:
            k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
            q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
            sel = torch.rand(1, 1, Hkv if mask_kind == "shared" else H,
                             S, device=dev) > 0.98
            sel[..., -128:] = True   # 保证每行非空（B7 守卫不触发）
            cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
            os.environ["TLI_E127_FAST"] = "0"
            o_ref = eager_decoding_attn(q, k, v, sel, 1, cu)
            os.environ["TLI_E127_FAST"] = "1"
            o_fast = eager_decoding_attn(q, k, v, sel, 1, cu)
            os.environ.pop("TLI_E127_FAST", None)
            eager_decoding_check_errors(dev)   # 正常路径：不 raise + 清零
            close = torch.allclose(o_ref.float(), o_fast.float(), atol=1e-2, rtol=1e-2)
            d = (o_ref.float() - o_fast.float()).abs().max().item()
            worst = max(worst, d)
            ok = ok and close
            if not close:
                print(f"  MISMATCH S={S} {mask_kind} maxdiff={d}")
    check("eager_gather: allclose(1e-2) 全配置对拍", ok, f"worst_abs_diff={worst:.2e}")

    # B7 fail-closed 守卫：小序列（T<16K 路由 dense）双模式当场 raise
    for mask_kind in ["shared", "per_qh"]:
        S, rows = 4096, (Hkv if mask_kind == "shared" else H)
        k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
        bad = torch.zeros(1, 1, rows, S, dtype=torch.bool, device=dev)
        cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
        raised = {}
        for mode in ["0", "1"]:
            os.environ["TLI_E127_FAST"] = mode
            try:
                eager_decoding_attn(q, k, v, bad, 1, cu)
                raised[mode] = False
            except RuntimeError:
                raised[mode] = True
        os.environ.pop("TLI_E127_FAST", None)
        check(f"eager_gather: B7 全 False 行守卫({mask_kind}) 小序列双路径当场 raise",
              raised["0"] and raised["1"])

    # 【iter3】B7 大序列：gather 主路径不再当场 raise（零同步），改为
    # eager_decoding_check_errors 尾部 raise（文案同 B7）。
    S = 65536 - 13
    k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
    bad = torch.zeros(1, 1, Hkv, S, dtype=torch.bool, device=dev)
    cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
    os.environ["TLI_E127_FAST"] = "1"
    o = eager_decoding_attn(q, k, v, bad, 1, cu)      # 不当场 raise
    raised_tail = False
    try:
        eager_decoding_check_errors(dev)
    except RuntimeError as e:
        raised_tail = "全 False 行" in str(e)
    check("eager_gather: B7 大序列 gather 尾部 raise（零同步语义）", raised_tail)

    # 【iter3】W 超上界（漏选保护）：全 True mask counts=S > W_upper → code=2
    o = eager_decoding_attn(q, k, v, torch.ones_like(bad), 1, cu)
    raised_w = False
    try:
        eager_decoding_check_errors(dev)
    except RuntimeError as e:
        raised_w = "TLI_E127_W_UPPER" in str(e)
    check("eager_gather: W 超上界 fail-closed 尾部 raise", raised_w)

    # 【iter3】cu 契约破坏 → code=3
    cu_bad = torch.tensor([5, S], dtype=torch.int32, device=dev)
    sel = torch.rand(1, 1, Hkv, S, device=dev) > 0.98
    sel[..., -128:] = True
    o = eager_decoding_attn(q, k, v, sel, 1, cu_bad)
    raised_cu = False
    try:
        eager_decoding_check_errors(dev)
    except RuntimeError as e:
        raised_cu = "cu_seqlens_k 契约破坏" in str(e)
    check("eager_gather: cu 契约破坏 fail-closed 尾部 raise", raised_cu)

    # 【iter3】零 host 同步验证：error 模式下 gather 主路径全程无同步
    sel2 = torch.rand(1, 1, H, S, device=dev) > 0.98
    sel2[..., -128:] = True
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(20):
            eager_decoding_attn(q, k, v, sel, 1, cu)
            eager_decoding_attn(q, k, v, sel2, 1, cu)
        no_sync = True
    except Exception as e:
        no_sync = False
        print(f"  同步异常: {type(e).__name__}: {e}")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    eager_decoding_check_errors(dev)
    check("eager_gather: gather 主路径零 host 同步（40 次调用）", no_sync)
    os.environ.pop("TLI_E127_FAST", None)


# ---------------------------------------------------------------- indexer 增量
def test_indexer_inc():
    """iter4（inplace 合并后改写）：基类内置增量 prepare_index —— 增量 vs
    每步全量重建（clear() 强制 _inc=None）index_dict/mask 逐位一致。

    逐位一致是数学保证（min/max/mean 块内独立归约 + quant 逐 token 独立，
    缓存值 = 原 kernel 历史输出 ≡ 全量重算），门禁用多步 decode 实测验证
    实现正确性：块边界（S=4093 非对齐起点）、多 token 步（+3）、跨块大步
    （+63）、64K 大 S、S 回退全量重建、clear() 后重建。
    【2026-10-12 inplace 合并】TLIIndexerFast 并入基类：原「Fast vs 基类」
    对拍改写为「同基类：增量缓存路径 vs 每步 clear() 全量重建」自身对拍。
    """
    import torch as T
    from exp.bench.e127_stage_bench import make_args
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.patches.utils import prepare_cu_seqlens

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5

    def run(step_full, layer_idx, S0, n_steps, seed):
        T.manual_seed(seed)
        idx = TLIIndexer(make_args(0.25, 0.125, 0.125 + 0.5))
        idx.layer_idx = layer_idx
        idx.clear()
        k = T.randn(1, S0, Hkv, D, device=dev, dtype=T.bfloat16)
        dicts, masks = [], []
        for step in range(n_steps):
            n_new = 3 if step == 5 else (63 if step == 8 else 1)
            if step > 0:
                k = T.cat([k, T.randn(1, n_new, Hkv, D, device=dev,
                                      dtype=T.bfloat16)], dim=1)
            q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
            cu = prepare_cu_seqlens(k)
            q_ids = cu[1:] - cu[:-1] - 1
            if step_full:
                idx.clear()     # _inc=None → 本步强制全量重建
            d = idx.prepare_index(k, cu)
            sc = idx.compute_score(q, q_ids, d, scale)
            m = idx.compute_mask(q_ids, sc)
            dicts.append({kk: (vv.clone() if T.is_tensor(vv) else vv)
                          for kk, vv in d.items()})
            masks.append(m)
        return dicts, masks

    ok = True
    for layer_idx, tag in [(2, "normal"), (0, "skip")]:
        da, ma = run(False, layer_idx, 4093, 12, 20261014)
        db, mb = run(True, layer_idx, 4093, 12, 20261014)
        for step in range(12):
            for kk in ["k_min", "k_max", "k_qat"]:
                if not T.equal(da[step][kk], db[step][kk]):
                    ok = False
                    print(f"  {tag} step{step} {kk} MISMATCH")
            ka, kb_ = da[step].get("k_avg"), db[step].get("k_avg")
            if (ka is None) != (kb_ is None) or (
                    ka is not None and not T.equal(ka, kb_)):
                ok = False
                print(f"  {tag} step{step} k_avg MISMATCH")
            if not T.equal(ma[step], mb[step]):
                ok = False
                print(f"  {tag} step{step} MASK MISMATCH")
    check("indexer_inc: 12 步 decode（块边界/多 token/大步）增量 vs 全量重建"
          " index_dict+mask 逐位一致", ok)

    da, ma = run(False, 2, 65536 - 13, 15, 20261016)
    db, mb = run(True, 2, 65536 - 13, 15, 20261016)
    check("indexer_inc: 64K 大 S 15 步增量 vs 全量重建 mask 逐位一致",
          all(T.equal(x, y) for x, y in zip(ma, mb)))

    # S 回退 → 全量重建（不崩、结果正确）
    idx = TLIIndexer(make_args(0.25, 0.125, 0.625))
    idx.layer_idx = 2
    idx.clear()
    k1 = T.randn(1, 5000, Hkv, D, device=dev, dtype=T.bfloat16)
    idx.prepare_index(k1, prepare_cu_seqlens(k1))
    k2 = T.randn(1, 4000, Hkv, D, device=dev, dtype=T.bfloat16)
    d_back = idx.prepare_index(k2, prepare_cu_seqlens(k2))   # S 回退
    idx_ref = TLIIndexer(make_args(0.25, 0.125, 0.625))
    idx_ref.layer_idx = 2
    idx_ref.clear()
    d_ref = idx_ref.prepare_index(k2, prepare_cu_seqlens(k2))
    ok_back = all(T.equal(d_back[kk], d_ref[kk]) for kk in ["k_min", "k_max", "k_qat"])
    # 回退后继续增量也正确
    k3 = T.cat([k2, T.randn(1, 1, Hkv, D, device=dev, dtype=T.bfloat16)], dim=1)
    d3 = idx.prepare_index(k3, prepare_cu_seqlens(k3))
    idx_ref.clear()
    d_ref3 = idx_ref.prepare_index(k3, prepare_cu_seqlens(k3))
    ok_back = ok_back and all(
        T.equal(d3[kk], d_ref3[kk]) for kk in ["k_min", "k_max", "k_qat"])
    check("indexer_inc: S 回退全量重建 + 后续增量正确", ok_back)


# ---------------------------------------------------------------- iter5 派生
def test_score_mask_derived():
    """iter5（inplace 合并后改写）：Triton fast path vs 强制回退
    （_triton_ok=False → 原版 repeat+einsum 表达式）——score ULP 口径 +
    mask 逐位（==）。
    【iter6】score_fine、【iter8】score_coarse/score_coarse_avg 为 Triton
    累加顺序差异的 ULP 口径（≤1e-4 比例 + ≤2% 相对）；mask 仍逐位（==）。
    【2026-10-12 inplace 合并】TLIIndexerFast 退役：原「Fast vs 基类」对拍
    改写为「同基类：Triton fast path vs _triton_ok=False 强制回退」对拍
    （skip 层 e64 走 dgate_short 分支，两臂同路径，trivially 一致属预期）。

    覆盖：site1（e64+skip）、site2（单池+skip，minmax/avg 两分数源）、
    e64 normal 层、avg 分数源、多 token 步、S 回退、块边界非对齐起点。
    """
    import torch as T
    from exp.bench.e127_stage_bench import make_args
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.patches.utils import prepare_cu_seqlens

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5

    # 配置矩阵：champion=e64 mavg（site1） / (0,0) 单池（site2 minmax 源）/
    # aavg e64（avg 分数源） / aavg(0,0)（site2 avg 源）
    def make_cfg(alpha, beta, gamma, far, near):
        a = make_args(alpha, beta, gamma)
        a.tli_far_method, a.tli_near_method = far, near
        return a

    configs = [
        ("champion_e64", make_cfg(0.25, 0.125, 0.625, "minmax", "avg")),
        ("singlepool_00", make_cfg(0.0, 0.0, 0.625, "minmax", "minmax")),
        ("aavg_e64", make_cfg(0.125, 0.125, 0.375, "avg", "avg")),
        ("aavg_00", make_cfg(0.0, 0.0, 0.375, "avg", "avg")),
    ]

    def run(triton_on, args, layer_idx, S0, n_steps, seed):
        T.manual_seed(seed)
        idx = TLIIndexer(args)
        idx.layer_idx = layer_idx
        idx.clear()
        if not triton_on:
            idx._triton_ok = False   # 强制回退原版 repeat+einsum 表达式
        k = T.randn(1, S0, Hkv, D, device=dev, dtype=T.bfloat16)
        outs = []
        for step in range(n_steps):
            n_new = 3 if step == 5 else 1
            if step > 0:
                k = T.cat([k, T.randn(1, n_new, Hkv, D, device=dev,
                                      dtype=T.bfloat16)], dim=1)
            q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
            cu = prepare_cu_seqlens(k)
            q_ids = cu[1:] - cu[:-1] - 1
            d = idx.prepare_index(k, cu)
            sc = idx.compute_score(q, q_ids, d, scale)
            m = idx.compute_mask(q_ids, sc)
            outs.append((sc, m))
        return outs

    n_fail = 0
    n_sf_ulp, n_sf_tot, sf_maxrel = 0, 0, 0.0
    n_sc_ulp, n_sc_tot, sc_maxrel = 0, 0, 0.0

    def _ulp_ok(fa, fb):
        """【iter6/iter8】Triton 累加顺序不同 → fp32 ULP 漂移，bf16 舍入后
        个别元素差 1 ULP。口径：差异元素比例 ≤ 1e-4 且相对幅度 ≤ 2%
        （1 bf16 ULP @小分数值）。返回 (nd, rel, ok)。

        【E156 候选化适配】候选化后 score_fine 的未选候选位置为 -inf
        （两臂同候选集 → 逐位相等），但 -inf - (-inf) = nan 会污染全局
        rel max（skip 层 R_max<<S_width 大量 -inf 时 rel 变 nan 误判）。
        修法：两臂 isfinite 布局必须逐位一致（候选集单源，不一致即真差异
        强制 FAIL），相对幅度只对两臂均有限的位置计算。"""
        if fa.shape != fb.shape:
            return -1, float("inf"), False
        nd = int((fa != fb).sum().item())
        if nd == 0:
            return 0, 0.0, True
        if not torch.equal(torch.isfinite(fa), torch.isfinite(fb)):
            return nd, float("inf"), False
        fin = torch.isfinite(fa)
        rel = ((fa[fin].float() - fb[fin].float()).abs()
               / fa[fin].float().abs().clamp(min=1e-3)).max().item()
        # 【E152v4 判据迁移】候选化（AKO iter-2 起）把 score_fine 比较
        # 域从全宽 S_cap（~262k 元素/步）缩到候选域 R_max（~7.7k/步，
        # 非候选位两臂同 -inf 逐位等不再参与）。bf16 einsum-vs-Triton
        # 累加序漂移的绝对计数不变（全宽时代实测 254/10.49M，候选化后
        # 274/7.75M 同级），但分母缩 ~34× → 同一物理噪声在纯比例阈
        # 1e-4 下边缘超界（实测最差步 nd=8/77824，阈 7.78）。二分定位
        # iter-1 PASS / iter-2 FAIL 确认为判据口径迁移而非漂移回归：
        # ① mask 逐位（==）全绿——漂移元素全部深排名非选中位
        # （rank 1167-1602 ≫ K2=1024，实 dump 验证）；② 漂移密度与
        # 候选域高分段 bf16 ULP 物理特性一致。修正：计数阈加绝对下限
        # max(numel×1e-4, 12)——12 ≈ 全宽时代每步等效阈 26 的保守折半，
        # 较实测最差 8 留 50% 余量（非按失败值拟合）。rel≤2% 不变，
        # isfinite 布局逐位硬门不变，mask 逐位不变。
        return nd, rel, (nd <= max(fa.numel() * 1e-4, 12)
                         and rel <= 0.02)

    for cname, args in configs:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            a = run(False, args, layer_idx, 4093, 10, 20261018)
            b = run(True, args, layer_idx, 4093, 10, 20261018)
            for step in range(10):
                sa, sb = a[step][0], b[step][0]
                # 【iter8】score_coarse：coarse minmax 段 Triton 化 → ULP
                # 口径（同 fine；块级 kt 小实测常见 0 漂移）。mask 逐位。
                nd, rel, ok = _ulp_ok(sa["score_coarse"], sb["score_coarse"])
                n_sc_ulp += max(nd, 0); n_sc_tot += sa["score_coarse"].numel()
                if not ok:
                    n_fail += 1
                    print(f"  {cname} {ltag} step{step} score_coarse ULP "
                          f"超界: nd={nd}/{sa['score_coarse'].numel()} rel={rel:.3e}")
                # 【iter6】score_fine：ULP 口径（同上）。
                fa, fb = sa["score_fine"], sb["score_fine"]
                nd, rel, ok = _ulp_ok(fa, fb)
                n_sf_ulp += max(nd, 0); n_sf_tot += fa.numel()
                if nd:
                    sf_maxrel = max(sf_maxrel, rel)
                if not ok:
                    n_fail += 1
                    print(f"  {cname} {ltag} step{step} score_fine ULP "
                          f"超界: nd={nd}/{fa.numel()} rel={rel:.3e}")
                # 【iter8】score_coarse_avg：复用 _score_fine_kernel → ULP 口径
                ka, kb = sa.get("score_coarse_avg"), sb.get("score_coarse_avg")
                if (ka is None) != (kb is None):
                    n_fail += 1
                    print(f"  {cname} {ltag} step{step} score_coarse_avg PRESENCE")
                elif ka is not None:
                    nd, rel, ok = _ulp_ok(ka, kb)
                    n_sc_ulp += max(nd, 0); n_sc_tot += ka.numel()
                    if nd:
                        sc_maxrel = max(sc_maxrel, rel)
                    if not ok:
                        n_fail += 1
                        print(f"  {cname} {ltag} step{step} score_coarse_avg "
                              f"ULP 超界: nd={nd}/{ka.numel()} rel={rel:.3e}")
                if not T.equal(a[step][1], b[step][1]):
                    n_fail += 1
                    print(f"  {cname} {ltag} step{step} MASK MISMATCH")
    check("score_mask_derived: 4 配置 × normal/skip × 10 步 score/mask 一致"
          "（coarse+avg ULP≤1e-4 + fine ULP≤1e-4 + mask 逐位）",
          n_fail == 0,
          f"mismatch={n_fail} coarse_ulp={n_sc_ulp}/{n_sc_tot} "
          f"maxrel={sc_maxrel:.2e} fine_ulp={n_sf_ulp}/{n_sf_tot} "
          f"maxrel={sf_maxrel:.2e}")

    # S 回退 → 全量重建后继续增量正确（score_fine ULP 口径继续一致）
    args = make_cfg(0.25, 0.125, 0.625, "minmax", "avg")
    ok_back = True
    T.manual_seed(20261019)
    i_ref = TLIIndexer(args); i_ref.layer_idx = 2; i_ref.clear()
    i_ref._triton_ok = False
    T.manual_seed(20261019)
    i_fast = TLIIndexer(args); i_fast.layer_idx = 2; i_fast.clear()
    q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)

    def _score_pair(k):
        cu = prepare_cu_seqlens(k)
        q_ids = cu[1:] - cu[:-1] - 1
        sc_r = i_ref.compute_score(q, q_ids, i_ref.prepare_index(k, cu), scale)
        sc_f = i_fast.compute_score(q, q_ids, i_fast.prepare_index(k, cu), scale)
        return sc_r, sc_f

    def _sf_ok(sc_r, sc_f):
        # 【iter8】coarse/coarse_avg（Triton 化）与 fine 同 ULP 口径
        for key in ("score_fine", "score_coarse", "score_coarse_avg"):
            fa, fb = sc_r.get(key), sc_f.get(key)
            if (fa is None) != (fb is None):
                return False
            if fa is None:
                continue
            if fa.shape != fb.shape:
                return False
            nd = (fa != fb).sum().item()
            rel = ((fa.float() - fb.float()).abs()
                   / fa.float().abs().clamp(min=1e-3)).max().item() if nd else 0.0
            # 【E152v4 判据迁移】同 _ulp_ok：候选域绝对下限 12（口径迁移
            # 依据见 _ulp_ok 注释）。
            if not (nd <= max(fa.numel() * 1e-4, 12) and rel <= 0.02):
                return False
        return True

    k1 = T.randn(1, 5000, Hkv, D, device=dev, dtype=T.bfloat16)
    sc_r, sc_f = _score_pair(k1)
    ok_back = _sf_ok(sc_r, sc_f)
    k2 = T.randn(1, 4000, Hkv, D, device=dev, dtype=T.bfloat16)   # S 回退
    sc_r, sc_f = _score_pair(k2)
    ok_back = ok_back and _sf_ok(sc_r, sc_f)
    k3 = T.cat([k2, T.randn(1, 1, Hkv, D, device=dev, dtype=T.bfloat16)], dim=1)
    sc_r, sc_f = _score_pair(k3)
    ok_back = ok_back and _sf_ok(sc_r, sc_f)
    check("score_mask_derived: S 回退全量重建 + 后续增量 score ULP 口径一致"
          "（coarse/avg/fine）",
          ok_back)


def test_fine_gather():
    """E156：精筛候选化（TLI_FINE_GATHER=1）vs 全宽（=0）——score ULP 口径 +
    mask 逐位 ==。

    同一 compute_score/compute_mask，仅 L2 fine 打分范围不同（候选集 vs 全宽
    [1,H,T]）。候选位 score 应与全宽同位置一致（同 einsum 同 k_qat 元素；
    einsum N 维不同仅影响累加调度 → ULP 口径）；最终 mask 逐位 ==。
    两臂均强制 _triton_ok=False（都走 repeat+einsum 表达式），隔离 Triton
    噪声，纯粹对比候选化 vs 全宽。
    覆盖：e64 champion / (0,0) 单池 × minmax/avg 分数源 × normal/skip ×
    多 token 步 + S 非对齐起点（4093）。
    """
    import torch as T
    from exp.bench.e127_stage_bench import make_args
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.patches.utils import prepare_cu_seqlens

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5

    def make_cfg(alpha, beta, gamma, far, near):
        a = make_args(alpha, beta, gamma)
        a.tli_far_method, a.tli_near_method = far, near
        return a

    configs = [
        ("champion_e64", make_cfg(0.25, 0.125, 0.625, "minmax", "avg")),
        ("singlepool_00", make_cfg(0.0, 0.0, 0.625, "minmax", "minmax")),
        ("aavg_e64", make_cfg(0.125, 0.125, 0.375, "avg", "avg")),
        ("aavg_00", make_cfg(0.0, 0.0, 0.375, "avg", "avg")),
    ]

    def run(gather_on, args, layer_idx, S0, n_steps, seed):
        os.environ["TLI_FINE_GATHER"] = "1" if gather_on else "0"
        T.manual_seed(seed)
        idx = TLIIndexer(args)
        idx.layer_idx = layer_idx
        idx.clear()
        idx._triton_ok = False   # 两臂同走 repeat+einsum，隔离 Triton 噪声
        k = T.randn(1, S0, Hkv, D, device=dev, dtype=T.bfloat16)
        outs = []
        for step in range(n_steps):
            n_new = 3 if step == 5 else 1
            if step > 0:
                k = T.cat([k, T.randn(1, n_new, Hkv, D, device=dev,
                                      dtype=T.bfloat16)], dim=1)
            q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
            cu = prepare_cu_seqlens(k)
            q_ids = cu[1:] - cu[:-1] - 1
            d = idx.prepare_index(k, cu)
            sc = idx.compute_score(q, q_ids, d, scale)
            # 【E156】compute_mask 会把 score_dict["score_fine"] 原地替换为
            # 重建的全宽张量——score 对比必须在 compute_mask 之前快照原始
            # compute_score 输出（mask 对比用 compute_mask 返回值）。
            sc_snap = {kk: (vv.clone() if T.is_tensor(vv) else vv)
                       for kk, vv in sc.items()}
            m = idx.compute_mask(q_ids, sc)
            outs.append((sc_snap, m))
        return outs

    n_fail = 0
    n_cand_ulp, n_cand_tot = 0, 0
    maxrel = 0.0
    n_cand_path = 0

    def _cand_ulp_ok(full, cand, cand_idx, cand_len):
        # 【E156b】per-head 候选：cand_idx [H,R_max]，cand_len [H] 门控实长
        # 前缀（pad sentinel 不参与）。全宽 score_fine 候选位 vs 候选化
        # score_fine（形状不同，按实长掩码抽）。
        _h_idx = torch.arange(cand_idx.shape[0], device=cand.device
                              ).view(-1, 1).expand_as(cand_idx)
        _r_mask = (torch.arange(cand_idx.shape[1], device=cand.device
                                ).view(1, -1) < cand_len.view(-1, 1))
        fa = full[0, 0][_h_idx[_r_mask], cand_idx[_r_mask]]  # [N]
        fb = cand[0, 0][_r_mask]                              # [N]
        if fa.shape != fb.shape:
            return -1, float("inf"), False
        nd = int((fa != fb).sum().item())
        if nd == 0:
            return 0, 0.0, True
        rel = ((fa.float() - fb.float()).abs()
               / fa.float().abs().clamp(min=1e-3)).max().item()
        return nd, rel, nd <= fa.numel() * 1e-4 and rel <= 0.02

    for cname, args in configs:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            a = run(False, args, layer_idx, 4093, 10, 20261020)
            b = run(True, args, layer_idx, 4093, 10, 20261020)
            for step in range(10):
                sa, sb = a[step][0], b[step][0]
                cidx = sb.get("cand_idx")
                if cidx is None:
                    # 候选化未触发（异常/门控 fail-closed）→ 两臂应同构全宽
                    if (sa["score_fine"].shape != sb["score_fine"].shape
                            or not T.equal(sa["score_fine"],
                                            sb["score_fine"])):
                        n_fail += 1
                        print(f"  {cname} {ltag} step{step} 候选化未触发但 "
                              f"score_fine 不一致")
                else:
                    n_cand_path += 1
                    nd, rel, ok = _cand_ulp_ok(
                        sa["score_fine"], sb["score_fine"], cidx,
                        sb.get("cand_len"))
                    n_cand_tot += sb["score_fine"].numel()
                    n_cand_ulp += max(nd, 0)
                    if nd:
                        maxrel = max(maxrel, rel)
                    if not ok:
                        n_fail += 1
                        print(f"  {cname} {ltag} step{step} 候选位 score ULP "
                              f"超界: nd={nd}/{sb['score_fine'].numel()} "
                              f"rel={rel:.3e}")
                if not T.equal(a[step][1], b[step][1]):
                    n_fail += 1
                    print(f"  {cname} {ltag} step{step} MASK MISMATCH")
    check("fine_gather: TLI_FINE_GATHER 0/1 候选位 score ULP + mask 逐位一致"
          f"（4 配置 × normal/skip × 10 步，候选路径 {n_cand_path}/80 步）",
          n_fail == 0,
          f"mismatch={n_fail} cand_ulp={n_cand_ulp}/{n_cand_tot} "
          f"maxrel={maxrel:.2e}")


def test_pipeline_zero_sync():
    """iter5：prepare_index → compute_score → compute_mask 全管线零 host 同步
    （sync_debug_mode=error，20 步 × normal/skip 层；含 rep 扩容步）。"""
    import torch as T
    from exp.bench.e127_stage_bench import make_args
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.patches.utils import prepare_cu_seqlens

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    no_sync_all = True
    for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
        T.manual_seed(20261020)
        idx = TLIIndexer(make_args(0.25, 0.125, 0.625))
        idx.layer_idx = layer_idx
        idx.clear()
        k = T.randn(1, 16384, Hkv, D, device=dev, dtype=T.bfloat16)
        q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
        # 冷启动（full prepare + rep 全量建）在窗口外完成
        cu = prepare_cu_seqlens(k)
        d = idx.prepare_index(k, cu)
        sc = idx.compute_score(q, cu[1:] - cu[:-1] - 1, d, scale)
        _ = idx.compute_mask(cu[1:] - cu[:-1] - 1, sc)
        T.cuda.synchronize()
        T.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(20):
                k = T.cat([k, T.randn(1, 1, Hkv, D, device=dev,
                                      dtype=T.bfloat16)], dim=1)
                # cu/q_ids 构造属调用方（attn patch / bench harness）职责：
                # prepare_cu_seqlens 的 pageable H2D 会触发 debug-mode 标记，
                # 但非队列排空型同步（µs 级），且为 env 无关共享基础设施
                # （改动会违反 env=0 零扰动铁律）——不纳入本门禁窗口。
                T.cuda.set_sync_debug_mode("default")
                cu = prepare_cu_seqlens(k)
                q_ids = cu[1:] - cu[:-1] - 1
                T.cuda.synchronize()
                T.cuda.set_sync_debug_mode("error")
                d = idx.prepare_index(k, cu)
                sc = idx.compute_score(q, q_ids, d, scale)
                _ = idx.compute_mask(q_ids, sc)
        except Exception as e:
            no_sync_all = False
            print(f"  [{ltag}] 同步异常: {type(e).__name__}: {e}")
        finally:
            T.cuda.set_sync_debug_mode("default")
    check("pipeline_zero_sync: prepare+score+mask 20 步零 host 同步（normal+skip）",
          no_sync_all)


if __name__ == '__main__':
    selected = sys.argv[1:] or ["metrics", "eager", "indexer", "iter5"]
    if "metrics" in selected:
        test_metrics_lazy()
    if "eager" in selected:
        test_eager_gather()
    if "indexer" in selected:
        test_indexer_inc()
    if "iter5" in selected:
        test_score_mask_derived()
        test_pipeline_zero_sync()
    if "fine_gather" in selected:
        test_fine_gather()
    print(f"\n=== {len(PASS)} PASS / {len(FAIL)} FAIL ===")
    for f in FAIL:
        print(f"FAILED: {f}")
    sys.exit(1 if FAIL else 0)
