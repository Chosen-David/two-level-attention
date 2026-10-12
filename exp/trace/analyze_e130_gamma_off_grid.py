# E130：mavg (α,β) 网格 + γ=off 细筛自由竞争臂（用户指令 2026-10-12：γ 维在原有
#   9 档基础上补 γ=off 海选，只扫有效组合；禁止改 e2e 代码，仅 mass 分析器扩维）
# ── 生产语义来源（已逐行核实，sparse_attn/indexer/tli_indexer.py）──
#   * L1 页级粗筛照旧（L985-1015）：far 区 topk nb_far 页 + near 区 topk nb_near 页，
#     α 区域切分 + β 页预算 + 角点守卫全保留；sink/swa 块完全排除 L1 池（L1008-1011）。
#   * 细筛分只在 L1 选中页 token 上非零（L1069-1071 topk_mask 过滤 score_fine → softmax p）。
#   * γ=off 默认细筛分路径（L1252-1257，mavg 适用，无簇分量纲问题）：mid_p = p[整个 mid]，
#     k2_free = min(K2_mid, mid 宽) 单池 topk——即 near+far L1 已选页 token 并集上自由竞争，
#     旁路 nt_near/far_budget 配额切分（L1265 起的配额逻辑整体旁路）。
#   * sink/swa 正交强制区直接置位必选不占预算（L1260-1263）。
# ── 约束（用户 2026-10-12 指令公式）──
#   A（off 仍适用）：α·mid_L ≥ β·budget_page_topk·page_size（E98 同款
#     `do_near and near_L < nb_near*BS: continue`）
#   B（γ≠off 才有，off 臂豁免）：β·budget_page_topk·page_size ≥ γ·budget_token
#   far 侧约束 far_budget ≤ far_L 对 off 臂不适用（off 无 per-region token 预算）
#   角点守卫 do_near=(α>0∧β>0)/do_far=(α<1∧β<1) 保留（off 臂 L1 页级切分仍在）
#   mid_len<4096 跳过等既有样本级规则照抄 E98。
# ── 口径 ── 与 E98 相同 trace（/tmp/trace/qwen3-8b 16 样本）、真实全维 softmax 行级
#   mass coverage、B_TOK=2048、BP=64、BS=64；只跑 mavg combo 的 γ=off 臂（键名
#   mavg_a{a}_b{b}_goff）。既有 E98 720 臂不重算，复用 e98_abg_full_grid.json 作对比锚点。
# ── 退化等价自测（数学事实，用作实现正确性端到端校验）──
#   off(0,b) ≡ E98 mavg_a0_b{b}_g0.0（do_near=false → near 零贡献，far 单池拿全预算）
#   off(1,1) ≡ E98 mavg_a1_b1_g1.0（do_far=false → near 单池 min(4096,B_TOK)=B_TOK）
# ── 用法 ── 分片：E98_PROC/E98_NPROC/E98_DEVICE 环境变量（约定保留），shard 写
#   /tmp/e130_shard_{proc}.json；合并出报告：E130_MERGE=1；smoke 自测：E130_SMOKE=1
#   （单样本单层 + 四项验收断言 + γ 档复算对照）。
import json
import os

import torch
import torch.nn.functional as F

import analyze_e64a_ab_grid as base

TRACE = base.TRACE
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e130_mavg_gamma_off.json"
E98_JSON = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_abg_full_grid.json"
D2I, BS, SINK, SWA, TAIL_N = base.D2I, base.BS, base.SINK, base.SWA, base.TAIL_N
B_TOK = 2048
BP = 64                     # 页池总量（与 E98 同协议）
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
FAR_M, NEAR_M = "mavg", "aavg"   # mavg combo（far=minmax 页分 / near=avg 页分），细筛分同源 sc_tok


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    # mavg 不需要贪心聚类 build（cavg 专用）——E130 相对 E98 的唯一省时点
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]
    smoke_ctx = {}
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        mid_hi = t_r + 1 - SWA
        mid_len = mid_hi - SINK
        if mid_len < 4096:
            continue
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        ksub = ksub_layer
        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = F.pad(ksub, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax, kavg = kc.amin(1), kc.amax(1), kc.mean(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kavg)
        # token 子空间精确细筛分（mavg/aavg 同源——生产 off 默认路径 near/far 同用细筛分 p）
        sc_tok = torch.einsum("hgd,shd->hgs", q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32), ksub).sum(1)
        sc_blk = {"mavg": sc_mm[:, SINK // BS:], "aavg": sc_av[:, SINK // BS:]}
        tok_score = {"mavg": sc_tok[:, SINK:mid_hi], "aavg": sc_tok[:, SINK:mid_hi]}
        sink_c = (pos < SINK).view(1, -1)
        swa_c = ((pos >= t_r + 1 - SWA) & (pos <= t_r)).view(1, -1)

        def select_pages(lo_off, hi_off, method, n_pages):
            """L1 页级粗筛（E98 select_sub 前半，只返回 pool：选中页 ∩ 区段 [lo,hi) 相对 SINK）。
            页选择分数源 sc_blk[method]；sink/swa 页天然排除（sc_blk 切片自 SINK//BS 起，
            页数上限 nblk_sub 覆盖至 mid 尾，swa 区页靠区段过滤截除——与 E98 完全同款）。"""
            if n_pages <= 0 or hi_off - lo_off <= 0:
                return torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            bi = torch.arange(sc_blk[method].shape[1], device=device)
            s1 = sc_blk[method].masked_fill(
                ~(((bi * BS + SINK) < SINK + hi_off) & ((bi * BS + BS - 1 + SINK) >= SINK + lo_off)).view(1, -1),
                float("-inf"))
            nblk_sub = s1.shape[-1]
            ib = torch.topk(s1, min(n_pages, nblk_sub), dim=-1).indices
            tok = (ib.unsqueeze(-1) * BS + torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1).clamp(max=mid_len - 1)
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            pool.scatter_(1, tok, True)
            pool &= ((torch.arange(mid_len, device=device) >= lo_off) &
                     (torch.arange(mid_len, device=device) < hi_off)).view(1, -1)
            return pool

        def select_sub(lo_off, hi_off, method, n_pages, n_tokens):
            """E98 原版两级选择（供 smoke 的 γ 档复算对照，逐位对齐 E98）。"""
            if n_pages <= 0 or n_tokens <= 0:
                return torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            pool = select_pages(lo_off, hi_off, method, n_pages)
            ts = tok_score[method].masked_fill(~pool, float("-inf"))
            it = torch.topk(ts, min(n_tokens, mid_len), dim=-1).indices
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand.scatter_(1, it + SINK, True)
            return cand

        def arm_off(a, b, do_near, do_far, nb_near, near_lo_off):
            """γ=off 臂（生产 L1252-1257 默认细筛分路径的 mass 等价实现）：
            L1 页级双池照旧 → near∪far 已选页 token 并集单池 → 细筛分 topk min(B_TOK, 候选数)。
            返回 (cand[Hkv,S], pool[Hkv,mid_len])。"""
            pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
            if do_near:
                pool |= select_pages(near_lo_off, mid_len, NEAR_M, nb_near)
            if do_far:
                pool |= select_pages(0, near_lo_off, FAR_M, BP - nb_near)
            ts = tok_score[FAR_M].masked_fill(~pool, float("-inf"))
            # 生产 k2_free = min(K2_mid, mid 宽)；mass 口径下有效候选 = pool（细筛分只在
            # L1 已选页非零），行候选不足时 topk 会选到 -inf——置位到绝对位置 0（真 sink
            # 区 token，已强制 True，幂等无害且不污染 mid 区计数），保证选中集严格 ⊆ pool
            # （本协议 BP·BS=4096>B_TOK 恒不触发）
            k = min(B_TOK, int(pool.sum(-1).max().item()))
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            if k > 0:
                vals, it = torch.topk(ts, k, dim=-1)
                valid = torch.isfinite(vals)
                it_safe = torch.where(valid, it + SINK, torch.zeros_like(it))
                cand.scatter_(1, it_safe, True)
            cand |= sink_c | swa_c
            return cand, pool

        def gamma_arm(a, b, g, do_near, do_far, nb_near, near_lo_off, far_L, noleak=False):
            """E98 γ 档单臂复算（smoke 对照用，含 E98 全部约束与 max(64,·) 保底口径）。
            noleak=False 为 E98/生产 γ 档原口径（token topk 选满预算，池不足时把池外
            token 也置 True——mask 虚高，生产 γ 档 L1135 min(K2,near 区宽) 同款行为）；
            noleak=True 仅供 AC4 公平断言（泄漏位置不计入，否则细筛分噪声级反例
            0.8056338 vs off 0.8056335 混淆语义验证）。返回 (cand|None, n_leak)。"""
            nt_near = min(int(nb_near * BS * g), B_TOK) if do_near else 0
            nt_far = max(64, B_TOK - nt_near) if do_far else 0
            if do_far and nt_far > far_L:
                return None, 0  # E98 约束：far_budget <= far_L
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            cand |= sink_c | swa_c
            n_leak = 0
            jobs = []
            if do_near:
                jobs.append((select_pages(near_lo_off, mid_len, NEAR_M, nb_near), nt_near))
            if do_far:
                jobs.append((select_pages(0, near_lo_off, FAR_M, BP - nb_near), nt_far))
            for pool_x, nt in jobs:
                if nt <= 0:
                    continue
                tsx = tok_score[FAR_M].masked_fill(~pool_x, float("-inf"))
                vx, ix = torch.topk(tsx, min(nt, mid_len), dim=-1)
                if noleak:
                    v = torch.isfinite(vx)
                    n_leak += int((~v).sum())
                    ix = torch.where(v, ix + SINK, torch.zeros_like(ix))
                    cand.scatter_(1, ix, True)
                else:
                    cand.scatter_(1, ix + SINK, True)
            return cand, n_leak

        row_arms = 0
        for a in GRID:
            near_L = int(a * mid_len)
            far_L = mid_len - near_L
            near_lo_off = mid_len - near_L
            for b in GRID:
                # 角点守卫（与 E98 同款）：α=0/β=0 → near 空；α=1/β=1 → far 空
                do_near = (a > 0) and (b > 0)
                do_far = (a < 1) and (b < 1)
                nb_near = int(round(BP * b)) if do_near else 0
                # 约束 A（off 仍适用）：near 区装得下页池（约束 B/far 约束 off 豁免）
                if do_near and near_L < nb_near * BS:
                    continue
                cand, pool = arm_off(a, b, do_near, do_far, nb_near, near_lo_off)
                res.setdefault(f"mavg_a{a}_b{b}_goff", []).append(cov_mass(cand))
                row_arms += 1
                if os.environ.get("E130_SMOKE"):
                    # 验收①：off 选中 mid token 数恒等于 min(B_TOK, 行 pool 大小)
                    n_sel = cand[:, SINK:mid_hi].sum(-1)
                    exp_sel = torch.clamp(pool.sum(-1), max=B_TOK)
                    assert torch.equal(n_sel, exp_sel), \
                        f"AC1 fail a={a} b={b}: {n_sel.tolist()} vs {exp_sel.tolist()}"
                    # 验收②：sink/swa 强制置位（每行全 True）
                    assert bool(cand[:, :SINK].all()) and bool(cand[:, t_r + 1 - SWA:t_r + 1].all()), \
                        f"AC2 fail a={a} b={b}"
        if os.environ.get("E130_SMOKE") and smoke_ctx.get("row") is None:
            smoke_ctx["row"] = True
            # ---- 验收③④：角点退化等价 + off ≥ 同 (α,β) 全部 γ 档（第一行做一次）----
            for a, b in [(0.0, 0.0), (0.0, 0.5), (1.0, 1.0)]:
                near_L = int(a * mid_len)
                do_near, do_far = (a > 0 and b > 0), (a < 1 and b < 1)
                nb_near = int(round(BP * b)) if do_near else 0
                if do_near and near_L < nb_near * BS:
                    continue
                cand_off, _ = arm_off(a, b, do_near, do_far, nb_near, mid_len - near_L)
                g_ref = 0.0 if a == 0 else 1.0   # 单侧退化参考档：(0,·)→g0（far 全预算）、(1,1)→g1（near 拿满）
                cand_g, _ = gamma_arm(a, b, g_ref, do_near, do_far, nb_near, mid_len - near_L, far_L=mid_len - near_L)
                assert cand_g is not None and torch.equal(cand_off, cand_g), \
                    f"AC3 corner fail a={a} b={b} g={g_ref}"
                print(f"[SMOKE] AC3 corner a={a} b={b} g{g_ref}: off == gamma-arm bitwise", flush=True)
            # 验收④：off ≥ 同 (α,β) 任意 γ 档（抽查冠军配置 (0.25,0.125)，β<0.5 同预算口径下
            # 数学事实：切分 topk 是单池 topk 的约束版）。γ 档 E98/生产口径 token topk 选满
            # 预算（池不足时池外 token 也置 True 虚高，g=1.0 实测 +2.4e-7 反例即此泄漏），
            # 故公平断言用 noleak 口径（泄漏位不计入），泄漏数单独报告。
            a, b = 0.25, 0.125
            near_L = int(a * mid_len)
            do_near, do_far = (a > 0 and b > 0), (a < 1 and b < 1)
            nb_near = int(round(BP * b))
            cand_off, _ = arm_off(a, b, do_near, do_far, nb_near, mid_len - near_L)
            m_off = cov_mass(cand_off)
            g_masses, g_leaks = {}, {}
            for g in GRID:
                cg, nl = gamma_arm(a, b, g, do_near, do_far, nb_near, mid_len - near_L,
                                   far_L=mid_len - near_L, noleak=True)
                if cg is None:
                    print(f"[SMOKE] AC4 a={a} b={b} g={g}: filtered (far_budget>far_L)", flush=True)
                    continue
                g_masses[g] = cov_mass(cg)
                g_leaks[g] = nl
            g_max = max(g_masses.values())
            assert m_off >= g_max - 1e-9, f"AC4 fail: off={m_off} < gamma_max(noleak)={g_max}"
            print(f"[SMOKE] AC4 a={a} b={b}: off={m_off:.9f} >= max_gamma(noleak)={g_max:.9f} "
                  f"({len(g_masses)}/9 gamma arms valid, leaks={ {k: v for k, v in g_leaks.items() if v} })",
                  flush=True)
        print(f"[E130 layer] ri={ri} mid_len={mid_len} off_arms={row_arms}", flush=True)
    del k, q, d, ksub
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def run_sample(name, device):
    n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
    agg = {}
    for li in range(0, n_layers, max(1, n_layers // 12)):
        r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
        if not r:
            continue
        for k2, v in r.items():
            agg.setdefault(k2, []).extend(v)
    return {k2: sum(v) / len(v) for k2, v in agg.items()}


def merge_and_report(n_proc):
    """合并 shards → e130_mavg_gamma_off.json（per_sample/mean/n_arms/note/comparison）。
    对比锚点从 e98_abg_full_grid.json 的 per_sample 重算未舍入均值（E98 mean 存 round4）。"""
    per_sample = {}
    for pid in range(n_proc):
        fp = f"/tmp/e130_shard_{pid}.json"
        if os.path.exists(fp):
            per_sample.update(json.load(open(fp)))
    assert per_sample, "no shards found under /tmp/e130_shard_*.json"
    mean = {}
    for arm in next(iter(per_sample.values())):
        vals = [per_sample[s][arm] for s in per_sample if arm in per_sample[s]]
        mean[arm] = sum(vals) / len(vals)
    e98 = json.load(open(E98_JSON))
    e98_ps = e98["per_sample"]
    def e98_mean(arm):
        vals = [e98_ps[s][arm] for s in e98_ps if arm in e98_ps[s]]
        return sum(vals) / len(vals)
    mavg_e98 = {k: e98_mean(k) for k in next(iter(e98_ps.values())) if k.startswith("mavg_")}
    champ_arm = "mavg_a0.25_b0.125_g0.625"
    champ_val = mavg_e98[champ_arm]
    e98_best_arm, e98_best_val = max(mavg_e98.items(), key=lambda x: x[1])
    off_rank = sorted(mean.items(), key=lambda x: -x[1])
    off_best_arm, off_best_val = off_rank[0]
    # 分区口径最优（双侧臂 do_near&do_far，排除退化单池角点——e2e 候选建议用）
    part_rank = [(k, v) for k, v in off_rank
                 if 0.0 < float(k.split("_a")[1].split("_")[0]) < 1.0
                 and 0.0 < float(k.split("_b")[1].split("_")[0]) < 1.0]
    # 端到端退化等价校验：off(0,0) ≡ E98 (0,0,g0.0)、off(1,1) ≡ E98 (1,1,g1.0)
    degen_check = {}
    for off_arm, e98_arm in [("mavg_a0.0_b0.0_goff", "mavg_a0.0_b0.0_g0.0"),
                             ("mavg_a0.0_b0.5_goff", "mavg_a0.0_b0.5_g0.0"),
                             ("mavg_a1.0_b1.0_goff", "mavg_a1.0_b1.0_g1.0")]:
        degen_check[off_arm] = {"e98_ref": e98_arm, "off_mean": mean.get(off_arm),
                                "e98_mean": mavg_e98.get(e98_arm),
                                "abs_diff": abs(mean.get(off_arm, 0) - mavg_e98.get(e98_arm, 1))}
    comparison = {
        "off_rank_top10": [{"arm": k, "mean": round(v, 6)} for k, v in off_rank[:10]],
        "anchor_champion_e98": {"arm": champ_arm, "mean": round(champ_val, 6)},
        "anchor_e98_mavg_best": {"arm": e98_best_arm, "mean": round(e98_best_val, 6)},
        "off_best": {"arm": off_best_arm, "mean": round(off_best_val, 6)},
        "delta_off_best_vs_champion": round(off_best_val - champ_val, 6),
        "delta_off_best_vs_e98_best": round(off_best_val - e98_best_val, 6),
        "off_partitioned_best_top3": [{"arm": k, "mean": round(v, 6)} for k, v in part_rank[:3]],
        "degenerate_equivalence_check": degen_check,
        "e2e_candidates": {
            "note": "off 最优 (α,β) e2e 候选（--tli_gamma off）；overall 含退化单池角点，"
                    "partitioned 仅双侧分区臂（α,β∈(0,1) 开区间）",
            "overall": [{"alpha": float(off_best_arm.split("_a")[1].split("_")[0]),
                         "beta": float(off_best_arm.split("_b")[1].split("_")[0]),
                         "mean": round(off_best_val, 6)}],
            "partitioned": [{"alpha": float(k.split("_a")[1].split("_")[0]),
                             "beta": float(k.split("_b")[1].split("_")[0]),
                             "mean": round(v, 6)} for k, v in part_rank[:2]],
        },
    }
    note = ("E130: mavg gamma=off free-competition arms on (alpha,beta) 9x9 grid, B_TOK=2048 BP=64 BS=64, "
            "16 samples TRACE=/tmp/trace/qwen3-8b. Semantics source (production, verified line-by-line): "
            "sparse_attn/indexer/tli_indexer.py L1188-1264 E122 gamma=off -- default fine-score path "
            "L1252-1257: single-pool topk of fine scores over L1-selected-page tokens of full mid "
            "(p nonzero only on L1-selected pages per L1069-1071 topk_mask filter); L1 near/far page "
            "split retained (L985-1015: near topk nb_near pages by avg page score, far topk BP-nb_near "
            "by minmax; sink/swa blocks excluded from L1 pool L1008-1011); sink/swa forced outside "
            "budget (L1260-1263). Constraints (user 2026-10-12): A alpha*mid_L >= beta*BP*BS kept "
            "(E98-equivalent near_L < nb_near*BS -> skip); B beta*BP*BS >= gamma*B_TOK exempted (no "
            "gamma); far-side far_budget <= far_L exempted (off has no per-region token budget); "
            "corner guards do_near=(a>0&b>0)/do_far=(a<1&b<1) kept; mid_len<4096 rows skipped as E98. "
            "E98 720-arm grid NOT recomputed; e98_abg_full_grid.json per_sample reused as comparison "
            "anchor (means recomputed unrounded). Degenerate equivalence (implementation check): "
            "off(0,b) == E98(0,b,g0.0), off(1,1) == E98(1,1,g1.0) exact.")
    out = {"per_sample": per_sample, "mean": mean, "n_arms": len(mean),
           "n_samples": len(per_sample), "note": note, "comparison": comparison}
    json.dump(out, open(OUT, "w"), indent=1)
    # ---- stdout 对比小报告 ----
    print("=" * 72)
    print("[E130] gamma=off mavg grid report  (n_arms=%d, n_samples=%d)" % (len(mean), len(per_sample)))
    print("-- off top10 (mean mass) --")
    for r in comparison["off_rank_top10"]:
        print(f"  {r['arm']:<28} {r['mean']:.6f}")
    print(f"-- anchors (E98, per_sample-recomputed unrounded) --")
    print(f"  champion {champ_arm}: {champ_val:.6f}")
    print(f"  e98 mavg best {e98_best_arm}: {e98_best_val:.6f}")
    print(f"-- off best: {off_best_arm} = {off_best_val:.6f}")
    print(f"  delta vs champion: {off_best_val - champ_val:+.6f}")
    print(f"  delta vs e98 best: {off_best_val - e98_best_val:+.6f}")
    print("-- off partitioned best (alpha,beta in open (0,1), e2e candidates) --")
    for r in comparison["e2e_candidates"]["partitioned"]:
        print(f"  alpha={r['alpha']} beta={r['beta']} mean={r['mean']:.6f}")
    print("-- degenerate equivalence check (|off - e98_ref| should be ~0) --")
    for k, v in degen_check.items():
        print(f"  {k} vs {v['e98_ref']}: |diff|={v['abs_diff']:.3e}")
    print("saved ->", OUT)
    return out


if __name__ == "__main__":
    import multiprocessing as mp  # noqa: F401  (与 E98 保持 import 形态一致)

    if os.environ.get("E130_MERGE"):
        merge_and_report(int(os.environ.get("E98_NPROC", "16")))
        raise SystemExit(0)
    proc_id = int(os.environ.get("E98_PROC", "0"))
    n_proc = int(os.environ.get("E98_NPROC", "16"))
    device = os.environ.get("E98_DEVICE", "cpu")
    torch.set_num_threads(1)
    names = sorted(n for n in os.listdir(TRACE)
                   if os.path.isfile(os.path.join(TRACE, n, "meta.json")))
    if os.environ.get("E130_SMOKE"):
        # smoke：单样本单层四项验收（语义手推对照），CPU 即可
        name = os.environ.get("E130_SMOKE_SAMPLE", names[0])
        lf = f"{TRACE}/{name}/layer00.pt"
        print(f"[SMOKE] sample={name} layer=layer00 device={device}", flush=True)
        r = eval_layer(lf, device)
        n_arm_rows = sum(len(v) for v in r.values())
        print(f"[SMOKE] OK: {len(r)} off arms, {n_arm_rows} (arm,row) entries, "
              f"all 4 acceptance checks passed", flush=True)
        raise SystemExit(0)
    mine = [n for i, n in enumerate(names) if i % n_proc == proc_id]
    print(f"[E130 proc{proc_id}/{n_proc}] device={device} samples={mine}", flush=True)
    out = {}
    for name in mine:
        out[name] = run_sample(name, device)
        print(f"[E130 proc{proc_id}] done {name} arms={len(out[name])}", flush=True)
    json.dump(out, open(f"/tmp/e130_shard_{proc_id}.json", "w"))
    print(f"[E130 proc{proc_id}] shard saved", flush=True)
