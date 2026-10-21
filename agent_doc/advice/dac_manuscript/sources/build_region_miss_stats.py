# -*- coding: utf-8 -*-
"""生成 (b) 面板「区域互补漏失」数据块（2026-10-21 第四轮重设计）。

从 pin 住的 E140b pooling probe 重算每条 query 的 near/far 漏失参考质量份额，
外加 near 预算敏感性扫描（hybrid near-avg + far-minmax vs 全局 minmax），
写入 sources/data/region_miss_stats.json（自包含档案，带输入 sha256）。

旧 (b)「quota floors far capture」配对臂面板已退役（nt 轴降格 + nb/nl 绝对配额
口径）；本数据块是其替代叙事的数值来源：
  avg 漏失 2/3 在 far / minmax 漏失 2/3 在 near → 近区平手 + 远区 4× 互补。

fail-closed：sel_avg/sel_minmax 与 probe 存档逐位复现断言；页↔bin 区域对齐
断言；任何断言失败即退出非零，不写半数据。
"""
import hashlib
import json
import sys

import numpy as np

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC_TRACE = f"{REPO}/exp/trace/results/e140_o_trace_v2.json"
SRC_PROBE = f"{REPO}/exp/trace/results/e140b_pooling_anatomy.json"
OUT = f"{REPO}/agent_doc/advice/dac_manuscript/sources/data/region_miss_stats.json"
BUDGET = 2048


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def select_pages(order_vals, ntoks, budget):
    """按分值降序逐页累积；接纳前检查 acc+n_tok<=budget（与 probe v2 同规则，
    并列由 stable sort 按页序打破）。"""
    acc, sel = 0, []
    for i in sorted(range(len(order_vals)), key=lambda i: -order_vals[i]):
        if acc + ntoks[i] <= budget:
            sel.append(i)
            acc += ntoks[i]
    return set(sel)


def main():
    trace = json.load(open(SRC_TRACE))
    probe = json.load(open(SRC_PROBE))
    cells = trace["cells"]

    rows, ams_all, mms_all = [], [], []
    repro_fail = 0
    for c in probe["cells"]:
        oc = next((t for t in cells
                   if (t["task"], t["sample"], t["layer"])
                   == (c["task"], c["sample"], c["layer"])), None)
        o1 = {o["q_pos"]: o for o in oc["o1"]} if oc else {}
        for q in c["queries"]:
            p = q["pages"]
            n = len(p["pos_lo"])
            so, sa, sm = set(p["sel_oracle"]), set(p["sel_avg"]), set(p["sel_minmax"])
            om = sum(p["mass"][i] for i in so)
            if om <= 0:
                continue
            # 选择复现断言（fail-closed）
            ra = select_pages(p["avg"], p["n_tok"], BUDGET)
            rm = select_pages(p["minmax"], p["n_tok"], BUDGET)
            if ra != sa or rm != sm:
                repro_fail += 1
            ams_all.append(sum(p["mass"][i] for i in so - sa) / om)
            mms_all.append(sum(p["mass"][i] for i in so - sm) / om)
            # 区域归属：页绝对区间对照 near_lo（页↔bin 对齐由 trace 断言）
            rb = q["region_bounds"]
            regs = []
            ok = True
            o = o1.get(q["q_pos"])
            if o is None:
                ok = False
            else:
                bl = {o["q_pos"] - b["d_hi"]: b for b in o["bins"]}
                for i, plo in enumerate(p["pos_lo"]):
                    b = bl.get(plo)
                    if b is None or plo + b["n_tok"] != plo + p["n_tok"][i]:
                        ok = False
                        break
                    end = plo + p["n_tok"][i]
                    if end <= rb["near_lo"]:
                        regs.append("far")
                    elif plo >= rb["near_lo"]:
                        regs.append("near")
                    else:
                        ok = False
                        break
            if not ok:
                continue
            nid = f"{c['task']}:s{c['sample']}:L{c['layer']}:q{q['q_pos']}"

            def reg_miss(sel):
                ms = so - sel
                return (sum(p["mass"][i] for i in ms if regs[i] == "near") / om,
                        sum(p["mass"][i] for i in ms if regs[i] == "far") / om,
                        sum(1 for i in ms if regs[i] == "near"),
                        sum(1 for i in ms if regs[i] == "far"))
            an, af, anp, afp = reg_miss(sa)
            mn, mf, mnp, mfp = reg_miss(sm)
            rows.append(dict(id=nid, n_ref=len(so),
                             avg_miss_near=an, avg_miss_far=af,
                             mm_miss_near=mn, mm_miss_far=mf,
                             avg_pages_near=anp, avg_pages_far=afp,
                             mm_pages_near=mnp, mm_pages_far=mfp))

    if repro_fail:
        sys.exit(f"selection reproduction failed for {repro_fail} queries")

    # ---- hybrid near 预算扫描（诊断平面；只作披露，不做进图）----
    hybrid = []
    global_mm = []
    for c in probe["cells"]:
        for q in c["queries"]:
            p = q["pages"]
            rb = q["region_bounds"]
            so = set(p["sel_oracle"])
            om = sum(p["mass"][i] for i in so)
            if om <= 0:
                continue
            idx_near = [i for i in range(len(p["pos_lo"]))
                        if p["pos_lo"][i] >= rb["near_lo"]]
            idx_far = [i for i in range(len(p["pos_lo"]))
                       if p["pos_lo"][i] + p["n_tok"][i] <= rb["near_lo"]]
            gmm = select_pages(p["minmax"], p["n_tok"], BUDGET)
            global_mm.append(sum(p["mass"][i] for i in so - gmm) / om)
            for b in range(128, 1536 + 1, 64):
                sel_n = select_pages([p["avg"][i] for i in idx_near],
                                      [p["n_tok"][i] for i in idx_near], b)
                sel_f = select_pages([p["minmax"][i] for i in idx_far],
                                     [p["n_tok"][i] for i in idx_far],
                                     BUDGET - b)
                hyb = {idx_near[j] for j in sel_n} | {idx_far[j] for j in sel_f}
                hybrid.append((b, sum(p["mass"][i] for i in so - hyb) / om))
    budgets = sorted({b for b, _ in hybrid})
    scan = {b: float(np.mean([m for bb, m in hybrid if bb == b]))
            for b in budgets}
    best_b = min(scan, key=scan.get)
    out = dict(
        provenance=dict(
            probe_source=SRC_PROBE.split("two-level-attention/")[1],
            probe_sha256=sha256(SRC_PROBE),
            trace_source=SRC_TRACE.split("two-level-attention/")[1],
            trace_sha256=sha256(SRC_TRACE),
            transformation=("Per-query near/far missed reference-mass shares "
                            "recomputed from pinned E140b pages (selection "
                            "reproduction asserted); region = absolute page "
                            "interval vs near_lo. Hybrid near-budget scan is "
                            "a diagnostic-plane disclosure, not a figure arm. "
                            "No model run."),
        ),
        metric=("missed reference mass / total reference mass, mean over "
                "correlated queries; near/far split by page region"),
        n_queries_total=128,
        n_queries_region=len(rows),
        independent_sequences=4,
        aggregates=dict(
            avg_miss_near=float(np.mean([r["avg_miss_near"] for r in rows])),
            avg_miss_far=float(np.mean([r["avg_miss_far"] for r in rows])),
            mm_miss_near=float(np.mean([r["mm_miss_near"] for r in rows])),
            mm_miss_far=float(np.mean([r["mm_miss_far"] for r in rows])),
            avg_pages_near=float(np.mean([r["avg_pages_near"] for r in rows])),
            avg_pages_far=float(np.mean([r["avg_pages_far"] for r in rows])),
            mm_pages_near=float(np.mean([r["mm_pages_near"] for r in rows])),
            mm_pages_far=float(np.mean([r["mm_pages_far"] for r in rows])),
            avg_miss_total=float(np.mean(ams_all)),
            mm_miss_total=float(np.mean(mms_all)),
        ),
        hybrid_scan=dict(
            note=("near pages by avg score, far pages by minmax, total budget "
                  "2048; mean missed reference mass vs near budget. Best "
                  "hybrid never beats global minmax on this diagnostic plane; "
                  "budget division belongs to the E155d grid axis."),
            best_near_budget=best_b,
            best_miss=scan[best_b],
            global_minmax_miss=float(np.mean(global_mm)),
            scan=scan,
        ),
        rows=rows,
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    a = out["aggregates"]
    print(f"queries(region)={out['n_queries_region']} "
          f"avg near/far={a['avg_miss_near']:.4f}/{a['avg_miss_far']:.4f} "
          f"mm near/far={a['mm_miss_near']:.4f}/{a['mm_miss_far']:.4f} "
          f"hybrid best={scan[best_b]:.4f}@{best_b} "
          f"global_mm={out['hybrid_scan']['global_minmax_miss']:.4f}")


if __name__ == "__main__":
    main()
