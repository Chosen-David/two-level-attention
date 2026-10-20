# -*- coding: utf-8 -*-
"""E155d 半覆盖律 mass 诊断：近区页质量浓度（离线重析 E140b）。

问题：E155d 固定配额网格的最优格落在 nb·64 = nl/2（近区 L1 页池恰好覆盖
近区一半）——近区 attention mass 本身是否集中在约一半页里？

口径（诚实边界）：
- 输入 = e140b_pooling_anatomy.json（v2 schema，父 trace e140_o_trace_v2.json，
  实际 q_norm/k_norm + 实际 RoPE 采集）；128 query × 32 Q-head mean 的
  z 分数页质量（x 128^-0.5），诊断单选择平面，非生产 near-mean 评分器，
  非任务质量。
- 每条 query 的 near 区（region_bounds.near_lo 起）逐 64-token 页 mass
  降序排序，累计份额在 f ∈ {1/8, 1/4, 1/2, 3/4} 处取值；均匀随机选页的
  期望份额 = f（基线）。near 页数 < 16 的 query 剔除（8/128 条，短序列
  近区不足）。
- 判决字段：per-query 份额 + 分位数汇总 + 与 E155d 网格最优对角线的
  对应关系（机制证据，非因果证明：mass 集中与网格最优的一致性）。

输出：exp/trace/results/e155d_half_coverage_anatomy.json
"""
import hashlib
import json
import os
from datetime import datetime

import numpy as np

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC = os.path.join(REPO, "exp/trace/results/e140b_pooling_anatomy.json")
OUT = os.path.join(REPO, "exp/trace/results/e155d_half_coverage_anatomy.json")
FRACS = [0.125, 0.25, 0.5, 0.75]

if not os.path.exists(SRC):
    raise SystemExit(f"missing input: {SRC}")
d = json.load(open(SRC))
src_sha = hashlib.sha256(open(SRC, "rb").read()).hexdigest()

rows, skipped = [], 0
for c in d["cells"]:
    for q in c["queries"]:
        m = np.asarray(q["pages"]["mass"], dtype=float)
        if len(m) < 16:
            skipped += 1
            continue
        ms = np.sort(m)[::-1]
        tot = float(ms.sum())
        if tot <= 0:
            skipped += 1
            continue
        cum = np.cumsum(ms) / tot
        n = int(len(ms))
        rec = {
            "task": c["task"], "sample": c["sample"], "layer": c["layer"],
            "q_pos": q["q_pos"], "n_near_pages": n, "near_mass": tot,
            "top_frac_share": {f"{f:.3f}": float(cum[max(1, int(round(f * n))) - 1])
                               for f in FRACS},
        }
        rows.append(rec)

if not rows:
    raise SystemExit("no valid queries")

summary = {}
for f in FRACS:
    v = np.array([r["top_frac_share"][f"{f:.3f}"] for r in rows])
    summary[f"{f:.3f}"] = {
        "mean": float(v.mean()), "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
        "min": float(v.min()),
        "uniform_baseline": f,
        "excess_over_uniform_mean": float(v.mean() - f),
    }

out = {
    "schema": "e155d-half-coverage-anatomy-v1",
    "status": "diagnostic-mechanism-illustration",
    "generated": datetime.now().isoformat(timespec="seconds"),
    "parent": {
        "file": "e140b_pooling_anatomy.json",
        "sha256": src_sha,
        "grandparent": d["meta"].get("parent_trace", "e140_o_trace_v2.json"),
        "score_rule": d["meta"].get("score_rule", ""),
    },
    "question": ("E155d grid optimum at nb*64 = nl/2 (near L1 pool covers half "
                 "the near region): is near-region attention mass itself "
                 "concentrated in about half the pages?"),
    "protocol": {
        "per_query": "near pages (64-token) sorted by mass desc; cumulative share at f",
        "exclusion": "queries with <16 near pages or zero total mass excluded",
        "baseline": "uniform random page choice has expected share = f",
        "boundary": ("32-Q-head-mean z-score page mass from a diagnostic single "
                     "selection plane; not the production near-average scorer; "
                     "not task quality; consistency with grid optimum, not causality"),
    },
    "n_query_total": 128, "n_query_valid": len(rows), "n_query_skipped": skipped,
    "n_valid_by_min_pages": {str(k): int(sum(1 for r in rows if r["n_near_pages"] >= k))
                             for k in (16, 32, 64, 128)},
    "summary_by_frac": summary,
    "queries": rows,
}
with open(OUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"valid {len(rows)}/128 queries (skipped {skipped})")
for f in FRACS:
    s = summary[f"{f:.3f}"]
    print(f"top-{f:.3f}: mean {s['mean']:.3f} median {s['median']:.3f} "
          f"p10 {s['p10']:.3f} p90 {s['p90']:.3f} (baseline {f})")
print("written:", OUT)
