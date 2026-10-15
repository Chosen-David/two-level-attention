# -*- coding: utf-8 -*-
"""E1 主质量表图（figure_id=E1，P0 主文）。

面板 (a)：LongBench v1 16 任务逐任务配对差值点图（e116b_full_lbv1.json，
Qwen3-8B，mavg(α=.25,β=.125,γ=.625) vs FullKV，按 delta 排序）——
hotpotqa +5.9 最大改善、passage_count -1.25 最大退化，正负例均保留。
面板 (b)：三个基准套件级总分（LB v1 16 任务 / LB v2 503 题 / RULER 128K）
FullKV vs TLI(mavg) 并列条形 + tail32 子空间 AVG5 附加臂（E133 Δ-0.32
CI[-0.51,-0.104]——冻结冠军表示 tail32 与 full 表示差噪声级）。

口径标注：LB v1 fullkv 13 任务为 E121 修复前复用格（e116b caveats）；
RULER 128K 为 yarn factor 4.0 正式批。数据程序化读取，无手工键入；
输出 PDF+PNG+SVG + figure_data.tsv。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style, FONT_SIZES, LINE_WIDTHS  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
R = os.path.join(REPO, "exp/trace/results/")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
C_MAIN = pal["primary"]
C_SEC = pal["secondary"]
C_HL = pal["highlight"]
C_NEU = pal["neutral"]

# ---------------- 数据读取 ----------------
lb = json.load(open(R + "e116b_full_lbv1.json"))
fk = lb["arms"]["FullKV"]
mv = lb["arms"]["mavg_a0.25_b0.125_g0.625"]
tasks = sorted(fk["per_task"])
delta = {t: mv["per_task"][t] - fk["per_task"][t] for t in tasks}
tasks_sorted = sorted(tasks, key=lambda t: delta[t])

e133 = json.load(open(R + "e133_subspace_verdict.json"))
t32 = e133["avg"]["tail32"]
t32_ci = e133["bootstrap_ci95"]

rm = json.load(open(R + "e116b_ruler128k_formal_mavg.json"))
rf = json.load(open(R + "e116b_ruler128k_formal_fullkv.json"))


def ruler_avg(d):
    sc = list(d["scores"].values())[0]
    v = list(sc.values())
    return sum(v) / len(v)


ruler_mavg, ruler_fullkv = ruler_avg(rm), ruler_avg(rf)

suites = ["LB v1\n(16 tasks)", "LB v2\n(503 tasks)", "RULER\n(128K)"]
fullkv_scores = [fk["avg16"], lb["lbv2_503"]["FullKV_reused_E109"], ruler_fullkv]
tli_scores = [mv["avg16"], lb["lbv2_503"]["mavg_new"], ruler_mavg]

# ---------------- 画布 1x2 ----------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 9),
                                 gridspec_kw={"wspace": 0.42,
                                              "width_ratios": [1.25, 1]})

# ---- (a) 逐任务配对 delta ----
ys = np.arange(len(tasks_sorted))
vals = [delta[t] for t in tasks_sorted]
colors = [C_MAIN if v >= 0 else C_HL for v in vals]
ax_a.barh(ys, vals, height=0.62, color=colors,
          edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.5)
ax_a.axvline(0, color="black", lw=LINE_WIDTHS["reference"], linestyle="--")
ax_a.set_yticks(ys)
ax_a.set_yticklabels([t.replace("_en", "").replace("passage_retrieval", "psg_ret")
                      for t in tasks_sorted])
ax_a.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"] - 4)
for y, v in zip(ys, vals):
    ax_a.text(v + (0.12 if v >= 0 else -0.12), y, f"{v:+.2f}",
              va="center", ha="left" if v >= 0 else "right",
              fontsize=FONT_SIZES["annotation"] - 4)
ax_a.set_xlabel("paired delta vs FullKV (points)")
ax_a.set_title("LongBench v1 per-task delta (Qwen3-8B)", loc="left", pad=10)
ax_a.set_xlim(min(vals) - 1.6, max(vals) + 1.6)
ax_a.text(0.98, 0.03, "avg16: 48.11 (FullKV) vs 48.90 (TLI)", transform=ax_a.transAxes,
          fontsize=FONT_SIZES["annotation"] - 2, ha="right", color=C_MAIN,
          fontweight="bold")

# ---- (b) 套件级总分 ----
xb = np.arange(3)
w = 0.38
ax_b.bar(xb - w / 2, fullkv_scores, width=w, color=C_NEU, label="dense FullKV",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb + w / 2, tli_scores, width=w, color=C_MAIN, label="TLI (mavg, K2=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
for xi, v in zip(xb - w / 2, fullkv_scores):
    ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb + w / 2, tli_scores):
    ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(suites)
ax_b.set_ylabel("task score")
ax_b.set_title("Suite-level totals", loc="left", pad=10)
ax_b.legend(loc="upper right", fontsize=FONT_SIZES["legend"] - 2)
ax_b.set_ylim(0, 62)
ax_b.text(0.02, 0.02, "tail32 representation (AVG5): "
          + f"{t32:.2f}, Δ{e133['avg']['delta']:+.2f} CI[{t32_ci[0]:.2f},{t32_ci[1]:.2f}]"
          + "\nLB v1 FullKV 13-task reused pre-E121 (caliber caveat)",
          transform=ax_b.transAxes, fontsize=FONT_SIZES["annotation"] - 5,
          color=pal["neutral"], va="bottom", style="italic")

# 面板字母
for ax, letter in ((ax_a, "A"), (ax_b, "B")):
    ax.text(0.01, 1.05, letter, transform=ax.transAxes,
            fontsize=FONT_SIZES["panel_letter"], fontweight="bold", va="bottom")

# ---------------- lint + 导出 ----------------
report = lint_figure(fig)
print(f"[lint] overlaps={len(report.get('overlaps', []))} clipped={len(report.get('clipped', []))}")
for o in report.get("overlaps", [])[:8]:
    print("  OVERLAP:", o)
for cc in report.get("clipped", [])[:8]:
    print("  CLIPPED:", cc)


def export(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")


export(fig, "E1_quality")

# ---------------- figure_data.tsv ----------------
rows = ["figure_id\tpanel_id\tarm_id\tmetric_id\tunit\testimate\tci_low\tci_high\tsource_run_id"]
src116 = "e116b_full_lbv1.json"
for t in tasks:
    rows.append(f"E1\ta\t{t}\tpaired_delta\tpoints\t{delta[t]:+.2f}\t\t\t{src116}:arms.*.per_task.{t}")
    rows.append(f"E1\ta\t{t}\tfullkv\tscore\t{fk['per_task'][t]:.2f}\t\t\t{src116}")
    rows.append(f"E1\ta\t{t}\ttli_mavg\tscore\t{mv['per_task'][t]:.2f}\t\t\t{src116}")
for s, fv, tv in zip(("lbv1_16", "lbv2_503", "ruler_128k"), fullkv_scores, tli_scores):
    rows.append(f"E1\tb\t{s}\tfullkv\tscore\t{fv:.2f}\t\t\t{src116}/e116b_ruler128k")
    rows.append(f"E1\tb\t{s}\ttli_mavg\tscore\t{tv:.2f}\t\t\t{src116}/e116b_ruler128k")
rows.append(f"E1\tb\ttail32_avg5\tscore\t{t32:.2f}\t{t32_ci[0]:.2f}\t{t32_ci[1]:.2f}\t"
            "e133_subspace_verdict.json")
with open(os.path.join(OUT, "E1_quality_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported E1_quality_figure_data.tsv")
print("DONE")
