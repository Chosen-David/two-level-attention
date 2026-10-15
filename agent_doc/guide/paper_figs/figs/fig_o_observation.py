# -*- coding: utf-8 -*-
"""Historical observation renderer; archived results are not corrected evidence.

See ../observation/E140_VALIDITY.md. Existing JSON/figures remain unchanged.
This renderer cannot consume v2 records or repair old near/far/QK boundaries.
"""
import os
if os.environ.get("E140_ALLOW_HISTORICAL_DIAGNOSTIC") != "1":
    raise SystemExit("Historical observation invalid for paper conclusions; see E140_VALIDITY.md. No files written.")

import json
import math
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
WT = "/tmp/wt_e138_ablation"
SRC = os.path.join(WT, "exp/trace/results/e140_o_trace.json")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
C_MAIN = pal["primary"]
C_SEC = pal["secondary"]
C_TER = pal["tertiary"]
C_HL = pal["highlight"]
C_NEU = pal["neutral"]

d = json.load(open(SRC))
cells = d["cells"]

# ---------------- (a) 距离密度聚合 ----------------
pairs = []
for c in cells:
    for o1 in c["o1"]:
        for b in o1["bins"]:
            pairs.append((b["d_lo"], b["d_hi"], b["mass"], b["n_tok"]))
d_lo = np.array([p[0] for p in pairs])
d_hi = np.array([p[1] for p in pairs])
m = np.array([p[2] for p in pairs])
n_tok = np.array([p[3] for p in pairs])
dc = (d_lo + d_hi) / 2
edges = np.logspace(math.log10(300), math.log10(17000), 25)
dens, cx = [], []
for i in range(len(edges) - 1):
    sel = (dc >= edges[i]) & (dc < edges[i + 1])
    if sel.sum() == 0:
        continue
    dens.append(m[sel].sum() / n_tok[sel].sum())
    cx.append(math.sqrt(edges[i] * edges[i + 1]))
dens = np.array(dens)
cx = np.array(cx)

nl = [o1["near_len"] for c in cells for o1 in c["o1"]]
near_len_med = float(np.median(nl))

# ---------------- (b) 四区质量 + far worst head ----------------
zones = ["sink", "swa", "near", "far"]
zmean = [np.mean([o1["mass_" + z] for c in cells for o1 in c["o1"]]) for z in zones]
fw = float(np.mean([o1["far_worst_head"] for c in cells for o1 in c["o1"]]))

# ---------------- (c) O3 三臂 ----------------
arms = ["tail32", "random32", "high32"]
mets = ["page_recall", "overlap_recall", "retained_mass"]
arm_val = {}
for a in arms:
    arm_val[a] = [np.mean([c["o3"][a][mm]["mean"] for c in cells]) for mm in mets]

# ---------------- 画布 1x3 ----------------
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(21, 9),
                                       gridspec_kw={"wspace": 0.46,
                                                    "width_ratios": [1.1, 0.85, 1.1]})

# ---- (a) 距离密度 ----
ax_a.plot(cx, dens, "-o", color=C_MAIN, markersize=10,
          markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"],
          lw=LINE_WIDTHS["data"], label="mid density (mass/token)")
ax_a.axvline(near_len_med, color=C_HL, lw=LINE_WIDTHS["reference"], linestyle="--")
ax_a.text(near_len_med * 1.12, dens.max() * 0.75,
          "near/far boundary\n(" + r"$\alpha$=.25, median " + f"{near_len_med:.0f})",
          fontsize=FONT_SIZES["annotation"] - 3, color=C_HL)
ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(270, 21000)
ax_a.set_xticks([300, 1000, 3000, 10000])
ax_a.set_ylim(1e-5, 1e-3)
ax_a.set_yticks([1e-5, 1e-4, 1e-3])
ax_a.set_xlabel("distance d = q_pos - k_pos (tokens)")
ax_a.set_ylabel("attention density (prob mass per token)")
ax_a.set_title("Attention mass vs distance (mid zone)", loc="left", pad=10)
ax_a.legend(loc="upper right", fontsize=FONT_SIZES["legend"] - 2)
ax_a.text(0.02, 0.02, "far tail rebounds at d>9K:\nretrieval hotspots beyond near",
          transform=ax_a.transAxes, fontsize=FONT_SIZES["annotation"] - 3,
          color=C_HL, va="bottom", style="italic")

# ---- (b) 四区质量 ----
xb = np.arange(4)
ax_b.bar(xb, zmean, width=0.6, color=[C_NEU, C_TER, C_SEC, C_MAIN],
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.plot([3], [fw], marker="^", color=C_HL, markersize=18,
          markeredgecolor="black", markeredgewidth=LINE_WIDTHS["marker_edge"] * 0.6,
          linestyle="none", label="far, worst head")
for xi, v in zip(xb, zmean):
    ax_b.text(xi, v + 0.015, f"{v:.2f}", ha="center",
              fontsize=FONT_SIZES["annotation"] - 2)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(["sink", "swa", "near", "far"])
ax_b.set_ylabel("mean prob mass (group mean)")
ax_b.set_title("Where the mass lives (4 zones)", loc="left", pad=10)
ax_b.legend(loc="upper left", fontsize=FONT_SIZES["legend"] - 2)
ax_b.text(0.98, 0.04, "far = 83% of mid mass", transform=ax_b.transAxes,
          fontsize=FONT_SIZES["annotation"], color=C_HL, ha="right")
ax_b.set_ylim(0, 0.85)

# ---- (c) O3 三臂 ----
xc = np.arange(3)
wc = 0.26
colors3 = [C_MAIN, C_NEU, C_SEC]
for i, (a, c) in enumerate(zip(arms, colors3)):
    ax_c.bar(xc + (i - 1) * wc, arm_val[a], width=wc, color=c,
             edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6,
             label=a)
    for j, (xi, v) in enumerate(zip(xc + (i - 1) * wc, arm_val[a])):
        if j == 0:  # page recall 三臂 0.87-0.94 挤叠，数值进 tsv 不上图
            continue
        ax_c.text(xi, v + 0.015, f"{v:.2f}", ha="center",
                  fontsize=FONT_SIZES["annotation"] - 4)
ax_c.set_xticks(xc)
ax_c.set_xticklabels(["page\nrecall", "overlap\nrecall", "retained\nmass"])
ax_c.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] - 2)
ax_c.set_ylabel("score vs full-ref TopK-768")
ax_c.set_title("Subspace frequency arms (32 of 128 dims)", loc="left", pad=10)
ax_c.legend(loc="upper center", fontsize=FONT_SIZES["legend"] - 2)
ax_c.text(0.98, 0.04, "tail32 keeps 79% of mass;\nhigh32 only 26%", transform=ax_c.transAxes,
          fontsize=FONT_SIZES["annotation"], color=C_HL, ha="right")
ax_c.set_ylim(0, 1.12)

# 面板字母（嵌入标题左侧空位由 loc="left" 保证不撞）
for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
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


export(fig, "O_observation")

# ---------------- figure_data.tsv ----------------
rows = ["figure_id\tpanel_id\tarm_id\tmetric_id\tunit\testimate\tsource_run_id\tsource_field"]
src = "e140_o_trace.json"
for x, y in zip(cx, dens):
    rows.append(f"O\ta\tmid_distance\tdensity\tmass/token\t{y:.4e}\t{src}\tbins(log-grid)")
for z, v in zip(zones, zmean):
    rows.append(f"O\tb\t{z}\tmean_mass\tfrac\t{v:.4f}\t{src}\tmass_{z}")
rows.append(f"O\tb\tfar_worst_head\tmean_mass\tfrac\t{fw:.4f}\t{src}\tfar_worst_head")
rows.append(f"O\tb\tnear_len_median\tboundary\ttokens\t{near_len_med:.0f}\t{src}\tnear_len")
for a in arms:
    for mm, v in zip(mets, arm_val[a]):
        rows.append(f"O\tc\t{a}\t{mm}\tfrac\t{v:.4f}\t{src}\to3.{a}.{mm}.mean")
with open(os.path.join(OUT, "O_observation_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported O_observation_figure_data.tsv")
print("DONE")
