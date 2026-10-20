# -*- coding: utf-8 -*-
"""E155d 固定配额三维网格消融图（figure_id=A6，nb/nt/nl 绝对配额网格）。

数据源：exp/trace/results/e155d_grid_verdict.json（87/87 格零失败终版判决，
2026-10-21 06:19，六机单主机完整副本口径、排序按 AVG5 点估计、判决按 top-3
配对 CI）。全部程序化读取，不硬编码数字（champion/CTRL 引用自 JSON 字段）。

面板设计（两轴故事 + 半覆盖机制 + nt 安全降格，2026-10-21 用户裁决）：
  (a) nb×nl 交互热图（格值 = 该 (nb,nl) 下全部 nt 臂 AVG5 均值）：
      半覆盖律对角线 nb·64 = nl/2（即 nb = nl/128，四格 (8,1024)(16,2048)
      (32,4096)(64,8192) 描边）；冠军格 (32,4096,nt448) 星标（臂值 48.668，
      格均值 48.26 另在 TSV）；小 nb × nl8192 崩塌区（覆盖率 < 1/4）视觉呈现。
  (b) 边缘均值（非加权，跨 nt 臂）：nb 轴单调-平台（46.37→48.21，nb4→8
      +0.97 为最大台阶）；nl 轴 1024-4096 平台 + 8192 惩罚（−0.57）。
      无 CI —— 判决仅 top-3 配对 CI（全含 0）；边缘为格均值口径，正式配对
      CI 属补充实验（见 advice 2026-10-21）。
  (c) 半覆盖律 mass 机制（e155d_half_coverage_anatomy.json，120/128 query
      离线重析 E140b）：近区 64-token 页按质量降序的累计份额曲线，
      top-½ 页 median 0.933 / mean 0.909 / p10 0.802（均匀基线 0.5）——
      近区质量集中在约一半页里，与网格最优 nb·64=nl/2 一致。口径边界：
      32 Q-head mean z 分数诊断平面，非生产评分器、非任务质量。
  (d) nt 主效应 null：69 对 nt-on(格内最优) vs nt-off 配对差按 nb 分组散点；
      分 nb 均值 nb32 +0.23 / nb64 −0.20 / 其余 ≈0；总均值 +0.04、42/69 为正。
      nt=640 仅 nb≥16 格存在（支撑集混淆），不影响本图按 (nb,nl) 配对口径。
  (e) 冠军 vs 单池对照（同批、预算对齐 budget_avg 1021.5 vs 1024.0）：
      逐任务配对差 bar（hotpot +2.11 / musique −1.32 / gov +0.10 / qasper
      +0.40 / repo +4.74）+ AVG5 总差 +1.21 —— 两级+分区 vs 单池自由竞争的
      同批干净对照（CTRL 无 α/β/γ 耦合）。

版式硬约束（用户 2026-10-16）：图内不堆说明文字、无 note box，数字/结论进
caption/元数据；本图沿用 O 图小字号口径（title 15 / label 13 / tick 11.5 /
annotation 10.5），figsize 9.5in 对 IEEE 双栏缩放比 ≈0.75。

诚实边界：in_place_top1=false（当前手稿紧凑设计 (8,320) 族非网格选择）；
top-3 配对 CI 全含 0（同族格噪声带内等价）；5 任务 LongBench 口径
（RULER 长度未测，T4/T6 待跑）；nt 轴去留由 RULER 64K 七臂判决（T4 预登记）。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC = os.path.join(REPO, "exp/trace/results/e155d_grid_verdict.json")
SRC_HC = os.path.join(REPO, "exp/trace/results/e155d_half_coverage_anatomy.json")
for _p in (SRC, SRC_HC):
    if not os.path.exists(_p):
        raise SystemExit(f"missing input: {_p}")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
C_MAIN = pal["primary"]
C_SEC = pal["secondary"]
C_HL = pal["highlight"]
C_NEU = pal["neutral"]

T, L, TK, AN = 15, 13, 11.5, 10.5   # O 图小字号口径

d = json.load(open(SRC))
rank = d["ranking"]
ctrl = d["ctrl_singlepool"]["pred_ctrl_singlepool"]
champ = rank[0]

hc = json.load(open(SRC_HC))   # 半覆盖律 mass 诊断（120/128 query）
HC_FRACS = [0.125, 0.25, 0.5, 0.75]

NBS = [4, 8, 16, 32, 64]
NLS = [1024, 2048, 4096, 8192]

# ---- (a) nb×nl 交互格均值（跨 nt 臂非加权）----
cell = {(nb, nl): [] for nb in NBS for nl in NLS}
cell_arm = {}          # (nb,nl,nt) -> avg5
for e in rank:
    nb, nl, nt = e["nb"], e["nl"], e["nt"]
    cell_arm[(nb, nl, str(nt))] = e["avg5"]
    cell[(nb, nl)].append(e["avg5"])
grid = np.full((len(NBS), len(NLS)), np.nan)
for i, nb in enumerate(NBS):
    for j, nl in enumerate(NLS):
        if cell[(nb, nl)]:
            grid[i, j] = float(np.mean(cell[(nb, nl)]))

# 半覆盖律对角线格（nb·64 = nl/2）
diag_cells = {(nb, nl) for nb in NBS for nl in NLS if nb * 64 == nl // 2}
champ_nb, champ_nl, champ_nt = champ["nb"], champ["nl"], str(champ["nt"])

# ---- (b) 边缘均值 ----
nb_mar = [float(np.mean([x for (nb, nl), vs in cell.items()
                         if nb == b and vs for x in vs]))
          for b in NBS]
nl_mar = [float(np.mean([x for (nb, nl), vs in cell.items()
                         if nl == n and vs for x in vs]))
          for n in NLS]

# ---- (c) nt 配对（best_nt in-cell vs ntoff，69 对）----
pairs = d["stories"]["nt_vs_ntoff_pairs"]
by_nb = {b: [p["delta"] for p in pairs if p["nb"] == b] for b in NBS}
nb_delta_mean = {b: float(np.mean(v)) for b, v in by_nb.items() if v}
all_deltas = [p["delta"] for p in pairs]
pos_frac = sum(1 for x in all_deltas if x > 0) / len(all_deltas)

# ---- (d) 冠军 vs CTRL 逐任务 ----
TASKS = ["hotpotqa", "musique", "gov_report", "qasper", "repobench"]
TASK_LAB = ["hotpot", "musique", "gov_rep", "qasper", "repo"]
task_delta = [champ["tasks"][t] - ctrl["tasks"][t] for t in TASKS]
avg_delta = champ["avg5"] - ctrl["avg5"]

# ---------------- 画布 2x3，(e) 横跨下排右侧两列 ----------------
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.6),
                         gridspec_kw={"wspace": 0.50, "hspace": 0.44})
ax_a, ax_b, ax_c = axes[0]
ax_d, ax_e = axes[1][:2]
fig.delaxes(axes[1][2])
_p1 = axes[1][1].get_position()
_p2 = axes[1][2].get_position()
ax_e.set_position([_p1.x0, _p1.y0, _p2.x1 - _p1.x0, _p1.height])

# ---- (a) 热图 ----
cmap = LinearSegmentedColormap.from_list(
    "score", ["#B2182B", "#F7F7F7", pal["primary"]])
im = ax_a.imshow(grid, cmap=cmap, vmin=44.0, vmax=48.9, aspect="auto",
                 origin="lower")
for i, nb in enumerate(NBS):
    for j, nl in enumerate(NLS):
        v = grid[i, j]
        if np.isnan(v):
            continue
        dark = v < 45.4 or v > 48.3
        ax_a.text(j, i, f"{v:.2f}", ha="center", va="center",
                  fontsize=TK - 2, color="white" if dark else "black")
        if (nb, nl) in diag_cells:
            ax_a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                        edgecolor=C_HL, lw=2.2))
            ax_a.text(j, i - .34, "½", ha="center", va="center",
                      fontsize=AN - 1.5, color=C_HL, fontweight="bold")
        if (nb, nl) == (champ_nb, champ_nl):
            ax_a.plot(j + .30, i - .30, marker="*", markersize=15,
                      color=C_HL, markeredgecolor="black",
                      markeredgewidth=.7, zorder=5)
ax_a.set_xticks(range(len(NLS)))
ax_a.set_xticklabels([str(n) for n in NLS], fontsize=TK - 1)
ax_a.set_yticks(range(len(NBS)))
ax_a.set_yticklabels([str(b) for b in NBS], fontsize=TK - 1)
ax_a.set_xlabel("near length $n_l$ (tokens)", fontsize=L)
ax_a.set_ylabel("near pages $n_b$", fontsize=L)
ax_a.set_title("(a)  Grid AVG5: half-coverage diagonal", loc="left",
               pad=10, fontsize=T, fontweight="bold")
# 不加 colorbar：20 格全部印有点估计，量程进 caption（用户版式铁律）

# ---- (b) 边缘均值 ----
xb = np.arange(len(NBS))
ax_b.bar(xb, nb_mar, width=0.58, color=C_MAIN,
         edgecolor="black", linewidth=.7)
for xi, v in zip(xb, nb_mar):
    ax_b.text(xi, v + 0.09, f"{v:.2f}", ha="center", va="bottom",
              fontsize=AN - 2, rotation=90)
for k, (xi, dv) in enumerate(zip(xb[1:], np.diff(nb_mar))):
    ax_b.annotate("", xy=(xi - .5, max(nb_mar) + 1.55),
                  xytext=(xi - 1.45, max(nb_mar) + 1.55),
                  arrowprops=dict(arrowstyle="->", color=C_NEU, lw=1.1))
    ax_b.text(xi - .95, max(nb_mar) + 1.72 + (0.60 if k % 2 else 0),
              f"{dv:+.2f}", ha="center", fontsize=AN - 3, color=C_NEU)
# nl 边缘放同轴右侧（第二组）
xn = np.arange(len(NLS)) + len(NBS) + 1
ax_b.bar(xn, nl_mar, width=0.58, color=C_SEC,
         edgecolor="black", linewidth=.7)
for xi, v in zip(xn, nl_mar):
    ax_b.text(xi, v + 0.09, f"{v:.2f}", ha="center", va="bottom",
              fontsize=AN - 2, rotation=90)
ax_b.set_xticks(list(xb) + list(xn))
ax_b.set_xticklabels([str(b) for b in NBS] + [str(n // 1024) + "K" for n in NLS],
                     fontsize=TK - 2.5)
ax_b.set_xlim(-0.7, xn[-1] + 0.95)
ax_b.set_ylabel("marginal mean (points)", fontsize=L)
ax_b.set_title("(b)  Marginals: $n_b$ axis; $n_l$ penalty at 8192",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_b.set_ylim(43.8, 51.3)
ax_b.set_yticks([44, 46, 48, 50])
ax_b.plot([], [], color=C_MAIN, lw=6, label="$n_b$ (near pages)")
ax_b.plot([], [], color=C_SEC, lw=6, label="$n_l$ (near length)")
ax_b.legend(loc="lower center", fontsize=AN - 1.5, framealpha=0.92)

# ---- (c) 半覆盖律 mass 机制（120/128 query 离线重析）----
qmat = np.array([[q["top_frac_share"][f"{f:.3f}"] for f in HC_FRACS]
                 for q in hc["queries"]])
for row in qmat:
    ax_c.plot(HC_FRACS, row, color=C_NEU, lw=0.5, alpha=0.13, zorder=1)
med = np.median(qmat, axis=0)
p10 = np.percentile(qmat, 10, axis=0)
p90 = np.percentile(qmat, 90, axis=0)
ax_c.fill_between(HC_FRACS, p10, p90, color=C_SEC, alpha=0.22,
                  zorder=2, label="p10–p90")
ax_c.plot(HC_FRACS, med, color=C_HL, lw=2.6, marker="o", markersize=5,
          zorder=4, label="median")
ax_c.plot([0, 1], [0, 1], color="black", lw=1.0, linestyle="--",
          zorder=3, label="uniform")
hi = HC_FRACS.index(0.5)
ax_c.annotate(f"{med[hi]:.3f}", xy=(0.5, med[hi]),
              xytext=(0.545, med[hi] - 0.085), fontsize=AN - 1,
              color=C_HL, fontweight="bold")
ax_c.set_xticks(HC_FRACS)
ax_c.set_xticklabels(["1/8", "1/4", "1/2", "3/4"], fontsize=TK - 1)
ax_c.set_xlabel("top-$f$ fraction of near pages", fontsize=L)
ax_c.set_ylabel("cumulative mass share", fontsize=L)
ax_c.set_title("(c)  Half-coverage: near mass concentration", loc="left",
               pad=10, fontsize=T, fontweight="bold")
ax_c.set_xlim(0.09, 0.78)
ax_c.set_ylim(-0.02, 1.04)
ax_c.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_c.legend(loc="lower right", fontsize=AN - 2, framealpha=0.92)

# ---- (d) nt null ----
rng = np.random.default_rng(20261021)
for k, b in enumerate(NBS):
    vals = by_nb[b]
    xs = rng.uniform(k - 0.16, k + 0.16, len(vals))
    ax_d.scatter(xs, vals, s=26, color=C_SEC, alpha=0.75,
                 edgecolor="black", linewidth=.45, zorder=3)
    m = nb_delta_mean[b]
    ax_d.plot([k - 0.30, k + 0.30], [m, m], color=C_HL, lw=2.6,
              solid_capstyle="round", zorder=4)
    ax_d.text(k, 0.62 + (0.13 if k % 2 else 0), f"{m:+.2f}", ha="center",
              fontsize=AN - 3, color=C_HL, fontweight="bold")
ax_d.axhline(0, color="black", lw=1.0, linestyle="--")
ax_d.set_xticks(range(len(NBS)))
ax_d.set_xticklabels([str(b) for b in NBS], fontsize=TK - 1)
ax_d.set_xlabel("near pages $n_b$", fontsize=L)
ax_d.set_ylabel("best-$n_t$ minus $n_t$-off (points)", fontsize=L)
ax_d.set_title("(d)  $n_t$ main effect: null (69 pairs)", loc="left",
               pad=10, fontsize=T, fontweight="bold")
ax_d.set_yticks([-0.5, 0.0, 0.5])
ax_d.set_ylim(-0.75, 0.85)
ax_d.text(0.985, 0.05,
          f"mean {np.mean(all_deltas):+.2f};  {sum(1 for x in all_deltas if x>0)}/69 > 0",
          transform=ax_d.transAxes, fontsize=AN - 1, ha="right",
          color=C_NEU)

# ---- (e) 冠军 vs 单池 ----
xd = np.arange(len(TASKS) + 1)
vals = task_delta + [avg_delta]
cols = [C_MAIN if v >= 0 else "#B2182B" for v in vals[:-1]] + [C_HL]
ax_e.bar(xd, vals, width=0.58, color=cols,
         edgecolor="black", linewidth=.7)
for xi, v in zip(xd, vals):
    ax_e.text(xi, v + (0.14 if v >= 0 else -0.34), f"{v:+.2f}",
              ha="center", fontsize=AN - 1.5,
              color=C_HL if xi == len(TASKS) else "black",
              fontweight="bold" if xi == len(TASKS) else "normal")
ax_e.axhline(0, color="black", lw=1.0)
ax_e.set_xticks(xd)
ax_e.set_xticklabels(TASK_LAB + ["AVG5"], fontsize=TK - 2.5)
ax_e.set_ylabel("champion minus single pool (points)", fontsize=L - 1)
ax_e.set_title("(e)  Champion vs single-pool control", loc="left",
               pad=10, fontsize=T, fontweight="bold")
ax_e.set_yticks([-2, 0, 2, 4, 6])
ax_e.set_ylim(-2.4, 6.0)

# ---------------- lint + 导出 ----------------
report = lint_figure(fig)
print(f"[lint] overlaps={len(report.get('overlaps', []))} clipped={len(report.get('clipped', []))}")
for o in report.get("overlaps", [])[:8]:
    print("  OVERLAP:", o)
for c in report.get("clipped", [])[:8]:
    print("  CLIPPED:", c)

for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(OUT, f"A_e155d_quota_grid.{ext}"), dpi=300,
                bbox_inches="tight")

# ---------------- figure_data.tsv ----------------
rows = ["figure_id\tpanel_id\tarm_id\tmetric_id\tunit\testimate\tci_low\tci_high\tsource_run_id"]
for i, nb in enumerate(NBS):
    for j, nl in enumerate(NLS):
        if not np.isnan(grid[i, j]):
            rows.append(f"A6\ta\tnb{nb}_nl{nl}\tcell_mean_avg5\tpoints\t{grid[i, j]:.3f}\t\t\te155d_grid_verdict")
for nb, v in zip(NBS, nb_mar):
    rows.append(f"A6\tb\tnb{nb}\tmarginal_mean_avg5\tpoints\t{v:.3f}\t\t\te155d_grid_verdict")
for nl, v in zip(NLS, nl_mar):
    rows.append(f"A6\tb\tnl{nl}\tmarginal_mean_avg5\tpoints\t{v:.3f}\t\t\te155d_grid_verdict")
for f in HC_FRACS:
    s = hc["summary_by_frac"][f"{f:.3f}"]
    rows.append(f"A6\tc\ttop{f:.3f}\tcumulative_mass_share_median\tshare\t{s['median']:.4f}\t\t\te155d_half_coverage_anatomy")
    rows.append(f"A6\tc\ttop{f:.3f}\tcumulative_mass_share_p10\tshare\t{s['p10']:.4f}\t\t\te155d_half_coverage_anatomy")
    rows.append(f"A6\tc\ttop{f:.3f}\tcumulative_mass_share_p90\tshare\t{s['p90']:.4f}\t\t\te155d_half_coverage_anatomy")
for p in pairs:
    rows.append(f"A6\td\tnb{p['nb']}_nl{p['nl']}\tbestnt_minus_ntoff\tpoints\t{p['delta']:.4f}\t\t\te155d_grid_verdict:stories")
for t, v in zip(TASKS, task_delta):
    rows.append(f"A6\te\t{t}\tchampion_minus_ctrl\tpoints\t{v:.3f}\t\t\te155d_grid_verdict:ranking+ctrl")
rows.append(f"A6\te\tAVG5\tchampion_minus_ctrl\tpoints\t{avg_delta:.3f}\t\t\te155d_grid_verdict:ranking+ctrl")
rows.append(f"A6\tmeta\tchampion\tavg5\tpoints\t{champ['avg5']:.3f}\t\t\te155d_grid_verdict:ranking[0]")
rows.append(f"A6\tmeta\tctrl_singlepool\tavg5\tpoints\t{ctrl['avg5']:.3f}\t\t\te155d_grid_verdict:ctrl_singlepool")
rows.append(f"A6\tmeta\tchampion\tbudget_avg\ttokens\t{champ['budget_avg']:.2f}\t\t\te155d_grid_verdict:ranking[0]")
rows.append(f"A6\tmeta\tctrl_singlepool\tbudget_avg\ttokens\t{ctrl['budget_avg']:.2f}\t\t\te155d_grid_verdict:ctrl_singlepool")
rows.append(f"A6\tmeta\tin_place_top1\tbool\t-\t{d['stories']['in_place_top1']}\t\t\te155d_grid_verdict:stories")
for k, v in d["paired_ci"].items():
    rows.append(f"A6\tmeta\t{k}\tpaired_diff\tpoints\t{v['point_diff']:.4f}\t{v['ci95'][0]:.4f}\t{v['ci95'][1]:.4f}\te155d_grid_verdict:paired_ci")
with open(os.path.join(OUT, "A_e155d_quota_grid_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported A_e155d_quota_grid_figure_data.tsv")
print("DONE")
