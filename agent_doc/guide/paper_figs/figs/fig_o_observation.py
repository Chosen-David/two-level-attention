# -*- coding: utf-8 -*-
"""O 系列观察图（figure_id=O1/O3/O4，P0 主文动机图）。

设计叙事（参照 StreamingLLM「竖虚线临界点标注」、Quest「图内文字直接给结论」、
SparQ「方案对照同图竞技」三家的观察图范式），四个面板一条链：
  (a) 质量在哪  —— O1 距离密度：mid 区密度随距离衰减，但 d>9K 远端长尾回升
      （检索热点在 near 之外）→ 分区动机：near/far 统计性质相反。
  (b) 质量有多少 —— O1 四区占比：far 独占 mid 质量 83%，且 GQA 组内最差头
      0.65 远超组均值 0.48 → 独立 far 预算防挤出动机。
  (c) 用什么维度看 —— O3 子空间对照：同 32/128 维预算，tail32 保留 79%
      full-ref 质量 vs random 47% vs high 26% → 位置先验（低频完整 RoPE 对）
      而非「任意 32 维」。
  (d) 选择器选对了没 —— O4 选择行为闭环：生产口径两级选择器选中 token 的
      距离分布贴合 oracle 的远端回升；K1=128 时 far 配额被配额公式压到 128
      断供，K1=64 恢复 432，overlap 0.47→0.53。

可视化手法：
  - near/far 分区底色在 (a)(d) 两面板复用（视觉连续性，读者一眼认出同一边界）；
  - near/far 边界用竖虚线 + 标签（StreamingLLM 临界点标注法）；
  - 每面板结论用白底框注写在图内（Quest 图内结论法），避开刻度与曲线；
  - (c) 只画预登记主指标 retained_mass（guide O3：二选一，recall 进 tsv/附录）。

数据全部程序化读取自 repo 内 e140_o_trace.json + e140_o4_select_trace.json；
输出 PDF+PNG+SVG + figure_data.tsv。
"""
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
from style_config import apply_style  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC = os.path.join(REPO, "exp/trace/results/e140_o_trace.json")
if not os.path.exists(SRC):  # 兼容旧临时路径
    SRC = "/tmp/wt_e138_ablation/exp/trace/results/e140_o_trace.json"
SRC4 = os.path.join(REPO, "exp/trace/results/e140_o4_select_trace.json")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
C_MAIN = pal["primary"]
C_SEC = pal["secondary"]
C_TER = pal["tertiary"]
C_HL = pal["highlight"]
C_NEU = pal["neutral"]

# 本图为 1x4 宽版多面板，style_config 的 PPT 级字号过大，按面板宽度收敛
T = 15      # 面板标题
L = 13      # 轴标签
TK = 11.5   # 刻度
AN = 10.5   # 图内注释
LG = 10.5   # 图例
LW = 2.4    # 数据线宽

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

# ---------------- (c) O3 三臂（主指标 retained_mass） ----------------
arms = ["tail32", "random32", "high32"]
arm_lab = {"tail32": "tail32\n(low-freq)",
           "random32": "random32\n(unpaired)",
           "high32": "high32\n(high-freq)"}
arm_rm = {a: float(np.mean([c["o3"][a]["retained_mass"]["mean"] for c in cells]))
          for a in arms}
# 次级指标进 tsv
mets_all = ["page_recall", "overlap_recall", "retained_mass"]
arm_val = {a: [float(np.mean([c["o3"][a][mm]["mean"] for c in cells]))
               for mm in mets_all] for a in arms}

# ---------------- (d) O4 选择行为 ----------------
d4 = json.load(open(SRC4))
cells4 = d4["cells"]
ARM4 = [("oracle", "oracle top-768 (full 128d)", C_NEU, "--"),
        ("tli_k128", "TLI selector, K1=128 (prod.)", C_MAIN, "-"),
        ("tli_k64", "TLI selector, K1=64", C_HL, "-")]
edges4 = np.logspace(math.log10(300), math.log10(17000), 25)


def bucket_rate(cells, arm):
    """log 距离网格上的选中率 = Σ n_sel / Σ n_tok（跨 task/sample/layer/query）。"""
    sel_sum = np.zeros(len(edges4) - 1)
    tok_sum = np.zeros(len(edges4) - 1)
    for c in cells:
        for b in c["o4"]["arms"][arm]["bins"]:
            dcen = math.sqrt(b["d_lo"] * b["d_hi"])
            i = np.searchsorted(edges4, dcen) - 1
            if 0 <= i < len(sel_sum):
                sel_sum[i] += b["n_sel"]
                tok_sum[i] += b["n_tok"]
    cx4 = np.sqrt(edges4[:-1] * edges4[1:])
    rate = np.where(tok_sum > 0, sel_sum / np.maximum(tok_sum, 1), np.nan)
    return cx4, rate


bnd4 = float(np.median([c["o4"]["q_pos"] - c["o4"]["near_lo"] for c in cells4]))
# far 区选中 token 数：trace 按 8 个 kv-head 聚合存储，/8 得每 kv-head 均值
far_avg = {a: float(np.mean([c["o4"]["arms"][a]["far_sel"]
                             for c in cells4])) / 8 for a, *_ in ARM4}
ovl_avg = {a: float(np.mean([c["o4"]["arms"][a]["overlap_recall"]
                             for c in cells4])) for a, *_ in ARM4}

# ---------------- 画布 1x4 ----------------
fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
    1, 4, figsize=(21.5, 5.2),
    gridspec_kw={"wspace": 0.36, "width_ratios": [1.12, 0.82, 0.80, 1.12]})

XLO, XHI = 270, 21000


def shade_zones(ax, bnd):
    ax.axvspan(XLO, bnd, color=C_MAIN, alpha=0.07, lw=0)
    ax.axvspan(bnd, XHI, color=C_HL, alpha=0.08, lw=0)
    ax.axvline(bnd, color=C_HL, lw=2.0, linestyle="--")


def note(ax, x, y, text, ha="left", va="top", color="black", size=AN):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=size, color=color,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="0.78", alpha=0.92))


# ---- (a) 距离密度 ----
shade_zones(ax_a, near_len_med)
ax_a.plot(cx, dens, "-o", color=C_MAIN, markersize=5.5,
          markeredgecolor="white", markeredgewidth=1.2, lw=LW, zorder=5)
ax_a.text(near_len_med * 0.90, 3.5e-4, f"boundary (median {near_len_med:.0f})",
          fontsize=AN - 1.5, color=C_HL, va="center", ha="right",
          rotation=90)
ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(XLO, XHI)
ax_a.set_xticks([300, 1000, 3000, 10000])
ax_a.set_ylim(1e-5, 1.3e-3)
ax_a.set_yticks([1e-5, 1e-4, 1e-3])
ax_a.set_xlabel("distance d = q_pos − k_pos (tokens)", fontsize=L)
ax_a.set_ylabel("attention density\n(mass per token)", fontsize=L)
ax_a.set_title("(a) Density steps by distance", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_a.tick_params(labelsize=TK)
ax_a.text(0.28, 0.965, "near", transform=ax_a.transAxes, ha="center",
          fontsize=AN, color=C_MAIN, fontweight="bold")
ax_a.text(0.74, 0.965, "far", transform=ax_a.transAxes, ha="center",
          fontsize=AN, color=C_HL, fontweight="bold")
note(ax_a, 0.04, 0.44, "near: dense →\nblock-avg suffices",
     color=C_MAIN, size=AN - 1)
note(ax_a, 0.04, 0.03, "far: sparse, tail rebounds at d>9K\n"
     "→ no-miss bound + token fine-select", color=C_HL, size=AN - 1,
     va="bottom")

# ---- (b) 四区质量：100% 堆叠横条 + 最差头标记 ----
zcol = {"sink": C_NEU, "swa": C_TER, "near": C_SEC, "far": C_MAIN}
left = 0.0
for z, v in zip(zones, zmean):
    ax_b.barh([0], [v], left=left, height=0.7, color=zcol[z],
              edgecolor="black", linewidth=1.2)
    if v >= 0.15:
        ax_b.text(left + v / 2, 0, f"{z}\n{v:.2f}", ha="center", va="center",
                  fontsize=AN, color="white" if z == "far" else "black")
    left += v
# near 段太窄，向下引线标注（底部空间空闲）
near_c = zmean[0] + zmean[1] + zmean[2] / 2
ax_b.annotate(f"near {zmean[2]:.2f}", xy=(near_c, -0.36),
              xytext=(0.56, -0.62), fontsize=AN - 1, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color="0.4", lw=1.0))
# far 最差头标记（文字直接放标记左侧，避免箭头横穿）
ax_b.plot([fw], [0.68], marker="v", color=C_HL, markersize=11,
          markeredgecolor="black", markeredgewidth=1.2, zorder=6)
ax_b.text(fw - 0.03, 0.68, f"far, worst kv-head: {fw:.2f}", fontsize=AN - 1,
          color=C_HL, ha="right", va="center")
ax_b.text(0.02, 0.97, "far = 83% of mid-zone mass\n→ dedicated far budget",
          transform=ax_b.transAxes, fontsize=AN - 1, color=C_HL,
          ha="left", va="top",
          bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                    edgecolor="0.78", alpha=0.92))
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(-0.75, 1.35)
ax_b.set_yticks([])
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_xlabel("share of total mass", fontsize=L)
ax_b.set_title("(b) Mass by zone", loc="left", pad=10, fontsize=T,
               fontweight="bold")
ax_b.tick_params(labelsize=TK)
ax_b.spines[["left", "right", "top"]].set_visible(False)

# ---- (c) O3 三臂（主指标 retained mass） ----
colors3 = [C_MAIN, C_NEU, C_SEC]
for i, (a, col) in enumerate(zip(arms, colors3)):
    ax_c.bar([i], [arm_rm[a]], width=0.58, color=col, edgecolor="black",
             linewidth=1.2)
    ax_c.text(i, arm_rm[a] + 0.025, f"{arm_rm[a]:.2f}", ha="center",
              fontsize=AN, fontweight="bold")
ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels([arm_lab[a] for a in arms], fontsize=TK - 1)
ax_c.set_ylabel("retained full-ref mass", fontsize=L)
ax_c.set_ylim(0, 1.0)
ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_c.set_title("(c) Which 32 of 128 dims", loc="left", pad=10, fontsize=T,
               fontweight="bold")
ax_c.tick_params(labelsize=TK)
note(ax_c, 0.96, 0.96, "position priors (paired\nlow-freq RoPE dims),\n"
     "not “any 32 dims”", ha="right", va="top", color=C_HL, size=AN - 1)

# ---- (d) O4 选择行为 ----
shade_zones(ax_d, bnd4)
for a, lab, col, ls in ARM4:
    cx4, rate = bucket_rate(cells4, a)
    ax_d.plot(cx4, rate, ls, color=col, lw=LW, label=lab, zorder=5)
ax_d.set_xscale("log")
ax_d.set_yscale("log")
ax_d.set_xlim(XLO, XHI)
ax_d.set_ylim(3e-3, 1.1)
ax_d.set_yticks([1e-2, 1e-1, 1])
ax_d.set_xticks([300, 1000, 3000, 10000])
ax_d.set_xlabel("distance d = q_pos − k_pos (tokens)", fontsize=L)
ax_d.set_ylabel("selection rate\n(share of bucket)", fontsize=L)
ax_d.set_title("(d) Selected tokens by distance", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_d.tick_params(labelsize=TK)
ax_d.legend(loc="lower left", fontsize=LG - 1, framealpha=0.92)
note(ax_d, 0.96, 0.92,
     f"far sel per query: oracle {far_avg['oracle']:.0f}\n"
     f"K1=128: {far_avg['tli_k128']:.0f} (far starved)\n"
     f"K1=64: {far_avg['tli_k64']:.0f}; overlap "
     f"{ovl_avg['tli_k128']:.2f}→{ovl_avg['tli_k64']:.2f}",
     ha="right", va="top", color=C_HL, size=AN - 1)
ax_d.text(0.24, 0.965, "near", transform=ax_d.transAxes, ha="center",
          fontsize=AN, color=C_MAIN, fontweight="bold")
ax_d.text(0.60, 0.965, "far", transform=ax_d.transAxes, ha="center",
          fontsize=AN, color=C_HL, fontweight="bold")

fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.17)

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
src = os.path.basename(SRC)
for x, y in zip(cx, dens):
    rows.append(f"O\ta\tmid_distance\tdensity\tmass/token\t{y:.4e}\t{src}\tbins(log-grid)")
for z, v in zip(zones, zmean):
    rows.append(f"O\tb\t{z}\tmean_mass\tfrac\t{v:.4f}\t{src}\tmass_{z}")
rows.append(f"O\tb\tfar_worst_head\tmean_mass\tfrac\t{fw:.4f}\t{src}\tfar_worst_head")
rows.append(f"O\tb\tnear_len_median\tboundary\ttokens\t{near_len_med:.0f}\t{src}\tnear_len")
for a in arms:
    for mm, v in zip(mets_all, arm_val[a]):
        rows.append(f"O\tc\t{a}\t{mm}\tfrac\t{v:.4f}\t{src}\to3.{a}.{mm}.mean")
for a, *_ in ARM4:
    rows.append(f"O\td\t{a}\tfar_sel_per_query\ttokens\t{far_avg[a]:.2f}\t"
                f"e140_o4_select_trace.json\to4.arms.{a}.far_sel")
    rows.append(f"O\td\t{a}\toverlap_recall\tfrac\t{ovl_avg[a]:.4f}\t"
                f"e140_o4_select_trace.json\to4.arms.{a}.overlap_recall")
    cx4, rate = bucket_rate(cells4, a)
    for x, y in zip(cx4, rate):
        if not np.isnan(y):
            rows.append(f"O\td\t{a}\tsel_rate\tfrac\t{y:.4e}\t"
                        f"e140_o4_select_trace.json\to4_bins(log-grid)")
with open(os.path.join(OUT, "O_observation_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported O_observation_figure_data.tsv")
print("DONE")
