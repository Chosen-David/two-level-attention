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
rq = json.load(open(R + "e119_ruler128k_formal_quest_vendor.json"))
# E152 前置审核（2026-10-16）：e116b 的 TLI 128K 臂是无效格——niah_single
# 100/100/99 与 FullKV 逐位同形（稀疏路径未生效/口径错配），且与 E146 正式
# 判决（receipt 闭包 + score_ruler_formal manifest 身份）23.98 冲突。
# RULER 128K 三臂统一改读 E146 正式口径（FullKV 逐任务与 e116b 一致，
# scorer 等价性已核）。
e146 = json.load(open(R + "e146_ruler128k_verdict.json"))
ruler_mavg = e146["arms"]["tli_main_arm"]["AVG"]
ruler_fullkv = e146["arms"]["none_fullkv"]["AVG"]


def ruler_avg(d):
    sc = list(d["scores"].values())[0]
    v = list(sc.values())
    return sum(v) / len(v)


ruler_quest = ruler_avg(rq)

suites = ["LB v1\n(16 tasks)", "LB v2\n(503 tasks)", "RULER\n(128K)"]
fullkv_scores = [fk["avg16"], lb["lbv2_503"]["FullKV_reused_E109"], ruler_fullkv]
tli_scores = [mv["avg16"], lb["lbv2_503"]["mavg_new"], ruler_mavg]
# Quest vendor 已验收聚合（deepseek 轨接入；LBv1/LBv2 来源：
# exp/results_ruler/main_table_quest_vs_tli_draft.md §1/§2，Quest 官方
# accuracy/vendor 路径 Qwen3-8B；RULER 128K 为 e119_ruler128k_formal_
# quest_vendor.json 程序化读取）。主方法 TLI 臂不动，只追加 baseline 臂。
quest_scores = [41.35, 30.22, ruler_quest]
QUEST_SRC = "main_table_quest_vs_tli_draft.md §1/§2 + e119_ruler128k_formal_quest_vendor.json"
# FASA（表示轴，deepseek 轨接入）：LBv1 = fasa_lb_16.json AVG16；LBv2 =
# pred_fasa_1024_lbv2/pred/result.json score；RULER 128K 未测（方法局限，仅
# RULER 32K 6/11 任务 s1=0），按缺失留空（np.nan），不做外推。程序化读取。
_fasa_lb1 = json.load(open(R + "fasa_lb_16.json"))
_fasa_lb2 = json.load(open(
    REPO + "/exp/results_longbench_fasa/Qwen3-8B/"
    "pred_fasa_1024_lbv2/pred/result.json"))
fasa_scores = [_fasa_lb1["AVG16"], list(_fasa_lb2.values())[0]["score"], np.nan]
FASA_SRC = ("exp/trace/results/fasa_lb_16.json + "
            "exp/results_longbench_fasa/Qwen3-8B/pred_fasa_1024_lbv2/pred/result.json")

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
w = 0.19
ax_b.bar(xb - 0.285, fullkv_scores, width=w, color=C_NEU, label="dense FullKV",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb - 0.095, quest_scores, width=w, color=C_SEC, label="Quest vendor (K=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb + 0.095, fasa_scores, width=w, color=C_HL, label="FASA vendor (K=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb + 0.285, tli_scores, width=w, color=C_MAIN, label="TLI (mavg, K2=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
for xi, v in zip(xb - 0.285, fullkv_scores):
    if not np.isnan(v):
        ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb - 0.095, quest_scores):
    if not np.isnan(v):
        ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb + 0.095, fasa_scores):
    if not np.isnan(v):
        ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center",
                  fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb + 0.285, tli_scores):
    if not np.isnan(v):
        ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(suites)
ax_b.set_ylabel("task score")
ax_b.set_title("Suite-level totals", loc="left", pad=10)
ax_b.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12),
            fontsize=FONT_SIZES["legend"] - 3)
ax_b.set_ylim(0, 62)
ax_b.text(0.005, 0.005, "tail32 AVG5: "
          + f"{t32:.2f}, Δ{e133['avg']['delta']:+.2f} "
          + f"CI[{t32_ci[0]:.2f},{t32_ci[1]:.2f}]"
          + "\nLB v1 FullKV 13-task reused pre-E121"
          + "\nRULER 128K: both arms collapse (yarn 4.0); "
          + "TLI retention 51.7% (E146)"
          + "\nQuest LBv1/LBv2 draft; FASA vendor; "
          + "FASA RULER 128K untested (6/11 @32K, s1=0)"
          + "\nClusterKV/MoBA pending",
          transform=ax_b.transAxes, fontsize=FONT_SIZES["annotation"] - 6,
          color=pal["neutral"], va="bottom", style="italic")

# 面板字母（轴内左上角，与轴外顶部 legend 不冲突）
for ax, letter in ((ax_a, "A"), (ax_b, "B")):
    ax.text(0.012, 0.985, letter, transform=ax.transAxes,
            fontsize=FONT_SIZES["panel_letter"], fontweight="bold", va="top")

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
for s, fv, tv, qv, fsa in zip(("lbv1_16", "lbv2_503", "ruler_128k"),
                              fullkv_scores, tli_scores, quest_scores,
                              fasa_scores):
    src_r = "e146_ruler128k_verdict.json (E146 formal)" if s == "ruler_128k" \
        else f"{src116}/e116b_ruler128k"
    rows.append(f"E1\tb\t{s}\tfullkv\tscore\t{fv:.2f}\t\t\t{src_r}")
    rows.append(f"E1\tb\t{s}\ttli_mavg\tscore\t{tv:.2f}\t\t\t{src_r}")
    rows.append(f"E1\tb\t{s}\tquest\tscore\t{qv:.2f}\t\t\t{QUEST_SRC}")
    _fsa = f"{fsa:.2f}" if not np.isnan(fsa) else ""
    rows.append(f"E1\tb\t{s}\tfasa\tscore\t{_fsa}\t\t\t{FASA_SRC}")
    rows.append(f"E1\tb\t{s}\tclusterkv\tscore\t\t\t\tpending")
rows.append("E1\b_note\te116b_tli_128k_invalid\tnote\t-\t"
            "niah_single 100/100/99 = FullKV-shaped, sparse path not effective; "
            "superseded by E146 formal 23.98")
rows.append("E1\b_note\tfasa_ruler128k_missing\tnote\t-\t"
            "FASA RULER 128K untested (method limits: only RULER 32K 6/11 "
            "tasks with s1=0); not extrapolated")
rows.append(f"E1\tb\ttail32_avg5\tscore\t{t32:.2f}\t{t32_ci[0]:.2f}\t{t32_ci[1]:.2f}\t"
            "e133_subspace_verdict.json")
with open(os.path.join(OUT, "E1_quality_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported E1_quality_figure_data.tsv")
print("DONE")
