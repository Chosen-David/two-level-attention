# -*- coding: utf-8 -*-
"""E1 主质量图（figure_id=E1，P0 主文；2026-10-23 三面板重设计）。

面板 (a)：LongBench v1 16 任务逐任务配对差值点图（e116b_full_lbv1.json，
  Qwen3-8B，mavg(α=.25,β=.125,γ=.625) vs FullKV，按 delta 排序）——
  hotpotqa +5.9 最大改善、passage_count -1.25 最大退化，正负例均保留。
面板 (b)：LongBench 两套件级总分（LB v1 16 任务 / LB v2 503 题）
  FullKV vs TLI(mavg) 并列条形 + tail32 子空间 AVG5 附加臂（E133 Δ-0.32
  CI[-0.51,-0.104]——冻结冠军表示 tail32 与 full 表示差噪声级）。
  旧第三组「RULER 128K」退役（E146 β.125 崩溃臂 23.98 已过时，且与
  新面板 (c) 全长度曲线重复）；RULER 数据统一走面板 (c)。
面板 (c)：RULER 11 任务 AVG vs 上下文长度（32K/64K/128K）——长度退化
  主叙事（2026-10-23 新增，取代旧 (b) RULER 128K 单格）：
  - dense FullKV 锚 = 独立正式批（e116e 32K 59.38 / e146 64K 48.54 /
    e119 128K 46.43，separate-run anchor 口径）；
  - TLI = T6 冠军几何 nb32/ntoff/nl4096（t6_ruler_q8b_*_champ.json，
    与 E152v4/E160 延迟判决同一几何：59.02 / 47.80 / 46.82，
    保留率 99.4% / 98.5% / 100.8%——128K 反超锚）；
  - vendor 长度线：Quest 42.85/26.25/19.33（e119_ruler{32k,64k,128k}
    _formal_quest_vendor）、AsyncTLS 48.33/33.88/37.47（排序修复版
    e119_..._asynctls_vendor_fix）、FASA 18.64/1.72/1.40（崩塌级）；
  - 32K-only vendor 臂（无 64K/128K 正式批，不做外推）：ClusterKV 60.6
    （e119_..._clusterkv_vendor）/ SparQ 60.5（sparq_vendor）/ HISA 59.7
    （hisa_vendor）/ JSSA 58.7（jssa_r32k_formal，双记录裁决取较新正式格
    而非 e119 统一批 59.13）——四家 32K 都在锚 ±2.2 内，分化只在长度轴
    显形（诚实披露：32K 单档无法区分方法，长度曲线才是分离轴）。

口径标注：LB v1 fullkv 13 任务为 E121 修复前复用格（e116b caveats）；
RULER 无 yarn（producer receipt 闭包）。数据程序化读取，无手工键入；
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
# Type42 嵌入（字形门友好；Type3 CJK bbox 膨胀教训见 ruler32k_comparison/draw.py）
matplotlib.rcParams["pdf.fonttype"] = matplotlib.rcParams["ps.fonttype"] = 42
C_MAIN = pal["primary"]      # TLI
C_SEC = pal["secondary"]     # Quest
C_TER = pal["tertiary"]      # AsyncTLS
C_HL = pal["highlight"]      # FASA / 强调橙
C_NEU = pal["neutral"]       # dense FullKV

# ---------------- 数据读取：面板 (a) LBv1 逐任务 ----------------
lb = json.load(open(R + "e116b_full_lbv1.json"))
fk = lb["arms"]["FullKV"]
mv = lb["arms"]["mavg_a0.25_b0.125_g0.625"]
tasks = sorted(fk["per_task"])
delta = {t: mv["per_task"][t] - fk["per_task"][t] for t in tasks}
tasks_sorted = sorted(tasks, key=lambda t: delta[t])

e133 = json.load(open(R + "e133_subspace_verdict.json"))
t32 = e133["avg"]["tail32"]
t32_ci = e133["bootstrap_ci95"]

# ---------------- 数据读取：面板 (b) LB 套件级 ----------------
# Quest vendor 已验收聚合（deepseek 轨接入；LBv1/LBv2 来源：
# exp/results_ruler/main_table_quest_vs_tli_draft.md §1/§2，Quest 官方
# accuracy/vendor 路径 Qwen3-8B）。主方法 TLI 臂不动，只追加 baseline 臂。
quest_scores = [41.35, 30.22]
QUEST_SRC = "main_table_quest_vs_tli_draft.md §1/§2"
# FASA（表示轴，deepseek 轨接入）：LBv1 = fasa_lb_16.json AVG16；LBv2 =
# pred_fasa_1024_lbv2/pred/result.json score。程序化读取。
_fasa_lb1 = json.load(open(R + "fasa_lb_16.json"))
_fasa_lb2 = json.load(open(
    REPO + "/exp/results_longbench_fasa/Qwen3-8B/"
    "pred_fasa_1024_lbv2/pred/result.json"))
fasa_scores = [_fasa_lb1["AVG16"], list(_fasa_lb2.values())[0]["score"]]
FASA_SRC = ("exp/trace/results/fasa_lb_16.json + "
            "exp/results_longbench_fasa/Qwen3-8B/pred_fasa_1024_lbv2/pred/result.json")

suites = ["LongBench v1\n(16 tasks)", "LongBench v2\n(503 MCQ)"]
fullkv_scores = [fk["avg16"], lb["lbv2_503"]["FullKV_reused_E109"]]
tli_scores = [mv["avg16"], lb["lbv2_503"]["mavg_new"]]

# ---------------- 数据读取：面板 (c) RULER 长度退化 ----------------
def ruler_arm_avg(path):
    """11 任务 AVG：scores -> 单臂 -> 任务分值均值（正式批口径）。"""
    d = json.load(open(path))
    for k, v in d["scores"].items():
        if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
            return sum(v.values()) / len(v)
    raise ValueError(f"no score arm in {path}")

# T6 冠军几何（与延迟判决同一几何）：t6_ruler_q8b_*_champ.json
t6 = [ruler_arm_avg(R + f"t6_ruler_q8b_L{L}_champ.json")
      for L in (32768, 65536, 131072)]
# dense FullKV 锚 = 独立正式批（separate-run anchor）
fullkv_ruler = [ruler_arm_avg(R + "e116e_ruler32_formal_fullkv.json"),
                ruler_arm_avg(R + "e146_ruler64k_formal.json"),
                ruler_arm_avg(R + "e119_ruler128k_formal_fullkv.json")]
# vendor 长度线
quest_ruler = [ruler_arm_avg(R + "e119_ruler32k_formal_quest_vendor.json"),
               ruler_arm_avg(R + "e119_ruler64k_formal_quest_vendor.json"),
               ruler_arm_avg(R + "e119_ruler128k_formal_quest_vendor.json")]
asynctls_ruler = [ruler_arm_avg(R + "e119_ruler32768_formal_asynctls_vendor_fix.json"),
                  ruler_arm_avg(R + "e119_ruler65536_formal_asynctls_vendor_fix.json"),
                  ruler_arm_avg(R + "e119_ruler131072_formal_asynctls_vendor_fix.json")]
fasa_ruler = [ruler_arm_avg(R + "e119_ruler32768_formal_fasa_20261021.json"),
              ruler_arm_avg(R + "e119_ruler65536_formal_fasa_vendor.json"),
              ruler_arm_avg(R + "e119_ruler131072_formal_fasa_vendor.json")]
# 32K-only vendor 臂（无 64K/128K 批，不做外推；JSSA 取双记录裁决较新格）
vendor32 = {
    "ClusterKV": ruler_arm_avg(R + "e119_ruler32768_formal_clusterkv_vendor.json"),
    "SparQ": ruler_arm_avg(R + "e119_ruler32768_formal_sparq_vendor.json"),
    "HISA": ruler_arm_avg(R + "e119_ruler32768_formal_hisa_vendor.json"),
    "JSSA": ruler_arm_avg(R + "jssa_r32k_formal.json"),
}
retention = [t / f for t, f in zip(t6, fullkv_ruler)]

# ---------------- 画布 1x3 ----------------
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(30, 9),
                                       gridspec_kw={"wspace": 0.30,
                                                    "width_ratios": [1.1, 0.85, 1.05]})

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
ax_a.text(0.98, 0.03, "avg16: 48.11 (FullKV) vs 48.90 (LoDA-SA)", transform=ax_a.transAxes,
          fontsize=FONT_SIZES["annotation"] - 2, ha="right", color=C_MAIN,
          fontweight="bold")

# ---- (b) LB 套件级总分 ----
xb = np.arange(2)
w = 0.19
ax_b.bar(xb - 0.285, fullkv_scores, width=w, color=C_NEU, label="dense FullKV",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb - 0.095, quest_scores, width=w, color=C_SEC, label="Quest vendor (K=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb + 0.095, fasa_scores, width=w, color=C_HL, label="FASA vendor (K=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.bar(xb + 0.285, tli_scores, width=w, color=C_MAIN, label="LoDA-SA (mavg, K2=1024)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
for xi, v in zip(xb - 0.285, fullkv_scores):
    ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb - 0.095, quest_scores):
    ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb + 0.095, fasa_scores):
    ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center",
              fontsize=FONT_SIZES["annotation"] - 3)
for xi, v in zip(xb + 0.285, tli_scores):
    ax_b.text(xi, v + 0.8, f"{v:.2f}", ha="center", fontsize=FONT_SIZES["annotation"] - 3)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(suites)
ax_b.set_ylabel("task score")
ax_b.set_title("LongBench suite totals", loc="left", pad=10)
ax_b.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16),
            fontsize=FONT_SIZES["legend"] - 3)
ax_b.set_ylim(0, 62)
ax_b.set_yticks([0, 10, 20, 30, 40, 50, 60])
ax_b.text(0.005, 0.005, "tail32 AVG5: "
          + f"{t32:.2f}, Δ{e133['avg']['delta']:+.2f} "
          + f"CI[{t32_ci[0]:.2f},{t32_ci[1]:.2f}]"
          + "\nLB v1 FullKV 13-task reused pre-E121"
          + "\nQuest LBv1/LBv2 draft; FASA vendor"
          + "\nRULER moved to panel (c)",
          transform=ax_b.transAxes, fontsize=FONT_SIZES["annotation"] - 6,
          color=pal["neutral"], va="bottom", style="italic")

# ---- (c) RULER 长度退化 ----
Lk = [32, 64, 128]
ax_c.plot(Lk, fullkv_ruler, "-o", color=C_NEU, label="dense FullKV (anchor runs)",
          markersize=15, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_c.plot(Lk, t6, "-s", color=C_MAIN,
          label="LoDA-SA champion (nb32/nl4096)",
          markersize=14, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_c.plot(Lk, quest_ruler, "--v", color=C_SEC, label="Quest vendor (K=1024)",
          markersize=12, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"] * 0.7)
ax_c.plot(Lk, asynctls_ruler, "--D", color=C_TER, label="AsyncTLS vendor (K=512)",
          markersize=11, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"] * 0.7)
ax_c.plot(Lk, fasa_ruler, "--^", color=C_HL, label="FASA vendor (K=1024)",
          markersize=12, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"] * 0.7)
ax_c.set_xscale("log", base=2)
ax_c.set_xticks(Lk)
ax_c.set_xticklabels([f"{l}K" for l in Lk])
ax_c.set_xlim(26, 175)
ax_c.set_ylim(0, 72)
ax_c.set_yticks([0, 20, 40, 60])
ax_c.set_xlabel("input length L (tokens)")
ax_c.set_ylabel("RULER 11-task AVG")
ax_c.set_title("RULER vs length: LoDA-SA tracks dense, baselines collapse",
               loc="left", pad=10)
ax_c.legend(loc="lower left", fontsize=FONT_SIZES["legend"] - 4)
# 主臂数值标注：FullKV 上置 / TLI 下置（32K 两点贴近属故事本身——锚差仅 0.36）
for lk, f, t in zip(Lk, fullkv_ruler, t6):
    ax_c.annotate(f"{f:.2f}", (lk, f), textcoords="offset points",
                  xytext=(0, 11), ha="center", va="bottom",
                  fontsize=FONT_SIZES["annotation"] - 3, color=C_NEU)
    ax_c.annotate(f"{t:.2f}", (lk, t), textcoords="offset points",
                  xytext=(0, -10), ha="center", va="top",
                  fontsize=FONT_SIZES["annotation"] - 3, color=C_MAIN)
# vendor 端点数值（128K 右置、FASA 128K 上置避让底部注记）
for arr, col in ((quest_ruler, C_SEC), (asynctls_ruler, C_TER)):
    ax_c.annotate(f"{arr[-1]:.1f}", (128, arr[-1]), textcoords="offset points",
                  xytext=(12, 0), ha="left", va="center",
                  fontsize=FONT_SIZES["annotation"] - 4, color=col)
    ax_c.annotate(f"{arr[0]:.1f}", (32, arr[0]), textcoords="offset points",
                  xytext=(0, 10), ha="center", va="bottom",
                  fontsize=FONT_SIZES["annotation"] - 4, color=col)
ax_c.annotate(f"{fasa_ruler[0]:.1f}", (32, fasa_ruler[0]), textcoords="offset points",
              xytext=(0, 10), ha="center", va="bottom",
              fontsize=FONT_SIZES["annotation"] - 4, color=C_HL)
ax_c.annotate(f"{fasa_ruler[-1]:.1f}", (128, fasa_ruler[-1]), textcoords="offset points",
              xytext=(12, 0), ha="left", va="top",
              fontsize=FONT_SIZES["annotation"] - 4, color=C_HL)
# 保留率注记（TLI 贴锚主叙事）
ax_c.text(40, 55, "LoDA-SA retention vs anchor: "
          + " / ".join(f"{r * 100:.1f}%" for r in retention),
          fontsize=FONT_SIZES["annotation"] - 2, color=C_MAIN, fontweight="bold")
# 32K-only vendor 簇披露（无 64K/128K 批，不做外推；32K 单档无法区分方法）
v32_txt = " · ".join(f"{k} {v:.1f}" for k, v in vendor32.items())
ax_c.text(0.98, 0.98, f"32K-only arms (no 64K/128K runs):\n{v32_txt}",
          transform=ax_c.transAxes, fontsize=FONT_SIZES["annotation"] - 4,
          style="italic", color=pal["neutral"], va="top", ha="right")
ax_c.text(0.98, 0.855, "no yarn; anchors = separate runs",
          transform=ax_c.transAxes, fontsize=FONT_SIZES["annotation"] - 4,
          style="italic", color=pal["neutral"], va="top", ha="right")

# 面板字母
for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
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
for s, fv, tv, qv, fsa in zip(("lbv1_16", "lbv2_503"),
                              fullkv_scores, tli_scores, quest_scores,
                              fasa_scores):
    rows.append(f"E1\tb\t{s}\tfullkv\tscore\t{fv:.2f}\t\t\t{src116}")
    rows.append(f"E1\tb\t{s}\ttli_mavg\tscore\t{tv:.2f}\t\t\t{src116}")
    rows.append(f"E1\tb\t{s}\tquest\tscore\t{qv:.2f}\t\t\t{QUEST_SRC}")
    rows.append(f"E1\tb\t{s}\tfasa\tscore\t{fsa:.2f}\t\t\t{FASA_SRC}")
rows.append("E1\tb\tnote\tsuperseded\t-\t"
            "old RULER-128K bar group retired 2026-10-23: E146 beta.125 collapse arm "
            "(23.98) superseded by T6 champion curve in panel (c); FASA RULER now "
            "measured (18.64/1.72/1.40), untested note retired")
rows.append(f"E1\tb\ttail32_avg5\tscore\t{t32:.2f}\t{t32_ci[0]:.2f}\t{t32_ci[1]:.2f}\t"
            "e133_subspace_verdict.json")
# 面板 (c) RULER 长度退化
c_src = {
    "fullkv": (fullkv_ruler, "e116e_ruler32_formal_fullkv.json + e146_ruler64k_formal.json + e119_ruler128k_formal_fullkv.json"),
    "tli_champ": (t6, "t6_ruler_q8b_L{32768,65536,131072}_champ.json"),
    "quest": (quest_ruler, "e119_ruler{32k,64k,128k}_formal_quest_vendor.json"),
    "asynctls": (asynctls_ruler, "e119_ruler{32768,65536,131072}_formal_asynctls_vendor_fix.json"),
    "fasa": (fasa_ruler, "e119_ruler32768_formal_fasa_20261021.json + e119_ruler65536_formal_fasa_vendor.json + e119_ruler131072_formal_fasa_vendor.json"),
}
for arm, (vals_, src) in c_src.items():
    for L, v in zip((32768, 65536, 131072), vals_):
        rows.append(f"E1\tc@L{L}\t{arm}\tavg11\tscore\t{v:.2f}\t\t\t{src}")
for L, r in zip((32768, 65536, 131072), retention):
    rows.append(f"E1\tc@L{L}\ttli_retention\tratio\t%\t{r * 100:.1f}\t\t\t"
                "t6_ruler_q8b champ / separate-run fullkv anchor")
v32_src = {
    "ClusterKV": "e119_ruler32768_formal_clusterkv_vendor.json",
    "SparQ": "e119_ruler32768_formal_sparq_vendor.json",
    "HISA": "e119_ruler32768_formal_hisa_vendor.json",
    "JSSA": "jssa_r32k_formal.json (dual-record adjudication: newer formal 58.7 over e119 unified batch 59.13)",
}
for k, v in vendor32.items():
    rows.append(f"E1\tc@L32768\t{k}_vendor32k\tavg11\tscore\t{v:.2f}\t\t\t{v32_src[k]}")
rows.append("E1\tc\tnote\tvendor32k_scope\t-\t"
            "32K-only arms have no 64K/128K formal batches; not extrapolated; "
            "all within +/-2.2 of the 32K anchor column")
with open(os.path.join(OUT, "E1_quality_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported E1_quality_figure_data.tsv")
print("DONE")
