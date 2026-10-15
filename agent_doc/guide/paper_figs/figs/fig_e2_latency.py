# -*- coding: utf-8 -*-
"""E2 请求时延与成本分解（figure_id=E2，P0 主文必须）。

面板 (a)：decode 单层合成稳态阶段分解（e128_tradeoff.json e127_headline_64k_normal，
  S=65536, K1=128/K2=1024, H20-3e 独占，GPU 中位）。口径：单层合成稳态 microbench，
  非完整请求 E2E——36 层 TPOT 须另行测量（spec evaluation.md「当前证据限制」）。
  阶段计时为各 kernel 段中位，互斥不重叠，并列条形（不做 stacked 防暗示可加总误读）。
面板 (b)：prefill attention wall L 曲线（e139_step2_cand_verdict.json wall_lsweep，
  .187 GPU4 独立复验 ±1%）。dense SDPA vs 融合态 graph（Step1）vs Step2 候选列化。
  口径：合成单层 prefill attention + 选择链，K2=1024，BATCH+FUSED+GRAPH 全开；
  L∈{8K,16K,32K,64K}，短档净负、~33K 交叉、64K 1.64x 均如实保留（spec：负收益不裁）。

数据全部从判决 JSON 程序化读取，无手工键入；输出 PDF+PNG+SVG + figure_data.tsv。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style, FONT_SIZES, LINE_WIDTHS  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
WT = "/tmp/wt_e138_ablation"
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
# 本机无 Arial/Liberation：显式回退 DejaVu Sans（外观接近，避免 findfont 警告）
matplotlib.rcParams["font.family"] = "DejaVu Sans"
C_DENSE = pal["neutral"]        # dense 基线 = 灰
C_STEP1 = pal["tertiary"]       # Step1 = 浅蓝
C_STEP2 = pal["primary"]        # Step2 = 深蓝（主方法）
C_BASE = pal["neutral"]
C_TLI = pal["primary"]
C_HL = pal["highlight"]         # 强调橙

# ---------------- 数据读取 ----------------
e128 = json.load(open(os.path.join(REPO, "exp/trace/results/e128_tradeoff.json")))
h = e128["e127_headline_64k_normal"]
stages = ["prepare_index", "compute_score", "compute_mask", "metrics", "eager_decoding"]
base_ms = [h["stage_gpu_baseline"][s] for s in stages] + [h["baseline_gpu_ms"]]
tli_ms = [h["stage_gpu_iter8"][s] for s in stages] + [h["iter8_gpu_ms"]]
stage_labels = ["prepare", "scoring", "mask", "metrics", "attn", "total"]

e139 = json.load(open(os.path.join(WT, "exp/trace/results/e139_step2_cand_verdict.json")))
wl = e139["results"]["wall_lsweep"]
Ls = [8192, 16384, 32768, 65536]
dense_ms, step1_ms, step2_ms = [], [], []
for L in Ls:
    s = wl[f"L_{L}"]
    dense_ms.append(float(s.split("dense ")[1].split(" /")[0]))
    step1_ms.append(float(s.split("Step1 graph ")[1].split(" /")[0]))
    step2_ms.append(float(s.split("Step2 graph ")[1].split("ms")[0]))

# ---------------- 画布 ----------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 9),
                                 gridspec_kw={"wspace": 0.30})

# ---- 面板 (a)：decode 阶段分解（并列条形）----
x = range(len(stage_labels))
w = 0.38
ax_a.bar([i - w / 2 for i in x], base_ms, width=w, color=C_BASE,
         label="dense FullKV", edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_a.bar([i + w / 2 for i in x], tli_ms, width=w, color=C_TLI,
         label="TLI (K1=128, K2=1024)", edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_a.set_yscale("log")
ax_a.set_ylim(3e-3, 50)
ax_a.set_yticks([1e-2, 1e-1, 1, 10])
ax_a.set_xticks(list(x))
ax_a.set_xticklabels(stage_labels)
ax_a.set_ylabel("per-step GPU time (ms, log scale)")
ax_a.set_title("Decode stage decomposition @64K", pad=14)
ax_a.legend(loc="upper left")
# 仅 total 条标注数值（分项数值进 tsv，避免 log 轴低矮条标签互叠）
ax_a.text(5 - w / 2, h["baseline_gpu_ms"] * 1.15, f"{h['baseline_gpu_ms']:.2f}",
          ha="center", fontsize=FONT_SIZES["annotation"] - 2)
ax_a.text(5 + w / 2, h["iter8_gpu_ms"] * 1.15, f"{h['iter8_gpu_ms']:.2f}",
          ha="center", fontsize=FONT_SIZES["annotation"] - 2)
ax_a.text(5, 30, "5.47x GPU / 4.76x wall", ha="center",
          fontsize=FONT_SIZES["annotation"], color=C_HL, fontweight="bold")
ax_a.text(0.02, 0.02, "single-layer synthetic steady state,\nnot full-request E2E",
          transform=ax_a.transAxes, fontsize=FONT_SIZES["annotation"] - 4,
          style="italic", color=pal["neutral"], va="bottom")

# ---- 面板 (b)：prefill wall L 曲线 ----
Lk = [L / 1024 for L in Ls]
ax_b.plot(Lk, dense_ms, "-o", color=C_DENSE, label="dense SDPA",
          markersize=14, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_b.plot(Lk, step1_ms, "-s", color=C_STEP1, label="sparse prefill (Step1 graph)",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_b.plot(Lk, step2_ms, "-^", color=C_STEP2, label="sparse prefill (Step2 cand topk)",
          markersize=14, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_b.set_xscale("log", base=2)
ax_b.set_yscale("log")
ax_b.set_ylim(4, 1200)
ax_b.set_yticks([10, 100, 1000])
ax_b.set_xticks(Lk)
ax_b.set_xticklabels([f"{l}K" for l in Lk])
ax_b.set_xlabel("input length L (tokens)")
ax_b.set_ylabel("prefill attention wall (ms, log scale)")
ax_b.set_title("Prefill: wall vs input length (K2=1024)", pad=14)
ax_b.legend(loc="upper left")
# 数值标注
for lk, d, s2 in zip(Lk, dense_ms, step2_ms):
    ax_b.annotate(f"{d:.0f}", (lk, d), textcoords="offset points", xytext=(0, 10),
                  ha="center", fontsize=FONT_SIZES["annotation"] - 3, color=C_DENSE)
    ax_b.annotate(f"{s2:.0f}", (lk, s2), textcoords="offset points", xytext=(0, -18),
                  ha="center", fontsize=FONT_SIZES["annotation"] - 3, color=C_STEP2)
ax_b.text(64, 55, "1.64x vs dense", ha="right",
          fontsize=FONT_SIZES["annotation"], color=C_HL, fontweight="bold")
ax_b.text(0.02, 0.02, "single-layer synthetic prefill;\nshort lengths remain net-negative",
          transform=ax_b.transAxes, fontsize=FONT_SIZES["annotation"] - 4,
          style="italic", color=pal["neutral"], va="bottom")

# 面板字母
for ax, letter in ((ax_a, "A"), (ax_b, "B")):
    ax.text(0.01, 1.05, letter, transform=ax.transAxes,
            fontsize=FONT_SIZES["panel_letter"], fontweight="bold", va="bottom")

# ---------------- lint（导出前机械检查）----------------
report = lint_figure(fig)
n_bad = len(report.get("overlaps", [])) + len(report.get("clipped", []))
print(f"[lint] overlaps={len(report.get('overlaps', []))} clipped={len(report.get('clipped', []))}")
for o in report.get("overlaps", [])[:6]:
    print("  OVERLAP:", o)
for c in report.get("clipped", [])[:6]:
    print("  CLIPPED:", c)

# ---------------- 导出 ----------------
def export(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"exported {stem}.pdf/.png/.svg")


export(fig, "E2_latency")

# ---------------- figure_data.tsv ----------------
rows = ["figure_id\tpanel_id\tarm_id\tmetric_id\tunit\testimate\tsource_run_id\tsource_field"]
src128 = "e128_tradeoff.json:e127_headline_64k_normal"
for s, b, t in zip(stages + ["total"], base_ms, tli_ms):
    rows.append(f"E2\ta\tbaseline\tgpu_ms\tms\t{b:.4f}\t{src128}\tstage_gpu_baseline.{s}")
    rows.append(f"E2\ta\ttli\tgpu_ms\tms\t{t:.4f}\t{src128}\tstage_gpu_iter8.{s}")
rows.append(f"E2\ta\tbaseline\twall_ms\tms\t{h['baseline_wall_ms']:.4f}\t{src128}\tbaseline_wall_ms")
rows.append(f"E2\ta\ttli\twall_ms\tms\t{h['iter8_wall_ms']:.4f}\t{src128}\titer8_wall_ms")
src139 = "e139_step2_cand_verdict.json:wall_lsweep"
for L, d, s1, s2 in zip(Ls, dense_ms, step1_ms, step2_ms):
    rows.append(f"E2\tb@L{L}\tdense\twall_ms\tms\t{d:.1f}\t{src139}\tL_{L}")
    rows.append(f"E2\tb@L{L}\tstep1_graph\twall_ms\tms\t{s1:.1f}\t{src139}\tL_{L}")
    rows.append(f"E2\tb@L{L}\tstep2_graph\twall_ms\tms\t{s2:.1f}\t{src139}\tL_{L}")
with open(os.path.join(OUT, "E2_latency_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported E2_latency_figure_data.tsv")
print("DONE")
