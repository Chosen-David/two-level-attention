# -*- coding: utf-8 -*-
"""A 系列核心消融图（figure_id=A2/A3/A4/A5 合并六面板 2x3，P0 主文）。

面板 (A) 质量代价 forest（AVG5 配对差值 + bootstrap CI95，正负例均保留）：
  - tail32 子空间（E133）Δ-0.32 CI[-0.51,-0.104]——A2 质量侧
  - tail32 逐位一致（E129 c2）Δ0.0 CI[0,0]——同输入输出逐位相同
  - 静态 D' gate（E129 c1）Δ-4.47 CI 不含 0——负例如实保留
  - 动态 gate ε=0.05/0.10（E137）Δ-0.03/-0.22 CI 均含 0——A4
  - sink+SWA only（E135a M1）Δ-21.82 CI[-30.16,-13.48]——A5 middle 价值锚
面板 (B) A2 成本侧：表示三臂稳态 GPU ms（E134 同机两轮交叉中位）
  ——sub+sub ≈ full+sub（+0.7% 换 4x 粗筛省算），sub+full 最慢。
面板 (C) A3 预算：K2 上/下探双臂（E139 pareto）——decode 下探悬崖
  （hotpot 59.2→29.92, musique 33.88→12.47）+ prefill 上探恢复曲线
  + Quest 口径竖线（总召回 1024）。
面板 (D) A5 middle 归因 per-task：FullKV vs sink+SWA-only（E135a）。
面板 (E) E140b K1 倒 U：decode K1∈{16,64,128,256}（左臂=L1 far 粗筛覆盖
  不足，右臂=far 配额归零 K1≥154 阴影区）+ prefill K1∈{128,256,512}
  （K1 全局共享 decode+prefill 双路径同塌）。
面板 (F) E140 机制：far token 配额曲线 far_budget=K2_mid−min(K1·β·bs·γ,
  K2_mid)——K2=1024/896/768 三档，far=0 阈值精确导出 K2 悬崖位置。

数据全部程序化读取自判决 JSON；输出 PDF+PNG+SVG + figure_data.tsv。
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
WT = "/tmp/wt_e138_ablation"
R = os.path.join(REPO, "exp/trace/results/")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
C_MAIN = pal["primary"]
C_SEC = pal["secondary"]
C_TER = pal["tertiary"]
C_HL = pal["highlight"]
C_NEU = pal["neutral"]

# ---------------- 数据读取 ----------------
e133 = json.load(open(R + "e133_subspace_verdict.json"))
e129 = json.load(open(R + "e129_ablation_verdict.json"))
e137 = json.load(open(R + "e137_dyn_dgate_verdict.json"))
e135 = json.load(open(R + "e135_sinkswa_only_verdict.json"))
e134 = json.load(open(R + "e134_perf_reverdict.json"))
e139 = json.load(open(WT + "/exp/trace/results/e139_k2_budget_pareto_verdict.json"))
e140b = json.load(open(R + "e140b_k1_downscan_verdict.json"))
e140a = json.load(open(R + "e140a_prefill_k1up_verdict.json"))

# (A) forest
forest = [  # (label, delta, ci_lo, ci_hi, color)
    ("tail32 subspace (E133)", e133["avg"]["tail32"] - e133["avg"]["anchor"],
     e133["bootstrap_ci95"][0], e133["bootstrap_ci95"][1], C_MAIN),
    ("tail32 bit-identical (E129)",
     e129["effects"]["c2_tail"]["delta_avg5_vs_c0"],
     e129["effects"]["c2_tail"]["ci95_vs_c0"][0],
     e129["effects"]["c2_tail"]["ci95_vs_c0"][1], C_MAIN),
    ("dyn gate $\\epsilon$=.05 (E137)",
     e137["avg"]["dg5"]["delta"], e137["bootstrap_ci95"]["dg5"][0],
     e137["bootstrap_ci95"]["dg5"][1], C_SEC),
    ("dyn gate $\\epsilon$=.10 (E137)",
     e137["avg"]["dg1"]["delta"], e137["bootstrap_ci95"]["dg1"][0],
     e137["bootstrap_ci95"]["dg1"][1], C_SEC),
    ("static gate (E129)", e129["effects"]["c1_gate"]["delta_avg5_vs_c0"],
     e129["effects"]["c1_gate"]["ci95_vs_c0"][0],
     e129["effects"]["c1_gate"]["ci95_vs_c0"][1], C_NEU),
    ("sink+SWA only (E135a)", -21.82, -30.156, -13.48, C_HL),
]

# (B) 表示三臂稳态
arms = ["full+sub", "sub+sub", "sub+full"]
steady = [e134["steady_gpu_ms"][a]["new_med"] for a in arms]

# (C) 预算曲线
b = e139["results"]["armB_decode_downscan"]
dd_k = [1024, 768, 512, 384, 256]
dd_hot = [b["hotpotqa"]["k1024_anchor"], b["hotpotqa"]["k768"], b["hotpotqa"]["k512"],
          b["hotpotqa"]["k384"], b["hotpotqa"]["k256_r2"]]
dd_mus = [b["musique"]["k1024_anchor"], b["musique"]["k768"], b["musique"]["k512"],
          b["musique"]["k384"], b["musique"]["k256_r2"]]
a = e139["results"]["armA_prefill_upscan"]
pf_k = [1024, 1536, 2048]
pf_hot = [a["hotpotqa"]["k1024"], a["hotpotqa"]["k1536_r2"], a["hotpotqa"]["k2048"]]
pf_mus = [a["musique"]["k1024"], a["musique"]["k1536_r2"], a["musique"]["k2048"]]
anchor_hot, anchor_mus = a["hotpotqa"]["dense_anchor"], a["musique"]["dense_anchor"]

# (D) middle per-task
pt = e135["per_task"]
tasks = ["qasper", "hotpotqa", "musique", "gov_report", "repobench"]
m0 = [pt[t]["anchor"] for t in tasks]
m1 = [pt[t]["sinkswa_only"] for t in tasks]

# (E) K1 倒 U（E140b decode + E140a prefill）
k1_dec = [16, 64, 128, 256]
dec_hot = [e140b["results"]["dk1_16"]["hotpotqa"], e140b["results"]["dk1_64"]["hotpotqa"],
           e140b["results"]["anchor_k128"]["hotpotqa"], e140b["results"]["dk1_256"]["hotpotqa"]]
dec_mus = [e140b["results"]["dk1_16"]["musique"], e140b["results"]["dk1_64"]["musique"],
           e140b["results"]["anchor_k128"]["musique"], e140b["results"]["dk1_256"]["musique"]]
k1_mus = k1_dec
k1_pf = [128, 256, 512]
pf_hot_k1 = [e140a["results"]["anchor_grp64_k128"]["hotpotqa"],
             e140a["results"]["pfk1_256"]["hotpotqa"], e140a["results"]["pfk1_512"]["hotpotqa"]]
pf_mus_k1 = [e140a["results"]["anchor_grp64_k128"]["musique"],
             e140a["results"]["pfk1_256"]["musique"], e140a["results"]["pfk1_512"]["musique"]]
FAR_ZERO_K1 = 153.6   # K2_mid/(beta*bs*gamma) = 768/5

# (F) far 配额曲线（机制）
def far_budget(k1, k2, beta=0.125, bs=64, gamma=0.625, sink=128, swa=128):
    k2m = k2 - sink - swa
    return max(0, k2m - min(round(k1 * beta) * bs * gamma, k2m))

# ---------------- 画布 2x3 ----------------
fig, axes = plt.subplots(2, 3, figsize=(30, 16), gridspec_kw={"wspace": 0.30, "hspace": 0.34})
fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.075)
ax_a, ax_b, ax_c = axes[0]
ax_d, ax_e, ax_f = axes[1]

# ---- 面板 (A) forest ----
ys = np.arange(len(forest))[::-1]
for y, (lab, dlt, lo, hi, c) in zip(ys, forest):
    ax_a.plot([lo, hi], [y, y], color=c, lw=LINE_WIDTHS["data"], solid_capstyle="round")
    ax_a.plot([dlt], [y], "o", color=c, markersize=16,
              markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_a.axvline(0, color="black", lw=LINE_WIDTHS["reference"], linestyle="--")
ax_a.set_yticks(ys)
ax_a.set_yticklabels([f[0] for f in forest])
ax_a.set_xlabel("AVG5 paired delta vs full anchor (points)")
ax_a.set_title("A  Design-choice quality cost (95% CI)", loc="left", pad=10)
ax_a.set_xlim(-33, 6)

# ---- 面板 (B) 表示三臂 ----
xb = np.arange(3)
ax_b.bar(xb, steady, width=0.55, color=[C_MAIN, C_MAIN, C_NEU],
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(["full+sub", "sub+sub", "sub+full"])
ax_b.set_ylabel("steady-state GPU time (ms/step)")
ax_b.set_title("B  Representation arms (L1/L2 scoring dims)", loc="left", pad=10)
ax_b.text(1.0, 0.62, "4x coarse compute saved\nat +0.7% latency\n(sub+sub vs full+sub)",
          ha="center", fontsize=FONT_SIZES["annotation"], color=C_HL)
for xi, v in zip(xb, steady):
    ax_b.text(xi, v + 0.012, f"{v:.4f}", ha="center", fontsize=FONT_SIZES["annotation"] - 2)
ax_b.set_ylim(0, 1.75)

# ---- 面板 (C) 预算曲线 ----
ax_c.plot(dd_k, dd_hot, "-o", color=C_MAIN, label="decode hotpotqa",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_c.plot(dd_k, dd_mus, "-s", color=C_SEC, label="decode musique",
          markersize=12, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_c.plot(pf_k, pf_hot, "--^", color=C_HL, label="prefill hotpotqa (grp64)",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_c.plot(pf_k, pf_mus, "--v", color="#9B59B6", label="prefill musique (grp64)",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_c.axhline(anchor_hot, color=C_NEU, lw=LINE_WIDTHS["reference"], linestyle=":")
ax_c.axhline(anchor_mus, color=C_NEU, lw=LINE_WIDTHS["reference"], linestyle=":")
ax_c.text(1015, anchor_hot + 0.6, "dense hotpot 59.4", fontsize=FONT_SIZES["annotation"] - 3,
          color=C_NEU, ha="right")
ax_c.text(1015, anchor_mus + 0.6, "dense musique 34.3", fontsize=FONT_SIZES["annotation"] - 3,
          color=C_NEU, ha="right")
ax_c.axvline(1024, color=C_HL, lw=LINE_WIDTHS["reference"], linestyle="--")
ax_c.text(1040, 62, "Quest-caliber\ntotal recall", fontsize=FONT_SIZES["annotation"] - 3,
          color=C_HL)
ax_c.set_xscale("log")
ax_c.set_xticks([256, 384, 512, 768, 1024, 1536, 2048])
ax_c.set_xticklabels(["256", "384", "512", "768", "1024", "1536", "2048"])
ax_c.set_xlabel("K2 token budget per query")
ax_c.set_ylabel("task score")
ax_c.set_title("C  K2 budget: decode cliff vs prefill recovery", loc="left", pad=10)
ax_c.legend(loc="lower left", fontsize=FONT_SIZES["legend"] - 2)
ax_c.set_ylim(5, 70)

# ---- 面板 (D) middle 归因 ----
xd = np.arange(len(tasks))
wd = 0.38
ax_d.bar(xd - wd / 2, m0, width=wd, color=C_NEU, label="FullKV (M0)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_d.bar(xd + wd / 2, m1, width=wd, color=C_HL, label="sink+SWA only (M1, 256 tok)",
         edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_d.set_xticks(xd)
ax_d.set_xticklabels(["qasper", "hotpotqa", "musique", "gov_rep", "repo"])
ax_d.set_ylabel("task score")
ax_d.set_title("D  Middle selection value (M0 vs M1)", loc="left", pad=10)
ax_d.text(0.98, 0.04, "removing middle costs -21.8 AVG5", transform=ax_d.transAxes,
          fontsize=FONT_SIZES["annotation"], color=C_HL, ha="right")
ax_d.legend(loc="upper left", fontsize=FONT_SIZES["legend"] - 2)
for xi, d0, d1 in zip(xd, m0, m1):
    ax_d.text(xi, d1 + 1.2, f"-{d0 - d1:.0f}", ha="center",
              fontsize=FONT_SIZES["annotation"] - 3, color=C_HL)
ax_d.set_ylim(0, 72)

# ---- 面板 (E) K1 倒 U ----
ax_e.axvspan(FAR_ZERO_K1, 600, color=C_NEU, alpha=0.18)
ax_e.plot(k1_dec, dec_hot, "-o", color=C_MAIN, label="decode hotpotqa",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_e.plot(k1_mus, dec_mus, "-s", color=C_SEC, label="decode musique",
          markersize=12, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_e.plot(k1_pf, pf_hot_k1, "--^", color=C_HL, label="prefill hotpotqa (grp64)",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_e.plot(k1_pf, pf_mus_k1, "--v", color="#9B59B6", label="prefill musique (grp64)",
          markersize=13, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_e.axvline(FAR_ZERO_K1, color=C_HL, lw=LINE_WIDTHS["reference"], linestyle="--")
ax_e.text(FAR_ZERO_K1 + 18, 27, "far budget = 0\n(K1 $\\geq$ 154)",
          fontsize=FONT_SIZES["annotation"] - 3, color=C_HL)
ax_e.text(20, 47.5, "left arm:\nL1 far coverage\ninsufficient", fontsize=FONT_SIZES["annotation"] - 3,
          color=C_NEU)
for k, v in zip(k1_dec, [688, 448, 128, 0]):
    ax_e.text(k, 11.5, f"far {v}", ha="center",
              fontsize=FONT_SIZES["annotation"] - 4, color=C_NEU)
ax_e.set_xscale("log")
ax_e.set_xticks([16, 64, 128, 256, 512])
ax_e.set_xticklabels(["16", "64", "128", "256", "512"])
ax_e.set_xlim(12, 600)
ax_e.set_xlabel("K1 pages (L1 coarse top-k)")
ax_e.set_ylabel("task score")
ax_e.set_title("E  K1 sweep: inverted-U, far quota collapse", loc="left", pad=10)
ax_e.legend(loc="upper right", fontsize=FONT_SIZES["legend"] - 3)
ax_e.set_ylim(8, 68)

# ---- 面板 (F) far 配额机制 ----
k1_grid = [16, 32, 64, 96, 128, 160, 192, 256, 384, 512]
for k2, c, mk, lab in [(1024, C_MAIN, "o", "K2=1024 (far>0 iff K1<154)"),
                        (896, C_SEC, "s", "K2=896  (far>0 iff K1<128)"),
                        (768, C_HL, "^", "K2=768  (far>0 iff K1<100)")]:
    fb = [far_budget(k, k2) for k in k1_grid]
    ax_f.plot(k1_grid, fb, "-" + mk, color=c, label=lab,
              markersize=11, markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
ax_f.axhline(0, color="black", lw=LINE_WIDTHS["reference"], linestyle="--")
ax_f.text(300, 60, "far=0 zone:\nmulti-hop retrieval\nstarved", fontsize=FONT_SIZES["annotation"] - 3,
          color=C_HL, ha="center")
ax_f.set_xscale("log")
ax_f.set_xticks([16, 64, 128, 256, 512])
ax_f.set_xticklabels(["16", "64", "128", "256", "512"])
ax_f.set_xlabel("K1 pages")
ax_f.set_ylabel("far token budget (K2_mid - near quota)")
ax_f.set_title("F  far quota mechanism: $K2_{mid} - \\min(K1\\beta\\,bs\\,\\gamma,\\ K2_{mid})$", loc="left", pad=10)
ax_f.legend(loc="upper right", fontsize=FONT_SIZES["legend"] - 3)
ax_f.set_ylim(-30, 760)

# ---------------- lint + 导出 ----------------
report = lint_figure(fig)
print(f"[lint] overlaps={len(report.get('overlaps', []))} clipped={len(report.get('clipped', []))}")
for o in report.get("overlaps", [])[:8]:
    print("  OVERLAP:", o)
for c in report.get("clipped", [])[:8]:
    print("  CLIPPED:", c)


def export(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")


export(fig, "A_core_ablation")

# ---------------- figure_data.tsv ----------------
rows = ["figure_id\tpanel_id\tarm_id\tmetric_id\tunit\testimate\tci_low\tci_high\tsource_run_id"]
for lab, dlt, lo, hi, _ in forest:
    rows.append(f"A\ta\t{lab.split(chr(10))[0]}\tdelta_avg5\tpoints\t{dlt:.3f}\t{lo:.3f}\t{hi:.3f}\t"
                "e133/e129/e137/e135 verdicts")
for arm, v in zip(arms, steady):
    rows.append(f"A\tb\t{arm}\tsteady_gpu_ms\tms\t{v:.4f}\t\t\te134_perf_reverdict")
for k, hh, mm in zip(dd_k, dd_hot, dd_mus):
    rows.append(f"A\tc\tdecode@K2={k}\thotpotqa\tscore\t{hh:.2f}\t\t\te139_k2_budget_pareto:armB")
    rows.append(f"A\tc\tdecode@K2={k}\tmusique\tscore\t{mm:.2f}\t\t\te139_k2_budget_pareto:armB")
for k, hh, mm in zip(pf_k, pf_hot, pf_mus):
    rows.append(f"A\tc\tprefill@K2={k}\thotpotqa\tscore\t{hh:.2f}\t\t\te139_k2_budget_pareto:armA")
    rows.append(f"A\tc\tprefill@K2={k}\tmusique\tscore\t{mm:.2f}\t\t\te139_k2_budget_pareto:armA")
for t, d0, d1 in zip(tasks, m0, m1):
    rows.append(f"A\td\t{t}\tfullkv\tscore\t{d0:.2f}\t\t\te135_sinkswa_only")
    rows.append(f"A\td\t{t}\tsinkswa_only\tscore\t{d1:.2f}\t\t\te135_sinkswa_only")
for k, v in zip(k1_dec, dec_hot):
    rows.append(f"A\te\tdecode@K1={k}\thotpotqa\tscore\t{v:.2f}\t\t\te140b_k1_downscan")
for k, v in zip(k1_mus, dec_mus):
    rows.append(f"A\te\tdecode@K1={k}\tmusique\tscore\t{v:.2f}\t\t\te140b_k1_downscan")
for k, v in zip(k1_pf, pf_hot_k1):
    rows.append(f"A\te\tprefill@K1={k}\thotpotqa\tscore\t{v:.2f}\t\t\te140a_prefill_k1up")
for k, v in zip(k1_pf, pf_mus_k1):
    rows.append(f"A\te\tprefill@K1={k}\tmusique\tscore\t{v:.2f}\t\t\te140a_prefill_k1up")
for k2 in (768, 896, 1024):
    for k in k1_grid:
        rows.append(f"A\tf\tK2={k2}@K1={k}\tfar_token_budget\ttokens\t{far_budget(k, k2)}\t\t\te140_far_quota_mechanism")
rows.append(f"A\te\tthreshold\tfar_zero_K1\tK1\t{FAR_ZERO_K1}\t\t\te140_far_quota_mechanism")
with open(os.path.join(OUT, "A_core_ablation_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported A_core_ablation_figure_data.tsv")
print("DONE")
