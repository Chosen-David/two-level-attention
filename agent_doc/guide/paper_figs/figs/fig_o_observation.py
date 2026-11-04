# -*- coding: utf-8 -*-
"""O 系列观察图（figure_id=O）—— NFA 本地几何口径版。

设计原则：一个面板对应一个创新点的 observation，叙事链 = 分区+异构打分 →
共享子空间 → 逐层 far gate。

本地生产几何（sparse_attn/indexer/nfa_layer_batch_selector.py，
sparse_attn/arguments.py 默认参数）：
  sink  = nfa_sink_tokens = 128
  swa   = nfa_swa_size    = 128
  near  = nfa_near_len    = 4096（页对齐 64）
  page  = tia_block_size  = 64
  far   = [sink_hi, far_hi)，far_hi = floor((t-128-4096)/64)*64
  near  = [far_hi, t-128)
  far 池 L1 粗筛 = minmax（块上界）；near 池 L1 粗筛 = avg（块均值）

面板：
  (a) 分区+异构打分：代表 query（lb_hotpotqa_0:layer03，长序列 t=16956）的
      far/near 区域选中页数对比——oracle（真实质量）在 far 选 23 页 / near 7 页，
      avg 全选 near（far 0 / near 30，far 稀疏尖峰被均值稀释漏选），
      minmax 在 far 救回 21 页（far 21 / near 10）。全 576 样本汇总：
      minmax 在 far 选更多页的样本占 86.1%，avg 在 near 选更多页的样本占 83.2%。
  (b) 共享子空间：tail32 0.84 vs random32 0.51 vs high32 0.24
      （selected_mass_over_reference_topk_mass）。
  (c) far gate：near/far 注意力质量中位柱（16 样本 × 36 层，本地口径）——
      far 中位 mass 低（多数层 <0.01 的样本占比高）→ 逐层 far gate 动机。

数据源：
  e140b_nfa_region_anatomy.json（本地口径页级 avg/minmax 选择解剖，图 a）
  e140_o_trace_v2.json（共享子空间三臂，图 b）
  e6b_near_far_profiles.json（本地口径 near/far mass，图 c）

输出 PDF+PNG+SVG + figure_data.tsv。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC_O = os.path.join(REPO, "exp/trace/results/e140_o_trace_v2.json")
SRC_ANAT = os.path.join(REPO, "exp/trace/results/e140b_nfa_region_anatomy.json")
SRC_MASS = os.path.join(REPO, "exp/trace/results/e6b_near_far_profiles.json")
for _p in (SRC_O, SRC_ANAT, SRC_MASS):
    if not os.path.exists(_p):
        raise SystemExit(f"missing input: {_p}")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
ZH = os.environ.get("O_LANG") == "zh"
if ZH:
    from matplotlib import font_manager as _fm
    for _fp in ("/home/wangyuanshuo02/.local/share/fonts/cjk/NotoSansCJKsc-Regular.otf",
                "/home/wangyuanshuo02/.local/share/fonts/cjk/NotoSansCJKsc-Bold.otf"):
        if os.path.exists(_fp):
            _fm.fontManager.addfont(_fp)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Noto Sans CJK SC" if ZH else "DejaVu Sans"
C_MAIN = pal["primary"]
C_SEC = pal["secondary"]
C_TER = pal["tertiary"]
C_HL = pal["highlight"]
C_NEU = pal["neutral"]
C_MISS = "#d62728"
C_EXTRA = "#9467bd"

T = 13
L = 13
TK = 11.5
AN = 10.5
LG = 10.5

# ================= 数据加载 =================
do = json.load(open(SRC_O))
cells = do["cells"]

danat = json.load(open(SRC_ANAT))
anat_meta = danat["meta"]
REP_KEY = ("lb_hotpotqa_0", 3)  # 代表 query：lb_hotpotqa_0:layer03
rep = danat["profiles"][REP_KEY[0]][REP_KEY[1]]
all_rows = [r for rows in danat["profiles"].values() for r in rows]
n_cells = len(all_rows)
frac_mm_more_far = float(np.mean([r["mm_far"] > r["avg_far"] for r in all_rows]))
frac_avg_more_near = float(np.mean([r["avg_near"] > r["mm_near"] for r in all_rows]))
# 全样本 far/near 质量捕获（组内区域质量归一）
avg_far_cap_mean = float(np.mean([r["avg_far_cap"] for r in all_rows]))
mm_far_cap_mean = float(np.mean([r["mm_far_cap"] for r in all_rows]))
avg_near_cap_mean = float(np.mean([r["avg_near_cap"] for r in all_rows]))
mm_near_cap_mean = float(np.mean([r["mm_near_cap"] for r in all_rows]))

# ---- 图 (b) 共享子空间三臂 ----
arms = ["tail32", "random32", "high32"]
arm_lab = {"tail32": "tail", "random32": "rand", "high32": "high"}
arm_rm = {a: float(np.mean([c["o3"][a]["selected_mass_over_reference_topk_mass"]["mean"]
                            for c in cells])) for a in arms}
mets_all = ["reference_token_page_coverage", "reference_token_overlap_recall",
            "selected_mass_over_reference_topk_mass", "selected_full_attention_mass"]
arm_val = {a: [float(np.mean([c["o3"][a][mm]["mean"] for c in cells]))
               for mm in mets_all] for a in arms}

# ---- 图 (c) near/far mass ----
dmass = json.load(open(SRC_MASS))
ROW_ORDER = ["lb_gov_report_0", "lb_gov_report_1", "lb_hotpotqa_0",
             "lb_hotpotqa_1", "lb_multifieldqa_en_0", "lb_multifieldqa_en_1",
             "lb_musique_0", "lb_musique_1", "lb_narrativeqa_0",
             "lb_narrativeqa_1", "lb_passage_retrieval_en_0",
             "lb_passage_retrieval_en_1", "lb_qasper_0", "lb_qasper_1",
             "natural32k", "needle32k"]
M = np.array([[dmass[k][li]["far"] for li in range(36)] for k in ROW_ORDER])
N = np.array([[dmass[k][li]["near"] for li in range(36)] for k in ROW_ORDER])
far_flat = M.ravel()
near_flat = N.ravel()
far_med = float(np.median(far_flat))
near_med = float(np.median(near_flat))
far_mean = float(far_flat.mean())
near_mean = float(near_flat.mean())
lay_med = np.median(M, axis=0)
n_light = int((lay_med < 0.01).sum())
skip_rng = (M < 0.01).sum(axis=1)

# ================= 画布 1x3 网格 =================
fig = plt.figure(figsize=(10.2, 3.9))
gs = fig.add_gridspec(
    1, 3, width_ratios=[1.30, 0.78, 0.76],
    wspace=0.50,
    left=0.095, right=0.90, top=0.85, bottom=0.20)
ax_a = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])
ax_d = fig.add_subplot(gs[0, 2])

# ---- (a) 区域选择对比：far/near x oracle/avg/minmax 选中页数 ----
regions = ["far", "near"]
region_lab = {"far": "远区" if ZH else "far", "near": "近区" if ZH else "near"}
arms3 = ["oracle", "avg", "minmax"]
arm3_lab = {"oracle": "质量参考" if ZH else "mass ref",
            "avg": "均值" if ZH else "avg",
            "minmax": "极值" if ZH else "minmax"}
arm3_col = {"oracle": C_NEU, "avg": C_MAIN, "minmax": C_HL}
arm3_val = {
    "oracle": {"far": rep["o_far"], "near": rep["o_near"]},
    "avg": {"far": rep["avg_far"], "near": rep["avg_near"]},
    "minmax": {"far": rep["mm_far"], "near": rep["mm_near"]},
}
xpos = {"far": 0, "near": 1}
width = 0.26
for ri, reg in enumerate(regions):
    for ai, arm in enumerate(arms3):
        v = arm3_val[arm][reg]
        x = xpos[reg] + (ai - 1) * width
        ax_a.bar([x], [v], width=width * 0.92, color=arm3_col[arm],
                 edgecolor="black", linewidth=1.0)
        ax_a.text(x, v + 0.6, f"{v}", ha="center", va="bottom",
                  fontsize=AN, fontweight="bold")
ax_a.set_xticks([xpos[r] for r in regions])
ax_a.set_xticklabels([region_lab[r] for r in regions], fontsize=TK)
ax_a.set_ylabel("选中页数" if ZH else "selected pages", fontsize=L)
ax_a.set_ylim(0, 36)
ax_a.set_yticks([0, 10, 20, 30])
for _sp in ("top", "right"):
    ax_a.spines[_sp].set_visible(False)
ax_a.grid(axis="y", color="#d5d5d5", linewidth=0.5)
ax_a.set_axisbelow(True)
leg_a = [Patch(fc=arm3_col[a], edgecolor="black", lw=0.5,
               label=arm3_lab[a]) for a in arms3]
ax_a.legend(handles=leg_a, loc="upper right", bbox_to_anchor=(0.995, 1.01),
            fontsize=LG - 4, framealpha=0.92, handlelength=1.0,
            borderpad=0.30, labelspacing=0.9, ncol=3, columnspacing=0.55)
ax_a.set_title("(a) 极值偏远端；均值偏近端" if ZH
               else "(a) Minmax favors far; avg near",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_a.tick_params(labelsize=TK)

# ---- (b) 共享子空间三臂 ----
colors3 = [C_MAIN, C_NEU, C_SEC]
for i, (a, col) in enumerate(zip(arms, colors3)):
    ax_c.bar([i], [arm_rm[a]], width=0.58, color=col, edgecolor="black",
             linewidth=1.2)
    ax_c.text(i, arm_rm[a] + 0.025, f"{arm_rm[a]:.2f}", ha="center",
              fontsize=AN, fontweight="bold")
ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels([arm_lab[a] for a in arms], fontsize=TK - 4.5)
ax_c.set_ylabel("选中 / 参考质量" if ZH else "selected / reference mass",
                fontsize=L)
ax_c.set_ylim(0, 1.0)
ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_c.set_title("(b) 选哪 32 维" if ZH else "(b) Which 32 dims",
               loc="left", pad=10, fontsize=T, fontweight="bold")
for _sp in ("top", "right"):
    ax_c.spines[_sp].set_visible(False)
ax_c.grid(axis="y", color="#d5d5d5", linewidth=0.5)
ax_c.set_axisbelow(True)
ax_c.tick_params(labelsize=TK)

# ---- (c) near/far 注意力质量中位柱 ----
vals = [far_med, near_med]
cols = [C_HL, C_MAIN]
labels = ["远区" if ZH else "far", "近区" if ZH else "near"]
for i, (v, col) in enumerate(zip(vals, cols)):
    ax_d.bar([i], [v], width=0.52, color=col, edgecolor="black", linewidth=1.0)
    ax_d.text(i, v * 1.18, f"{v:.2g}", ha="center", va="bottom",
              fontsize=AN, fontweight="bold")
ax_d.set_yscale("log")
ax_d.set_ylim(1e-4, 1.0)
ax_d.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
ax_d.set_xticks([0, 1])
ax_d.set_xticklabels(labels, fontsize=TK)
ax_d.set_ylabel("中位注意力质量" if ZH else "median attn mass", fontsize=L)
for _sp in ("top", "right"):
    ax_d.spines[_sp].set_visible(False)
ax_d.grid(axis="y", color="#d5d5d5", linewidth=0.5)
ax_d.set_axisbelow(True)
ax_d.set_title("(c) 近远区质量对比" if ZH else "(c) Near vs far mass",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_d.tick_params(labelsize=TK)

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


export(fig, "O_observation_zh" if ZH else "O_observation")

# ---------------- figure_data.tsv ----------------
rows = ["figure_id\tpanel_id\tarm_id\tmetric_id\tunit\testimate\tsource_run_id\tsource_field"]
anat_name = os.path.basename(SRC_ANAT)
rows.append(f"O\ta\trepr_query\tidentity\t-\t{REP_KEY[0]}:layer03:t{rep['t']}\t{anat_name}\tfixed illustrative example (local NFA geometry sink=128 near=4096 swa=128 page=64)")
rows.append(f"O\ta\trepr_query\tsel_pages_far\tcount\to={rep['o_far']},avg={rep['avg_far']},mm={rep['mm_far']}\t{anat_name}\toracle/avg/minmax selected far blocks")
rows.append(f"O\ta\trepr_query\tsel_pages_near\tcount\to={rep['o_near']},avg={rep['avg_near']},mm={rep['mm_near']}\t{anat_name}\toracle/avg/minmax selected near blocks")
rows.append(f"O\ta\trepr_query\tmass_capture_far\tfrac\tavg={rep['avg_far_cap']:.3f},mm={rep['mm_far_cap']:.3f}\t{anat_name}\tavg/minmax captured far-region mass fraction")
rows.append(f"O\ta\trepr_query\tmass_capture_near\tfrac\tavg={rep['avg_near_cap']:.3f},mm={rep['mm_near_cap']:.3f}\t{anat_name}\tavg/minmax captured near-region mass fraction")
rows.append(f"O\ta\tall_queries\tfrac_minmax_more_far\tfrac\t{frac_mm_more_far:.4f}\t{anat_name}\tfraction of {n_cells} cells where minmax selects more far blocks than avg")
rows.append(f"O\ta\tall_queries\tfrac_avg_more_near\tfrac\t{frac_avg_more_near:.4f}\t{anat_name}\tfraction of {n_cells} cells where avg selects more near blocks than minmax")
rows.append(f"O\ta\tall_queries\tmass_capture_far_mean\tfrac\tavg={avg_far_cap_mean:.3f},mm={mm_far_cap_mean:.3f}\t{anat_name}\tmean over {n_cells} cells")
rows.append(f"O\ta\tall_queries\tmass_capture_near_mean\tfrac\tavg={avg_near_cap_mean:.3f},mm={mm_near_cap_mean:.3f}\t{anat_name}\tmean over {n_cells} cells")
rows.append(f"O\ta\tgeometry\tnear_far_bounds\tcount\tsink={rep['sink_hi']},far_hi={rep['far_hi']},swa_hi={rep['swa_hi']}\t{anat_name}\tlocal NFA region bounds for representative query")
srcb = os.path.basename(SRC_O)
for a in arms:
    for mm, v in zip(mets_all, arm_val[a]):
        rows.append(f"O\tb\t{a}\t{mm}\tfrac\t{v:.4f}\t{srcb}\to3.{a}.{mm}.mean")
srcm = os.path.basename(SRC_MASS)
for k in ROW_ORDER:
    for li in range(36):
        rows.append(f"O\tc\t{k}\tfar_mass\tfrac\t{dmass[k][li]['far']:.4e}\t{srcm}\tlayer={li}")
for k in ROW_ORDER:
    for li in range(36):
        rows.append(f"O\tc\t{k}\tnear_mass\tfrac\t{dmass[k][li]['near']:.4e}\t{srcm}\tlayer={li}")
rows.append(f"O\tc\tall\tfar_median_mass\tfrac\t{far_med:.4f}\t{srcm}\tmedian over 16 prompts x 36 layers (local NFA geometry)")
rows.append(f"O\tc\tall\tnear_median_mass\tfrac\t{near_med:.4f}\t{srcm}\tmedian over 16 prompts x 36 layers (local NFA geometry)")
rows.append(f"O\tc\tall\tfar_mean_mass\tfrac\t{far_mean:.4f}\t{srcm}\tmean over 16 prompts x 36 layers")
rows.append(f"O\tc\tall\tnear_mean_mass\tfrac\t{near_mean:.4f}\t{srcm}\tmean over 16 prompts x 36 layers")
rows.append(f"O\tc\tall\tlayers_median_lt_0.01\tcount\t{n_light}\t{srcm}\t-")
rows.append(f"O\tc\tall\tlayers_lt_0.01_per_prompt\trange\t{int(skip_rng.min())}-{int(skip_rng.max())}\t{srcm}\t-")
if not ZH:
    with open(os.path.join(OUT, "O_observation_figure_data.tsv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("exported O_observation_figure_data.tsv")
print("DONE")
