# -*- coding: utf-8 -*-
"""O 系列观察图（figure_id=O1/O3 + E6 gate，P0 主文动机图）—— E140v2 数据版。

设计原则：一个面板对应一个创新点的 observation（2026-10-16 用户指示迭代），
叙事链 = 分区 → far 配额 → 子空间 → 逐层 gate：

  (a) 分区创新点（ClusterKV 式 token strip，真实数据）：一条 14K 真实 query
      （musique s1 layer24 q=14212，按「far 份额最接近 128 query 中位数 0.50」
      的预登记规则选出，非 cherry-pick）的逐 64-token bin 质量条带 + 其下方的
      「单池 top-2048（按实测 bin 质量排序 = 任意打分单池的上界）」选中条带。
      sink/swa 高密度保送、near 均匀中密度、far 占 76% mid token 只有稀疏尖峰
      （最高 bin = 均值 45 倍）→ near/far 统计性质相反 → 分区异构 method。
  (b) far 配额创新点：128 条 query 的散点——即使完美的质量单池（上界），
      far 质量捕获率均值 0.69 vs near 0.93（79% 的 query 上 near>far）；
      far 弥散时（share≤0.4）捕获率仅 0.41 → far 必须有保底配额（防挤出）。
      与 E88「总质量 mono≈part」不矛盾：本面板度量的是区域公平性而非总质量，
      分区的 e2e 增益来自让渡+保底（正文口径）。
  (c) 子空间创新点（不变）：同 32/128 维预算 tail32 0.84 vs random32 0.51
      vs high32 0.24（v2 主指标 selected_mass_over_reference_topk_mass）。
  (d) D' gate 创新点：E6 逐层 far mass 热图（16 样本 × 36 层，log 色标）——
      24/36 层中位数 <0.01；逐 prompt 有 12–32 层 far 极轻 → 逐层 far gate。

数据源（全部为有效输入，不再使用 O4 v2——其部分末页缺陷见 E140_VALIDITY.md，
本图不含 O4 面板，因此无需历史诊断警示）：
  e140_o_trace_v2.json（O1/O3，采集器已修复：实际 q_norm/k_norm + 实际 RoPE）
  e6b_far_profiles.json / e6_layer_skip.json（E6 逐层 far profile，gate 标定同源）

输出 PDF+PNG+SVG + figure_data.tsv（含 (b) 全部 128 个散点、(d) 全矩阵）。
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC = os.path.join(REPO, "exp/trace/results/e140_o_trace_v2.json")
SRC6B = os.path.join(REPO, "exp/trace/results/e6b_far_profiles.json")
SRC6 = os.path.join(REPO, "exp/trace/results/e6_layer_skip.json")
for _p in (SRC, SRC6B, SRC6):
    if not os.path.exists(_p):
        raise SystemExit(f"missing input: {_p}")
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

# ================= (a) 代表 query 的 token strip =================
# 预登记选择规则：q_pos > 9000 的 query 中 far 份额最接近全体中位数者
BUDGET_TOK = 2048  # 单池预算（mid token）


def far_share(o):
    nl = o["near_len"]
    fm = sum(b["mass"] for b in o["bins"] if b["d_lo"] >= nl)
    mm = sum(b["mass"] for b in o["bins"])
    return fm / mm if mm > 0 else 0.0


shares_all = [far_share(o) for c in cells for o in c["o1"]]
share_med = float(np.median(shares_all))
best = None
for c in cells:
    for o in c["o1"]:
        if o["q_pos"] > 9000:
            sc = abs(far_share(o) - share_med)
            if best is None or sc < best[0]:
                best = (sc, c, o)
_, c_a, o_a = best
rb = o_a["region_bounds"]
bins_a = o_a["bins"]
nl_a = o_a["near_len"]
q_a = o_a["q_pos"]

# bin 的绝对位置边缘（d = q_pos − pos）仅供核对；bins 间有重叠/间隙，
# 非连续覆盖，故绘制用 bin 中心等宽近似而非 pcolormesh 精确边缘
mass_a = np.array([b["mass"] for b in bins_a])
ntok_a = np.array([b["n_tok"] for b in bins_a])
# bins 按 d_lo 降序存储 → 绝对位置升序排列
order = np.argsort([q_a - b["d_hi"] for b in bins_a])
mass_a = mass_a[order]
ntok_a = ntok_a[order]

# 单池 top-2048（按实测 bin 质量排序 = 任意打分单池的质量捕获上界）
rank_idx = np.argsort(-mass_a)
acc = 0
sel_mask = np.zeros(len(mass_a), bool)
for i in rank_idx:
    if acc >= BUDGET_TOK:
        break
    sel_mask[i] = True
    acc += ntok_a[i]
pos_c = np.array([(q_a - b["d_hi"] + q_a - b["d_lo"]) / 2 for b in bins_a])[order]
is_far = pos_c < rb["near_lo"]  # mid_lo=sink_hi，far = [sink_hi, near_lo)
fm_a = mass_a[is_far].sum()
nm_a = mass_a[~is_far].sum()
fcap_a = mass_a[is_far & sel_mask].sum() / fm_a
ncap_a = mass_a[~is_far & sel_mask].sum() / nm_a
far_tok_frac = ntok_a[is_far].sum() / ntok_a.sum()

# 跨 query 集中度：far 前 5% bin 携带的 far 质量比
conc = []
for c in cells:
    for o in c["o1"]:
        nl = o["near_len"]
        fms = sorted((b["mass"] for b in o["bins"] if b["d_lo"] >= nl),
                     reverse=True)
        if fms and sum(fms) > 0:
            k = max(1, int(0.05 * len(fms)))
            conc.append(sum(fms[:k]) / sum(fms))
conc_mean = float(np.mean(conc))

# ================= (b) 单池 far 捕获率散点（128 queries） =================
scat = []  # (task, far_share, far_capture, near_capture)
for c in cells:
    for o in c["o1"]:
        nl = o["near_len"]
        bins = o["bins"]
        fm = sum(b["mass"] for b in bins if b["d_lo"] >= nl)
        nm = sum(b["mass"] for b in bins if b["d_hi"] <= nl)
        mm = sum(b["mass"] for b in bins)
        if fm <= 0 or mm <= 0:
            continue
        ranked = sorted(bins, key=lambda b: -b["mass"])
        acc = 0
        fcap = ncap = 0.0
        for b in ranked:
            if acc >= BUDGET_TOK:
                break
            acc += b["n_tok"]
            if b["d_lo"] >= nl:
                fcap += b["mass"]
            elif b["d_hi"] <= nl:
                ncap += b["mass"]
        scat.append((c["task"], fm / mm, fcap / fm, ncap / nm if nm > 0 else np.nan))
scat = np.array([(s, f, n) for _, s, f, n in scat])
tasks = [t for t, *_ in [(c["task"],) for c in cells for o in c["o1"]
                         if sum(b["mass"] for b in o["bins"] if b["d_lo"] >= o["near_len"]) > 0
                         and sum(b["mass"] for b in o["bins"]) > 0]]
sh, fc, nc = scat[:, 0], scat[:, 1], scat[:, 2]
nc_mean = float(np.nanmean(nc))
fc_mean = float(np.mean(fc))
fc_diffuse = float(np.mean(fc[sh <= 0.4]))
fc_heavy = float(np.mean(fc[sh > 0.6]))
n_under = float(np.mean(nc > fc))

# ================= (c) O3 三臂（主指标 selected/reference topK mass） =================
arms = ["tail32", "random32", "high32"]
arm_lab = {"tail32": "tail32\n(low-freq)",
           "random32": "random32\n(unpaired)",
           "high32": "high32\n(high-freq)"}
arm_rm = {a: float(np.mean([c["o3"][a]["selected_mass_over_reference_topk_mass"]["mean"]
                            for c in cells])) for a in arms}
mets_all = ["reference_token_page_coverage", "reference_token_overlap_recall",
            "selected_mass_over_reference_topk_mass", "selected_full_attention_mass"]
arm_val = {a: [float(np.mean([c["o3"][a][mm]["mean"] for c in cells]))
               for mm in mets_all] for a in arms}

# ================= (d) E6 逐层 far mass 热图 =================
d6b = json.load(open(SRC6B))
d6 = json.load(open(SRC6))
ROW_ORDER = ["lb_gov_report_0", "lb_gov_report_1", "lb_hotpotqa_0",
             "lb_hotpotqa_1", "lb_multifieldqa_en_0", "lb_multifieldqa_en_1",
             "lb_musique_0", "lb_musique_1", "lb_narrativeqa_0",
             "lb_narrativeqa_1", "lb_passage_retrieval_en_0",
             "lb_passage_retrieval_en_1", "lb_qasper_0", "lb_qasper_1",
             "natural32k", "needle32k"]
ROW_LAB = {"lb_gov_report_0": "gov_0", "lb_gov_report_1": "gov_1",
           "lb_hotpotqa_0": "hot_0", "lb_hotpotqa_1": "hot_1",
           "lb_multifieldqa_en_0": "mfld_0", "lb_multifieldqa_en_1": "mfld_1",
           "lb_musique_0": "mus_0", "lb_musique_1": "mus_1",
           "lb_narrativeqa_0": "nar_0", "lb_narrativeqa_1": "nar_1",
           "lb_passage_retrieval_en_0": "pr_0", "lb_passage_retrieval_en_1": "pr_1",
           "lb_qasper_0": "qas_0", "lb_qasper_1": "qas_1",
           "natural32k": "natural", "needle32k": "needle"}
M = np.array([d6b[k] for k in ROW_ORDER])  # 16 x 36
lay_med = np.median(M, axis=0)
n_light = int((lay_med < 0.01).sum())
skip_rng = [v["n_skippable"] for v in d6["per_prompt"].values()]

# ================= 画布 1x4 =================
fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
    1, 4, figsize=(23, 5.6),
    gridspec_kw={"wspace": 0.44, "width_ratios": [1.55, 1.0, 0.72, 1.1]})


def note(ax, x, y, text, ha="left", va="top", color="black", size=AN):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=size, color=color,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="0.78", alpha=0.92))


# ---- (a) token strip ----
cmap_mass = mcolors.LinearSegmentedColormap.from_list("mass", ["#ffffff", C_MAIN])
norm_mass = mcolors.LogNorm(vmin=2e-4, vmax=2e-2)
X0, X1 = float(min(pos_c)), float(max(pos_c))
# 上行：逐 bin 质量（log 色标）；下行：单池 top-2048 选择（bin 等宽近似）
ax_a.imshow(mass_a[None, :], aspect="auto", cmap=cmap_mass, norm=norm_mass,
            extent=[X0, X1, 1.0, 2.0], interpolation="nearest")
cmap_sel = mcolors.ListedColormap(["#e9e9e9", C_HL])
ax_a.imshow(sel_mask.astype(float)[None, :], aspect="auto", cmap=cmap_sel,
            vmin=0, vmax=1, extent=[X0, X1, 0.0, 1.0],
            interpolation="nearest")
# sink / swa 保护区（不在 mid bins 内；区域太窄无法内嵌文字，见说明框）
ax_a.axvspan(0, rb["sink_hi"], color=C_NEU, alpha=0.35, lw=0)
ax_a.axvspan(rb["swa_lo"], q_a + 1, color=C_NEU, alpha=0.35, lw=0)
# near/far 区域标注
ax_a.text((rb["sink_hi"] + rb["near_lo"]) / 2, 2.08,
          f"far ({far_tok_frac:.0%} of mid tokens)", ha="center",
          fontsize=AN, color=C_HL, fontweight="bold")
ax_a.text((rb["near_lo"] + rb["swa_lo"]) / 2, 2.08, "near", ha="center",
          fontsize=AN, color=C_MAIN, fontweight="bold")
ax_a.axvline(rb["near_lo"], color=C_HL, lw=1.6, linestyle="--")
ax_a.set_xlim(0, q_a + 1)
ax_a.set_ylim(0, 3.4)
ax_a.set_yticks([0.5, 1.5])
ax_a.set_yticklabels(["single-pool\ntop-2048", "attn mass"], fontsize=TK - 1.5)
ax_a.set_xlabel("key position (tokens); query at right end", fontsize=L)
ax_a.set_title("(a) Mass is position-segmented", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_a.tick_params(labelsize=TK)
note(ax_a, 0.015, 0.98,
     f"gray: sink ({o_a['mass_sink']:.2f}) / swa "
     f"({o_a['mass_swa_excluding_sink']:.2f}) mass auto-included; "
     f"far ≈ 50% of mid mass\nover {far_tok_frac:.0%} of mid tokens, as sparse spikes "
     f"(top-5% bins ≈ {conc_mean:.0%} of far mass);\nsingle pool captures near "
     f"{ncap_a:.2f} / far {fcap_a:.2f} of mass — smaller spikes missed\n"
     "→ near: block-avg · far: no-miss bound + fine select",
     color="black", size=AN - 1.5)
sm = plt.cm.ScalarMappable(cmap=cmap_mass, norm=norm_mass)
cb = fig.colorbar(sm, ax=ax_a, fraction=0.025, pad=0.012)
cb.set_label("bin mass (log)", fontsize=AN - 1.5)
cb.ax.tick_params(labelsize=TK - 3)

# ---- (b) 单池 far 捕获率散点 ----
for task, col, mk in (("hotpotqa", C_MAIN, "o"), ("musique", C_HL, "^")):
    idx = np.array([t == task for t in tasks])
    ax_b.scatter(sh[idx], fc[idx], s=26, color=col, marker=mk, alpha=0.75,
                 edgecolor="white", linewidth=0.5, label=f"{task} ({idx.sum()} queries)")
ax_b.axhline(nc_mean, color=C_NEU, lw=2.0, linestyle="--")
ax_b.text(0.30, nc_mean - 0.03, f"near capture, same pool: {nc_mean:.2f}",
          ha="center", va="top", fontsize=AN - 1.5, color=C_NEU)
ax_b.axvspan(-0.03, 0.4, color=C_HL, alpha=0.08, lw=0)
ax_b.text(0.19, 0.72, f"diffuse far:\ncapture {fc_diffuse:.2f}", ha="center",
          va="center", fontsize=AN - 1.5, color=C_HL)
ax_b.set_xlabel("far share of mid mass", fontsize=L)
ax_b.set_ylabel("far mass capture,\noracle single-pool top-2048", fontsize=L - 1)
ax_b.set_xlim(-0.03, 1.0)
ax_b.set_ylim(0, 1.05)
ax_b.set_title("(b) Single pool under-serves far", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_b.tick_params(labelsize=TK)
ax_b.legend(loc="lower right", fontsize=LG - 2, framealpha=0.92)
note(ax_b, 0.97, 0.97,
     f"far capture {fc_mean:.2f} ≪ near {nc_mean:.2f}\n"
     f"({n_under:.0%} of queries) → guaranteed\nfar quota (anti-eviction)",
     ha="right", va="top", color=C_HL, size=AN - 1)

# ---- (c) O3 三臂 ----
colors3 = [C_MAIN, C_NEU, C_SEC]
for i, (a, col) in enumerate(zip(arms, colors3)):
    ax_c.bar([i], [arm_rm[a]], width=0.58, color=col, edgecolor="black",
             linewidth=1.2)
    ax_c.text(i, arm_rm[a] + 0.025, f"{arm_rm[a]:.2f}", ha="center",
              fontsize=AN, fontweight="bold")
ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels([arm_lab[a] for a in arms], fontsize=TK - 1)
ax_c.set_ylabel("selected mass /\nreference topK mass", fontsize=L)
ax_c.set_ylim(0, 1.0)
ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_c.set_title("(c) Which 32 of 128 dims", loc="left", pad=10, fontsize=T,
               fontweight="bold")
ax_c.tick_params(labelsize=TK)
note(ax_c, 0.96, 0.96, "position priors (paired\nlow-freq RoPE dims),\n"
     "not “any 32 dims”", ha="right", va="top", color=C_HL, size=AN - 1)

# ---- (d) E6 逐层 far mass 热图 ----
norm6 = mcolors.LogNorm(vmin=1e-7, vmax=1.0)
im = ax_d.imshow(M, aspect="auto", cmap=cmap_mass, norm=norm6,
                 interpolation="nearest")
ax_d.set_xticks([0, 8, 16, 24, 32])
ax_d.set_yticks(range(len(ROW_ORDER)))
ax_d.set_yticklabels([ROW_LAB[k] for k in ROW_ORDER], fontsize=TK - 3.5)
ax_d.set_xlabel("layer", fontsize=L)
ax_d.set_title("(d) Far mass by layer (16 seqs)", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_d.tick_params(labelsize=TK)
note(ax_d, 0.97, 0.03,
     f"{n_light}/36 layers: median far mass <0.01;\n"
     f"per seq {min(skip_rng)}–{max(skip_rng)} far-light layers\n"
     "→ per-layer far gate (D′)",
     ha="right", va="bottom", color=C_HL, size=AN - 1.5)
cb6 = fig.colorbar(im, ax=ax_d, fraction=0.025, pad=0.012)
cb6.set_label("far mass (log)", fontsize=AN - 1.5)
cb6.ax.tick_params(labelsize=TK - 3)

fig.subplots_adjust(left=0.05, right=0.965, top=0.87, bottom=0.16)

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
rows.append(f"O\ta\trepr_query\tidentity\t-\t{c_a['task']}:s{c_a['sample']}:L{c_a['layer']}:q{q_a}\t{src}\tfar_share closest to median among q_pos>9000")
rows.append(f"O\ta\trepr_query\tfar_tok_frac\tfrac\t{far_tok_frac:.4f}\t{src}\tbins")
rows.append(f"O\ta\trepr_query\tsingle_pool_near_capture\tfrac\t{ncap_a:.4f}\t{src}\tbin mass-rank top-{BUDGET_TOK}")
rows.append(f"O\ta\trepr_query\tsingle_pool_far_capture\tfrac\t{fcap_a:.4f}\t{src}\tbin mass-rank top-{BUDGET_TOK}")
rows.append(f"O\ta\tall_queries\tfar_top5pct_bin_concentration\tfrac\t{conc_mean:.4f}\t{src}\tbins")
for x, msv, s in zip(pos_c, mass_a, sel_mask):
    rows.append(f"O\ta\trepr_strip\tbin_mass\tfrac\t{msv:.4e}\t{src}\tpos={x:.0f},sel={int(s)}")
for (tk, s, f, n) in [(t, a, b, cc) for (t, a, b, cc) in
                      zip(tasks, sh, fc, nc)]:
    rows.append(f"O\tb\t{tk}\tshare_far_capture\tfrac\t{s:.4f},{f:.4f},{n:.4f}\t{src}\tbin mass-rank top-{BUDGET_TOK}")
rows.append(f"O\tb\tall\tfar_capture_mean\tfrac\t{fc_mean:.4f}\t{src}\t-")
rows.append(f"O\tb\tall\tnear_capture_mean\tfrac\t{nc_mean:.4f}\t{src}\t-")
rows.append(f"O\tb\tdiffuse(share<=0.4)\tfar_capture\tfrac\t{fc_diffuse:.4f}\t{src}\t-")
for a in arms:
    for mm, v in zip(mets_all, arm_val[a]):
        rows.append(f"O\tc\t{a}\t{mm}\tfrac\t{v:.4f}\t{src}\to3.{a}.{mm}.mean")
for k, row in zip(ROW_ORDER, M):
    for li, v in enumerate(row):
        rows.append(f"O\td\t{k}\tfar_mass\tfrac\t{v:.4e}\te6b_far_profiles.json\tlayer={li}")
rows.append(f"O\td\tall\tlayers_median_lt_0.01\tcount\t{n_light}\te6b_far_profiles.json\t-")
rows.append(f"O\td\tall\tskippable_layers_per_prompt\trange\t{min(skip_rng)}-{max(skip_rng)}\te6_layer_skip.json\t-")
with open(os.path.join(OUT, "O_observation_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported O_observation_figure_data.tsv")
print("DONE")
