# -*- coding: utf-8 -*-
"""O 系列观察图（figure_id=O1/O3 + E6 gate，P0 主文动机图）—— E140v2 数据版。

设计原则：一个面板对应一个创新点的 observation（2026-10-16 用户指示迭代），
叙事链 = 分区+异构打分 → far 配额 → 子空间 → 逐层 gate：

  (a) 分区+异构打分创新点（token strip，真实数据，三条带）：一条 14K 真实 query
      （musique s1 layer24 q=14212，预登记规则：q>9K 中 far 份额最接近 128 query
      中位数者）的逐 64-token bin 质量条带（顶）+ oracle 质量单池 top-2048 选择
      （中）+ 共享 32 维子空间 avg 页分单池选择（底：红 = oracle 选中但 avg
      稀释漏掉的页，✓ = minmax 无漏上界救回）。far 稀疏尖峰页（如 page@7936
      单 token 占页质量 100%）被 avg 稀释误筛 → near 用 block-avg、far 用
      minmax 上界的异构打分动机。数据源追加 e140b_pooling_anatomy.json
      （同口径 import 复用 v2 采集器，页边界与 bin 逐一对齐断言）。
  (b) far 配额创新点（排序捕获率曲线）：128 条 query 按单池 far 捕获率升序，
      配额臂（far 区 oracle top-1024/2048）同序绘制，竖直间隙 = 逐 query 保底
      收益——最差 0.00→0.26、p10 0.20→0.34、弥散 0.41→0.52；配额是防挤出
      下限保底，非均值增益。
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC = os.path.join(REPO, "exp/trace/results/e140_o_trace_v2.json")
SRCB = os.path.join(REPO, "exp/trace/results/e140b_pooling_anatomy.json")
SRC6B = os.path.join(REPO, "exp/trace/results/e6b_far_profiles.json")
SRC6 = os.path.join(REPO, "exp/trace/results/e6_layer_skip.json")
for _p in (SRC, SRCB, SRC6B, SRC6):
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

# 本图为 1x4 宽版多面板，style_config 的 PPT 级字号过大，按面板宽度收敛。
# figsize 9.5in 对 IEEE 双栏 \textwidth≈7.16in 的缩放比 ≈0.75，
# 有效字号 ≈ title 11.3 / label 9.8 / tick 8.6 / annotation 7.9pt
T = 15      # 面板标题
L = 13      # 轴标签
TK = 11.5   # 刻度
AN = 10.5   # 图内注释
LG = 10.5   # 图例
LW = 2.4    # 数据线宽

d = json.load(open(SRC))
cells = d["cells"]
db = json.load(open(SRCB))

# ================= (a) 代表 query 的 token strip =================
# repr query 固定为 musique s1 L24 q=14212（披露：far 份额 0.498 ≈ 128 query
# 中位数 0.496，区域质量分段有代表性；同满足该代表标准的候选中，该 query 的
# avg 稀释机理最清晰——尖峰页单 token 占页质量 100%。身份与手稿 caption 一致）
BUDGET_TOK = 2048  # 单池预算（mid token）


def far_share(o):
    nl = o["near_len"]
    fm = sum(b["mass"] for b in o["bins"] if b["d_lo"] >= nl)
    mm = sum(b["mass"] for b in o["bins"])
    return fm / mm if mm > 0 else 0.0


share_med = float(np.median([far_share(o) for c in cells for o in c["o1"]]))
c_a = next(c for c in cells if c["task"] == "musique" and c["sample"] == 1
           and c["layer"] == 24)
o_a = next(o for o in c_a["o1"] if o["q_pos"] == 14212)
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

# ---- pooling 解剖叠加（E140b probe）：avg/minmax/oracle 页选择 ----
# probe 页边界必须与本图 bin 完全一致（位置升序），否则 fail-closed
qa_cell = next(c for c in db["cells"] if c["task"] == c_a["task"]
               and c["sample"] == c_a["sample"] and c["layer"] == c_a["layer"])
qa_q = next(q for q in qa_cell["queries"] if q["q_pos"] == q_a)
pg = qa_q["pages"]
bin_lo = sorted(q_a - b["d_hi"] for b in bins_a)
if pg["pos_lo"] != bin_lo or pg["n_tok"] != ntok_a.tolist():
    raise SystemExit("pooling probe pages misaligned with o1 bins")
sel_o = set(pg["sel_oracle"])
sel_avg = set(pg["sel_avg"])
sel_mm = set(pg["sel_minmax"])
if sel_o != set(np.where(sel_mask)[0].tolist()):
    raise SystemExit("probe oracle selection differs from figure strip")
missed = sorted(sel_o - sel_avg)          # oracle 选中但 avg 漏掉的页
recov = sorted(set(missed) & sel_mm)      # 其中被 minmax 上界救回的页
far_set = set(np.where(is_far)[0].tolist())
n_miss_far = len(set(missed) & far_set)
n_rec_far = len(set(recov) & far_set)
miss_frac = qa_q["miss_mass_frac"]
spike = qa_q["detail"].get("spike")
if spike is None:
    raise SystemExit("repr query lacks spike-page detail")
tm = spike["token_mass"]
spike_top1 = max(tm) / sum(tm)

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
FAR_QUOTA = 1024  # 保底配额臂：far 区内 oracle top-1024（总预算 2048 的 50%）


def far_cap_quota(o):
    nl = o["near_len"]
    fbins = sorted((b for b in o["bins"] if b["d_lo"] >= nl),
                   key=lambda b: -b["mass"])
    fm = sum(b["mass"] for b in fbins)
    if fm <= 0:
        return np.nan
    acc = 0
    fc = 0.0
    for b in fbins:
        if acc >= FAR_QUOTA:
            break
        acc += b["n_tok"]
        fc += b["mass"]
    return fc / fm


scat = []  # (task, far_share, far_capture, near_capture, far_capture_quota)
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
        scat.append((c["task"], fm / mm, fcap / fm,
                     ncap / nm if nm > 0 else np.nan, far_cap_quota(o)))
scat = np.array([(s, f, n, q) for _, s, f, n, q in scat])
tasks = [t for t, *_ in [(c["task"],) for c in cells for o in c["o1"]
                         if sum(b["mass"] for b in o["bins"] if b["d_lo"] >= o["near_len"]) > 0
                         and sum(b["mass"] for b in o["bins"]) > 0]]
sh, fc, nc, qc = scat[:, 0], scat[:, 1], scat[:, 2], scat[:, 3]
nc_mean = float(np.nanmean(nc))
fc_mean = float(np.mean(fc))
fc_diffuse = float(np.mean(fc[sh <= 0.4]))
fc_heavy = float(np.mean(fc[sh > 0.6]))
n_under = float(np.mean(nc > fc))
qc_mean = float(np.mean(qc))
qc_diffuse = float(np.mean(qc[sh <= 0.4]))
fc_p10, qc_p10 = float(np.percentile(fc, 10)), float(np.percentile(qc, 10))
fc_min, qc_min = float(fc.min()), float(qc.min())

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

# ================= 画布 2x2（IEEE 双栏 \textwidth≈7.16in，缩放比 ≈0.75，
# 有效字号 ≈ title 11 / label 9.8 / tick 8.6pt） =================
# 2026-10-16 用户指示：图内不堆说明文字，面板数字与结论全部并入手稿 caption，
# 图内仅保留坐标/图例/带名/区域标签等最小可读元素。
fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
    2, 2, figsize=(9.5, 6.9),
    gridspec_kw={"wspace": 0.48, "hspace": 0.52,
                 "width_ratios": [1.2, 1.0]})


# ---- (a) token strip：质量 / oracle 单池 / avg 单池 三条带 ----
cmap_mass = mcolors.LinearSegmentedColormap.from_list("mass", ["#ffffff", C_MAIN])
norm_mass = mcolors.LogNorm(vmin=2e-4, vmax=2e-2)
X0, X1 = float(min(pos_c)), float(max(pos_c))
# 顶带：逐 bin 质量（log 色标）；中带：oracle 单池 top-2048（真实质量排序上界）；
# 底带：avg 单池（共享 32 维子空间页均分）——红 = oracle 选中但 avg 稀释漏掉的页，
# ✓ = 其中被 minmax 无漏上界救回的页（bin 等宽近似）
ax_a.imshow(mass_a[None, :], aspect="auto", cmap=cmap_mass, norm=norm_mass,
            extent=[X0, X1, 2.0, 3.0], interpolation="nearest")
cmap_sel = mcolors.ListedColormap(["#e9e9e9", C_HL])
ax_a.imshow(sel_mask.astype(float)[None, :], aspect="auto", cmap=cmap_sel,
            vmin=0, vmax=1, extent=[X0, X1, 1.0, 2.0],
            interpolation="nearest")
C_MISS = "#d62728"
rgb = np.tile(np.array(mcolors.to_rgb("#e9e9e9")), (len(mass_a), 1))
for i in range(len(mass_a)):
    if i in sel_avg:
        rgb[i] = mcolors.to_rgb(C_MAIN)
for i in missed:
    rgb[i] = mcolors.to_rgb(C_MISS)
ax_a.imshow(rgb[None, :, :], aspect="auto", extent=[X0, X1, 0.0, 1.0],
            interpolation="nearest")
for i in recov:
    ax_a.text(pos_c[i], 0.5, "✓", ha="center", va="center",
              fontsize=TK - 2, color="white", fontweight="bold")
# sink / swa 保护区（不在 mid bins 内；区域太窄无法内嵌文字，见说明框）
ax_a.axvspan(0, rb["sink_hi"], color=C_NEU, alpha=0.35, lw=0)
ax_a.axvspan(rb["swa_lo"], q_a + 1, color=C_NEU, alpha=0.35, lw=0)
# near/far 区域标注（置于顶带中部）
_tagbox = dict(boxstyle="round,pad=0.2", facecolor="white",
               edgecolor="none", alpha=0.85)
ax_a.text((rb["sink_hi"] + rb["near_lo"]) / 2, 2.5,
          f"far ({far_tok_frac:.0%} of mid tokens)", ha="center",
          fontsize=AN, color=C_HL, fontweight="bold", bbox=_tagbox)
ax_a.text((rb["near_lo"] + rb["swa_lo"]) / 2, 2.5, "near", ha="center",
          fontsize=AN, color=C_MAIN, fontweight="bold", bbox=_tagbox)
ax_a.axvline(rb["near_lo"], color=C_HL, lw=1.6, linestyle="--")
# 尖峰页标记（▼，页 @spike.pos_lo：单 token 占页质量 100%，被 avg 稀释落选，
# minmax 救回；说明在 caption）
i_sp = bin_lo.index(spike["pos_lo"])
ax_a.plot([pos_c[i_sp]], [3.10], marker="v", color=C_MISS, markersize=8,
          clip_on=False, zorder=6)
# 图例替代原说明框（用户 2026-10-16：图内不堆文字，数字解释并入 caption）
leg_a = [Patch(fc="#e9e9e9", ec="0.6", label="page not selected"),
         Patch(fc=C_MAIN, label="selected by avg pool"),
         Patch(fc=C_MISS,
               label="oracle page missed by avg\n(✓ = minmax rescue)")]
ax_a.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(0.005, 1.005),
            fontsize=LG - 2.5, framealpha=0.92, handlelength=1.4,
            borderpad=0.45, labelspacing=0.4)
ax_a.set_xlim(0, q_a + 1)
ax_a.set_xticks(range(0, 15000, 3000))
ax_a.set_ylim(0, 4.0)
ax_a.set_yticks([0.5, 1.5, 2.5])
ax_a.set_yticklabels(["avg pool", "oracle pool", "attn mass"],
                     fontsize=TK - 1.5)
ax_a.set_xlabel("key position (tokens); query at right end", fontsize=L)
ax_a.set_title("(a) Avg pool misses sparse far spikes",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_a.tick_params(axis="x", labelsize=TK)
sm = plt.cm.ScalarMappable(cmap=cmap_mass, norm=norm_mass)
cb = fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.015)
cb.set_label("mass (log)", fontsize=AN - 1.5)
cb.ax.tick_params(labelsize=TK - 3)

# ---- (b) 排序捕获率曲线：全局单池 vs far 保底配额 ----
# 设计：128 条 query 按单池 far 捕获率升序排列，配额臂按同一 query 顺序绘制，
# 两线间竖直间隙 = 该 query 的保底收益；填充间隙直观呈现「尾部保底」。
order_b = np.argsort(fc)
rk = np.arange(1, len(fc) + 1)
fc_s, qc_s = fc[order_b], qc[order_b]
ax_b.plot(rk, fc_s, color=C_MAIN, lw=LW,
          label=f"single pool (mean {fc_mean:.2f})")
ax_b.plot(rk, qc_s, color=C_HL, lw=LW, linestyle="--",
          label=f"+ far quota {FAR_QUOTA}/{BUDGET_TOK} ({qc_mean:.2f})")
ax_b.fill_between(rk, fc_s, qc_s, where=qc_s > fc_s,
                  color=C_HL, alpha=0.15, lw=0)
ax_b.axhline(nc_mean, color=C_NEU, lw=1.8, linestyle=":",
             label=f"near capture: {nc_mean:.2f}")
# 最差情况保底箭头（最左端 = 单池捕获最低的 query）
i0 = int(np.argmin(fc_s))
ax_b.annotate("", xy=(rk[i0], qc_s[i0]), xytext=(rk[i0], fc_s[i0]),
              arrowprops=dict(arrowstyle="->", color=C_HL, lw=1.8))
ax_b.text(rk[i0] + 4, 0.055,
          f"worst {fc_min:.2f}→{qc_min:.2f}",
          fontsize=AN - 1.5, color=C_HL, va="center",
          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor="none", alpha=0.85))
ax_b.set_xlabel("128 queries, sorted by single-pool far capture", fontsize=L)
ax_b.set_ylabel("far mass capture\n(oracle top-2048)", fontsize=L - 1)
ax_b.set_xlim(0, len(fc) + 1)
ax_b.set_ylim(0, 1.05)
ax_b.set_title("(b) Quota floors far capture", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_b.tick_params(labelsize=TK)
ax_b.legend(loc="upper left", fontsize=LG - 2, framealpha=0.92)

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

# ---- (d) E6 逐层 far mass 热图 ----
norm6 = mcolors.LogNorm(vmin=1e-7, vmax=1.0)
im = ax_d.imshow(M, aspect="auto", cmap=cmap_mass, norm=norm6,
                 interpolation="nearest")
ax_d.set_xticks([0, 8, 16, 24, 32])
ax_d.set_yticks(range(len(ROW_ORDER)))
ax_d.set_yticklabels([ROW_LAB[k] for k in ROW_ORDER], fontsize=TK - 5)
ax_d.set_xlabel("layer", fontsize=L)
ax_d.set_title("(d) Far mass by layer (16 seqs)", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_d.tick_params(axis="x", labelsize=TK)
cb6 = fig.colorbar(im, ax=ax_d, fraction=0.04, pad=0.015)
cb6.set_label("far mass (log)", fontsize=AN - 2)
cb6.ax.tick_params(labelsize=TK - 3)

fig.subplots_adjust(left=0.10, right=0.93, top=0.94, bottom=0.09)

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
srcb = os.path.basename(SRCB)
rows.append(f"O\ta\trepr_query\tavg_pool_miss_pages\tcount\t{len(missed)}/{len(sel_o)}\t{srcb}\tsingle pool {BUDGET_TOK} tok, 32-dim tail subspace avg page score")
rows.append(f"O\ta\trepr_query\tavg_pool_miss_mass_frac\tfrac\t{miss_frac:.4f}\t{srcb}\tmissed oracle mass / oracle mass")
rows.append(f"O\ta\trepr_query\tminmax_recover_pages\tcount\t{len(recov)} (far {n_rec_far}/{n_miss_far})\t{srcb}\tQuest bound, mean over 32 Q-heads")
if spike:
    rows.append(f"O\ta\trepr_query\tspike_page_top1_token_share\tfrac\t{spike_top1:.4f}\t{srcb}\tpage@{spike['pos_lo']}")
rows.append(f"O\ta\tall_queries\tfar_top5pct_bin_concentration\tfrac\t{conc_mean:.4f}\t{src}\tbins")
for x, msv, s in zip(pos_c, mass_a, sel_mask):
    rows.append(f"O\ta\trepr_strip\tbin_mass\tfrac\t{msv:.4e}\t{src}\tpos={x:.0f},sel={int(s)}")
for (tk, s, f, n, q) in [(t, a, b, cc, dd) for (t, a, b, cc, dd) in
                         zip(tasks, sh, fc, nc, qc)]:
    rows.append(f"O\tb\t{tk}\tshare_far_capture\tfrac\t{s:.4f},{f:.4f},{n:.4f},{q:.4f}\t{src}\tbin mass-rank top-{BUDGET_TOK} (+far quota {FAR_QUOTA})")
rows.append(f"O\tb\tall\tfar_capture_mean\tfrac\t{fc_mean:.4f}\t{src}\t-")
rows.append(f"O\tb\tall\tnear_capture_mean\tfrac\t{nc_mean:.4f}\t{src}\t-")
rows.append(f"O\tb\tdiffuse(share<=0.4)\tfar_capture\tfrac\t{fc_diffuse:.4f}\t{src}\t-")
rows.append(f"O\tb\tquota{FAR_QUOTA}\tfar_capture_mean\tfrac\t{qc_mean:.4f}\t{src}\tfar-region top-{FAR_QUOTA}")
rows.append(f"O\tb\tquota{FAR_QUOTA}\tfar_capture_diffuse\tfrac\t{qc_diffuse:.4f}\t{src}\tfar-region top-{FAR_QUOTA}")
rows.append(f"O\tb\tall\tfar_capture_p10_single_vs_quota\tfrac\t{fc_p10:.4f},{qc_p10:.4f}\t{src}\t-")
rows.append(f"O\tb\tall\tfar_capture_min_single_vs_quota\tfrac\t{fc_min:.4f},{qc_min:.4f}\t{src}\t-")
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
