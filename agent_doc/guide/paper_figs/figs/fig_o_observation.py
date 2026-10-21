# -*- coding: utf-8 -*-
"""O 系列观察图（figure_id=O1/O3 + E6 gate，P0 主文动机图）—— E140v2 数据版。

设计原则：一个面板对应一个创新点的 observation（2026-10-16 用户指示迭代），
叙事链 = 分区+异构打分 → 区域互补漏失 → 子空间 → 逐层 gate（2026-10-21
第四轮重构：nt 降格后旧 (b) far 保底配额面板退役；叙事统一到 E155d 绝对
配额 nb/nl 口径，α/β/γ 比例语言全部移除）：

  (a) 分区+异构打分创新点（token strip，真实数据，四条带）：一条 14K 真实 query
      （musique s1 layer24 q=14212，固定示例披露：far 份额 0.494 = 128 query
      第 55 百分位，非规则产出；同等代表性候选中稀释机理最清晰）的逐
      64-token bin 质量条带（顶，非选集）+ mass-rank 整页参考 / avg / minmax
      三个选择带，对称四态编码（蓝=参考内选中、红斜线=参考漏页、橙点=参考外
      入选、灰=未选；斜线/点纹保证灰度可辨）。near/far 区域底色贯穿全高
      （far 淡高亮色 / near 淡主色），avg/minmax 各自漏掉的参考页落在哪个
      区域一眼可辨——avg 漏的页偏 far、minmax 漏的页偏 near。
      数据源追加 e140b_pooling_anatomy.json（同口径 import 复用 v2 采集器，
      页边界与 bin 逐一对齐断言）。
  (b) 区域互补漏失面板（2026-10-21 新，替代旧 far 保底配额曲线）：128 query
      均值的漏失参考质量按 near/far 区域堆叠分解，block-avg 与 minmax 两柱
      互补镜像——avg 漏失 2/3 集中在 far（0.170 vs 近端 0.084），minmax 漏失
      2/3 集中在 near（0.079 vs 远端 0.042）；near 区两者平手（0.084≈0.079，
      用廉价的 block-avg 无质量代价），far 区 minmax 4× 收益（0.170→0.042）
      → near-avg + far-minmax 异构打分的量化动机。预算分配（近区页数
      nb·64 vs 近区长 nl）是设计轴，由 E155d 网格判决（半覆盖律
      nb·64=nl/2，A6 图），不在本诊断平面内模拟（mass≠e2e）。
  (c) 子空间创新点（不变）：同 32/128 维预算 tail32 0.84 vs random32 0.51
      vs high32 0.24（v2 主指标 selected_mass_over_reference_topk_mass）。
  (d) D' gate 创新点：E6 逐层 far mass 热图（16 样本 × 36 层，log 色标，
      单一来源 e6b_far_profiles.json）——24/36 层逐样本中位 <0.01；同一
      热图逐 prompt 计数有 12–32 层 <0.01 → 逐层 far gate。

2026-10-16 第三轮迭代（用户指示「继续迭代你的」源图线，手稿侧归 GPT）：
  6) panel (a) 从三条带（质量/oracle/avg+✓）改为四条带对称呈现——顶带逐页
     质量（非选集）+ mass-rank 整页参考 / avg / minmax 三个选择带，吸收手稿侧
     GPT 7d8c397 的对称设计回源图；「oracle」更名「mass-rank 参考」（整页质量
     排序只是比较参考，非 token 级上界——GPT caption 同口径）；
  7) 灰度可辨编码（吸收 GPT d1d0d6e）：参考漏页红底+斜线、参考外入选橙底+
     点纹、黑边，去 ✓ 文字标记，四态无需颜色即可区分；
  8) 新增对称差集统计：minmax 页数漏更多（17>13）但质量漏损更低
     （26.5%<59.7%；128q 均值 12.1% vs 25.4%）——页数与质量口径分离。

2026-10-16 GPT 统计审查（e5f5b7e）修正并入：
  1) 预算：所有选择循环改为接纳前检查 acc+n_tok<=预算，放不下跳过
     （旧版先接纳后检查，全局/oracle/avg 至多 97/128 条超额、最大 2110 token）；
  2) 区域归属：bin 绝对区间 [q-d_hi, q-d_lo+1) 对照 region_bounds.near_lo
     （旧规则 d_lo>=near_len 把 65/128 个近区 token 误计入 far；far 份额
     中位数 0.49550018 → 0.45642671）；本批 bin 均不跨界，fail-closed 断言；
  3) panel (b) 两臂最小值分离披露 + 逐 query 差值保留负号（该面板已于
     2026-10-21 第四轮退役，见下）；
  4) 代表点披露统一为固定示例（撤销 docstring/TSV 残留的 closest-median 规则
     说法——该旧规则实际产出 L8 q13158，与固定点不符）；
  5) panel (d) 撤销 e6_layer_skip.json（仅 10 prompt 的异源摘要），skip 范围
     改从同一张 16×36 热图逐行计数（交叉验证 hotpotqa_0=22 等与 GPT 一致）。

2026-10-21 第四轮重构（用户指示：体现 avg 漏远端/minmax 漏近端的互补、
口径切 E155d nb/nl 绝对配额、nt 挤兑面板退役）：
  6) 旧 (b)「quota floors far capture」曲线（far 保底 1024/2048 配对臂，
     β 比例叙事）整体退役——nt 降格为安全配额后挤兑/保底不再是动机面板；
  7) 新 (b) = 区域互补漏失堆叠柱（128 query 均值，near/far 分解；页级
     区域归属沿用修正后绝对区间口径，bin↔page 一一对齐断言）；
  8) (a) 选择条带加 near/far 区域底色（贯穿全高），漏页区域归属可视化；
  9) 区域预算敏感性已离线扫描（near 预算 128..1536，hybrid 最优 0.137
     @1280 仍高于全局 minmax 0.121）：预算分配属 E155d 网格轴（任务精度
     口径），本图不做 hybrid 模拟臂（mass≠e2e，防口径混淆）。

数据源（全部为有效输入，不再使用 O4 v2——其部分末页缺陷见 E140_VALIDITY.md，
本图不含 O4 面板，因此无需历史诊断警示）：
  e140_o_trace_v2.json（O1/O3，采集器已修复：实际 q_norm/k_norm + 实际 RoPE）
  e140b_pooling_anatomy.json（E140b 页级 pooling 解剖，v2 预算修正版）
  e6b_far_profiles.json（E6 逐层 far profile，gate 标定同源；panel d 单一来源）

输出 PDF+PNG+SVG + figure_data.tsv（含 (b) 全部 128 query 的区域分解漏失、
(d) 全矩阵）。
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

SKILL = "/home/wangyuanshuo02/.claude/skills/publication-figures/scripts"
sys.path.insert(0, SKILL)
from style_config import apply_style  # noqa: E402
from figure_lint import lint_figure  # noqa: E402

REPO = "/home/wangyuanshuo02/two-level-attention"
SRC = os.path.join(REPO, "exp/trace/results/e140_o_trace_v2.json")
SRCB = os.path.join(REPO, "exp/trace/results/e140b_pooling_anatomy.json")
SRC6B = os.path.join(REPO, "exp/trace/results/e6b_far_profiles.json")
for _p in (SRC, SRCB, SRC6B):
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
# repr query 固定为 musique s1 L24 q=14212（披露：固定示例，非规则产出——
# far 份额 0.494 = 128 query 第 55 百分位、中位 0.456，区域质量分段有代表性；
# 同等代表性候选中其 avg 稀释机理最清晰。身份与手稿 caption 一致）
BUDGET_TOK = 2048  # 单池预算（mid token）


def bin_region(o, b):
    """bin 绝对区间 [q-d_hi, q-d_lo+1) 对照 region_bounds.near_lo 归属近远。

    GPT 2026-10-16 审查修正：旧规则 d_lo>=near_len 比较的是「距离」与「近区
    长度」两个不同量纲，在全部 128 行把 65 或 128 个近区 token 误计入 far
    （far 份额中位 0.4955→0.4564）。本批 bin 均不跨近远边界，fail-closed。
    """
    q, nl = o["q_pos"], o["region_bounds"]["near_lo"]
    lo, hi = q - b["d_hi"], q - b["d_lo"] + 1
    assert not (lo < nl < hi), \
        f"bin straddles near boundary: q={q} ({lo},{hi}) near_lo={nl}"
    return "far" if hi <= nl else "near"


def far_share(o):
    fm = sum(b["mass"] for b in o["bins"] if bin_region(o, b) == "far")
    mm = sum(b["mass"] for b in o["bins"])
    return fm / mm if mm > 0 else 0.0


share_med = float(np.median([far_share(o) for c in cells for o in c["o1"]]))
share_all = np.array([far_share(o) for c in cells for o in c["o1"]])
c_a = next(c for c in cells if c["task"] == "musique" and c["sample"] == 1
           and c["layer"] == 24)
o_a = next(o for o in c_a["o1"] if o["q_pos"] == 14212)
share_a = far_share(o_a)
share_pct = float((share_all < share_a).mean() * 100)
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

# 单池 top-2048（按实测 bin 质量排序 = 任意打分单池的质量捕获上界；
# 接纳前检查 acc+n_tok<=预算，放不下则跳过——GPT 审查修正：旧版先接纳后
# break，末页不足 64 token 时全局/oracle 至多 97/128 条超额、最大 2110）
rank_idx = np.argsort(-mass_a)
acc = 0
sel_mask = np.zeros(len(mass_a), bool)
for i in rank_idx:
    if acc + ntok_a[i] > BUDGET_TOK:
        continue
    sel_mask[i] = True
    acc += ntok_a[i]
oracle_tok_a = int(acc)  # (b) 循环会复用 acc 变量名，此处留存
pos_c = np.array([(q_a - b["d_hi"] + q_a - b["d_lo"]) / 2 for b in bins_a])[order]
is_far = np.array([bin_region(o_a, b) == "far" for b in bins_a])[order]
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
missed = sorted(sel_o - sel_avg)          # 参考选中但 avg 漏掉的页
recov = sorted(set(missed) & sel_mm)      # 其中被 minmax 上界救回的页
far_set = set(np.where(is_far)[0].tolist())
n_miss_far = len(set(missed) & far_set)
n_rec_far = len(set(recov) & far_set)
miss_frac = qa_q["miss_mass_frac"]
# 对称差集与质量口径（2026-10-16 第三轮：页数与质量分开陈述）
pg_mass = pg["mass"]
o_mass = sum(pg_mass[i] for i in sel_o)
mm_missed = sorted(sel_o - sel_mm)                    # minmax 仍漏的参考页
avg_extra = sorted(sel_avg - sel_o)                   # avg 参考外入选
mm_extra = sorted(sel_mm - sel_o)                     # minmax 参考外入选
mm_miss_frac = sum(pg_mass[i] for i in mm_missed) / o_mass
recov_mass_frac = sum(pg_mass[i] for i in recov) / o_mass
# 128q 汇总：avg vs minmax 的参考质量漏损率 + near/far 区域分解
# （页级区域归属 = 该页对齐 bin 的绝对区间口径，bin↔page 一一对齐断言；
#   对不齐的 query 只计入总体漏损、不计入区域统计——fail-closed）
ams_all, mms_all = [], []
reg_all = []   # (id, avg(near,far,npg_near,npg_far), mm(...), n_ref)
_o1_cells = {(c["task"], c["sample"], c["layer"]): c for c in cells}


def _reg_miss(pages, regs, sel_o, sel, om):
    """漏失页的 near/far 质量份额与页数（om=参考集总质量）。"""
    ms = sel_o - sel
    return (sum(pages["mass"][i] for i in ms if regs[i] == "near") / om,
            sum(pages["mass"][i] for i in ms if regs[i] == "far") / om,
            sum(1 for i in ms if regs[i] == "near"),
            sum(1 for i in ms if regs[i] == "far"))


for _c in db["cells"]:
    _oc = _o1_cells.get((_c["task"], _c["sample"], _c["layer"]))
    _o1_by_q = {o["q_pos"]: o for o in _oc["o1"]} if _oc else {}
    for _q in _c["queries"]:
        _p = _q["pages"]
        _so, _sa, _sm = (set(_p["sel_oracle"]), set(_p["sel_avg"]),
                         set(_p["sel_minmax"]))
        _om = sum(_p["mass"][i] for i in _so)
        if _om <= 0:
            continue
        ams_all.append(sum(_p["mass"][i] for i in _so - _sa) / _om)
        mms_all.append(sum(_p["mass"][i] for i in _so - _sm) / _om)
        _o = _o1_by_q.get(_q["q_pos"])
        if _o is None:
            continue
        _bl = {_o["q_pos"] - b["d_hi"]: b for b in _o["bins"]}
        _regs, _ok = {}, True
        for _i, _plo in enumerate(_p["pos_lo"]):
            if _plo not in _bl:
                _ok = False
                break
            _regs[_i] = bin_region(_o, _bl[_plo])
        if not _ok:
            continue
        reg_all.append((
            f"{_c['task']}:s{_c['sample']}:L{_c['layer']}:q{_q['q_pos']}",
            _reg_miss(_p, _regs, _so, _sa, _om),
            _reg_miss(_p, _regs, _so, _sm, _om), len(_so)))
ams_mean = float(np.mean(ams_all))
mms_mean = float(np.mean(mms_all))
# 区域分解均值（质量份额 + 页数）——(b) 面板主数据
avg_near_m = float(np.mean([r[1][0] for r in reg_all]))
avg_far_m = float(np.mean([r[1][1] for r in reg_all]))
mm_near_m = float(np.mean([r[2][0] for r in reg_all]))
mm_far_m = float(np.mean([r[2][1] for r in reg_all]))
avg_near_p = float(np.mean([r[1][2] for r in reg_all]))
avg_far_p = float(np.mean([r[1][3] for r in reg_all]))
mm_near_p = float(np.mean([r[2][2] for r in reg_all]))
mm_far_p = float(np.mean([r[2][3] for r in reg_all]))
spike = qa_q["detail"].get("spike")
if spike is None:
    raise SystemExit("repr query lacks spike-page detail")
tm = spike["token_mass"]
spike_top1 = max(tm) / sum(tm)

# 跨 query 集中度：far 前 5% bin 携带的 far 质量比（修正后区域归属）
conc = []
for c in cells:
    for o in c["o1"]:
        fms = sorted((b["mass"] for b in o["bins"]
                      if bin_region(o, b) == "far"), reverse=True)
        if fms and sum(fms) > 0:
            k = max(1, int(0.05 * len(fms)))
            conc.append(sum(fms[:k]) / sum(fms))
conc_mean = float(np.mean(conc))

# ================= (b) 区域互补漏失（2026-10-21 新，128 queries） =================
# 旧「quota floors far capture」配对臂已退役（nt 降格 + nb/nl 绝对配额口径）。
# 主数据 = reg_all 区域分解均值：avg 漏失 2/3 在 far（0.170 vs 0.084），
# minmax 漏失 2/3 在 near（0.079 vs 0.042）——near 区平手（avg 无质量代价）、
# far 区 minmax 4× 收益 → near-avg + far-minmax 异构打分动机。
# 预算分配（nb/nl 轴）由 E155d 网格判决（A6 图），本面板不做 hybrid 模拟。

# ================= (c) O3 三臂（主指标 selected/reference topK mass） =================
arms = ["tail32", "random32", "high32"]
# xtick 单行（2026-10-21：两行 xtick 字符级真实互叠 max 40px²，
# 副行语义进 caption/元数据）
arm_lab = {"tail32": "tail32", "random32": "random32", "high32": "high32"}
arm_rm = {a: float(np.mean([c["o3"][a]["selected_mass_over_reference_topk_mass"]["mean"]
                            for c in cells])) for a in arms}
mets_all = ["reference_token_page_coverage", "reference_token_overlap_recall",
            "selected_mass_over_reference_topk_mass", "selected_full_attention_mass"]
arm_val = {a: [float(np.mean([c["o3"][a][mm]["mean"] for c in cells]))
               for mm in mets_all] for a in arms}

# ================= (d) E6 逐层 far mass 热图 =================
# GPT 审查点 5：撤销 e6_layer_skip.json（仅 10 prompt 的异源摘要），
# 逐 prompt 计数改从同一张 16×36 热图按 <0.01 重数（单一来源）。
d6b = json.load(open(SRC6B))
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
skip_rng = (M < 0.01).sum(axis=1)  # 逐 prompt <0.01 层数（同源重算）

# ================= 画布 2x2（IEEE 双栏 \textwidth≈7.16in，缩放比 ≈0.75，
# 有效字号 ≈ title 11 / label 9.8 / tick 8.6pt） =================
# 2026-10-16 用户指示：图内不堆说明文字，面板数字与结论全部并入手稿 caption，
# 图内仅保留坐标/图例/带名/区域标签等最小可读元素。
fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
    2, 2, figsize=(9.5, 6.9),
    gridspec_kw={"wspace": 0.48, "hspace": 0.52,
                 "width_ratios": [1.2, 1.0]})


# ---- (a) token strip：质量带 + 三个选择带（对称四态编码，灰度可辨） ----
cmap_mass = mcolors.LinearSegmentedColormap.from_list("mass", ["#ffffff", C_MAIN])
norm_mass = mcolors.LogNorm(vmin=2e-4, vmax=2e-2)
X0, X1 = float(min(pos_c)), float(max(pos_c))
n_pg = len(mass_a)
w_pg = (X1 - X0) / n_pg  # bin 等宽近似（bins 非连续覆盖，见上文注释）
C_MISS = "#d62728"
C_EXTRA = "#e08214"
# 顶带：逐 bin 质量（log 色标，是统计量不是选集）
ax_a.imshow(mass_a[None, :], aspect="auto", cmap=cmap_mass, norm=norm_mass,
            extent=[X0, X1, 3.0, 4.0], interpolation="nearest")
# 三个选择带：mass-rank 整页参考 / avg / minmax。每页四态：
# 蓝 = 参考内选中，红+斜线 = 参考漏页，橙+点纹 = 参考外入选，灰 = 未选。
# （吸收手稿侧 GPT 7d8c397/d1d0d6e 的对称与灰度设计回源图）
for sel, y0 in ((sel_o, 2.0), (sel_avg, 1.0), (sel_mm, 0.0)):
    for i in range(n_pg):
        in_ref, in_x = i in sel_o, i in sel
        if in_ref and in_x:
            fc_i, ht_i = C_MAIN, None
        elif in_ref:
            fc_i, ht_i = C_MISS, "///"
        elif in_x:
            fc_i, ht_i = C_EXTRA, ".."
        else:
            fc_i, ht_i = "#e9e9e9", None
        ax_a.add_patch(Rectangle(
            (X0 + i * w_pg, y0), w_pg, 1.0, facecolor=fc_i, hatch=ht_i,
            edgecolor="black" if ht_i else "none",
            lw=0.5 if ht_i else 0.0, zorder=3))
# sink / swa 保护区（不在 mid bins 内；区域太窄无法内嵌文字）
ax_a.axvspan(0, rb["sink_hi"], color=C_NEU, alpha=0.35, lw=0)
ax_a.axvspan(rb["swa_lo"], q_a + 1, color=C_NEU, alpha=0.35, lw=0)
# near/far 区域底色贯穿全高（2026-10-21：漏页区域归属可视化——
# avg 漏的红斜线页偏右（far 底色带）、minmax 漏的偏左（near 底色带）；
# 与 (b) 面板堆叠配色一致：far=C_HL、near=C_MAIN；alpha 0.12 为像素
# 采样实测的一眼可辨强度，四态矩形 zorder=3 不受影响）
ax_a.axvspan(rb["sink_hi"], rb["near_lo"], color=C_HL, alpha=0.12,
             lw=0, zorder=2.5)
ax_a.axvspan(rb["near_lo"], rb["swa_lo"], color=C_MAIN, alpha=0.12,
             lw=0, zorder=2.5)
# near/far 区域标注（置于顶带中部）
_tagbox = dict(boxstyle="round,pad=0.2", facecolor="white",
               edgecolor="none", alpha=0.85)
# near/far 区域标注（顶带；far 标签置于图例右缘与 near 标签之间的空窗，
# 中心 x10500；far token 占比数字进 caption/TSV——版式铁律）
ax_a.text(10500, 3.5, "far", ha="center",
          va="center", fontsize=AN - 2, color=C_HL, fontweight="bold",
          bbox=_tagbox)
ax_a.text((rb["near_lo"] + rb["swa_lo"]) / 2, 3.5, "near", ha="center",
          fontsize=AN, color=C_MAIN, fontweight="bold", bbox=_tagbox)
ax_a.axvline(rb["near_lo"], color=C_HL, lw=1.6, linestyle="--")
# 尖峰页标记（▼，页 @spike.pos_lo：单 token 占页质量 99.8%，被 avg 稀释落选，
# minmax 救回；说明在 caption）
i_sp = bin_lo.index(spike["pos_lo"])
ax_a.plot([pos_c[i_sp]], [4.12], marker="v", color=C_MISS, markersize=8,
          clip_on=False, zorder=6)
# 紧凑图例（用户 2026-10-16：图内不堆文字，数字解释并入 caption；
# 2026-10-21 标签缩短——ncol=2 长标签组占满顶带左 60% 致 far 区标注无处安放）
leg_a = [Patch(fc=C_MAIN, label="in ref, selected"),
         Patch(fc=C_MISS, hatch="///", ec="black", lw=0.5,
               label="ref missed"),
         Patch(fc=C_EXTRA, hatch="..", ec="black", lw=0.5,
               label="outside ref"),
         Patch(fc="#e9e9e9", ec="0.6", label="unselected")]
ax_a.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(0.005, 1.005),
            fontsize=LG - 3, framealpha=0.92, handlelength=1.2,
            borderpad=0.35, labelspacing=1.3, ncol=2, columnspacing=0.9)
if os.environ.get("O_DEBUG_BB"):  # 量 legend/tag 数据坐标盒（布局调试用）
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _inv = ax_a.transData.inverted()
    for _art in ax_a.get_legend(), ax_a.texts[-2], ax_a.texts[-1]:
        _bb = _art.get_window_extent(_r).transformed(_inv)
        print(f"[bb] {str(_art)[:40]} x:[{_bb.x0:.0f},{_bb.x1:.0f}] "
              f"y:[{_bb.y0:.2f},{_bb.y1:.2f}]")
ax_a.set_xlim(0, q_a + 1)
ax_a.set_xticks(range(0, 15000, 3000))
ax_a.set_ylim(0, 4.0)
ax_a.set_yticks([0.5, 1.5, 2.5, 3.5])
ax_a.set_yticklabels(["minmax", "avg", "mass ref", "attn mass"],
                     fontsize=TK - 1.5)
ax_a.set_xlabel("key position (tokens); query at right end", fontsize=L,
                labelpad=12)
ax_a.set_title("(a) Avg pool misses sparse far spikes",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_a.tick_params(axis="x", labelsize=TK)
sm = plt.cm.ScalarMappable(cmap=cmap_mass, norm=norm_mass)
cb = fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.015)
cb.set_label("mass (log)", fontsize=AN - 1.5)
cb.ax.tick_params(labelsize=TK - 3)

# ---- (b) 区域互补漏失堆叠柱（128 query 均值）----
# 设计：两根柱 = block-avg / minmax 打分器，各自漏失的参考质量按区域堆叠
# （底段 near=C_MAIN、顶段 far=C_HL，与 (a) 区域底色/标注配色一致）。
# 数字（0.084/0.170/0.079/0.042 与页数口径）全部进 caption/TSV（版式铁律）。
xb = np.arange(2)
near_vals = [avg_near_m, mm_near_m]
far_vals = [avg_far_m, mm_far_m]
ax_b.bar(xb, near_vals, width=0.52, color=C_MAIN,
         edgecolor="black", linewidth=.7, label="near missed")
ax_b.bar(xb, far_vals, width=0.52, bottom=near_vals, color=C_HL,
         edgecolor="black", linewidth=.7, label="far missed")
ax_b.set_xticks(xb)
ax_b.set_xticklabels(["block-avg", "minmax"], fontsize=TK)
ax_b.set_ylabel("missed reference mass", fontsize=L)
ax_b.set_ylim(0, 0.30)
ax_b.set_yticks([0.0, 0.1, 0.2, 0.3])
ax_b.tick_params(axis="y", labelsize=TK)
ax_b.set_title("(b) Missed mass by region", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_b.legend(loc="upper right", fontsize=LG - 2, framealpha=0.92,
            labelspacing=0.75)

# ---- (c) O3 三臂 ----
colors3 = [C_MAIN, C_NEU, C_SEC]
for i, (a, col) in enumerate(zip(arms, colors3)):
    ax_c.bar([i], [arm_rm[a]], width=0.58, color=col, edgecolor="black",
             linewidth=1.2)
    ax_c.text(i, arm_rm[a] + 0.025, f"{arm_rm[a]:.2f}", ha="center",
              fontsize=AN, fontweight="bold")
ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels([arm_lab[a] for a in arms], fontsize=TK - 1)
ax_c.set_ylabel("selected / reference mass", fontsize=L)
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
ax_d.set_yticklabels([ROW_LAB[k] for k in ROW_ORDER], fontsize=TK - 5.3)
ax_d.set_xlabel("layer", fontsize=L, labelpad=12)
ax_d.set_title("(d) Far mass by layer", loc="left", pad=10,
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
rows.append(f"O\ta\trepr_query\tidentity\t-\t{c_a['task']}:s{c_a['sample']}:L{c_a['layer']}:q{q_a}\t{src}\tfixed illustrative example (not rule-derived): far share {share_a:.4f} = pct{share_pct:.0f} of 128 queries, median {share_med:.4f}; clearest dilution anatomy among comparably representative candidates")
rows.append(f"O\ta\trepr_query\tfar_share\tfrac\t{share_a:.4f}\t{src}\tbins, absolute-position region attribution")
rows.append(f"O\ta\tall_queries\tfar_share_median\tfrac\t{share_med:.4f}\t{src}\tbins, absolute-position region attribution")
rows.append(f"O\ta\trepr_query\tfar_tok_frac\tfrac\t{far_tok_frac:.4f}\t{src}\tbins")
rows.append(f"O\ta\trepr_query\tsingle_pool_near_capture\tfrac\t{ncap_a:.4f}\t{src}\tbin mass-rank top-{BUDGET_TOK}, accept-if-fits")
rows.append(f"O\ta\trepr_query\tsingle_pool_far_capture\tfrac\t{fcap_a:.4f}\t{src}\tbin mass-rank top-{BUDGET_TOK}, accept-if-fits")
rows.append(f"O\ta\trepr_query\toracle_strip_tokens\tcount\t{oracle_tok_a}\t{src}\tbudget {BUDGET_TOK}, accept-if-fits (GPT-review fix)")
srcb = os.path.basename(SRCB)
rows.append(f"O\ta\trepr_query\tavg_pool_miss_pages\tcount\t{len(missed)}/{len(sel_o)}\t{srcb}\tsingle pool {BUDGET_TOK} tok, 32-dim tail subspace avg page score")
rows.append(f"O\ta\trepr_query\tavg_pool_miss_mass_frac\tfrac\t{miss_frac:.4f}\t{srcb}\tmissed mass-rank-ref mass / ref mass")
rows.append(f"O\ta\trepr_query\tminmax_recover_pages\tcount\t{len(recov)} (far {n_rec_far}/{n_miss_far})\t{srcb}\tQuest bound, mean over 32 Q-heads")
rows.append(f"O\ta\trepr_query\tminmax_recover_mass_frac\tfrac\t{recov_mass_frac:.4f}\t{srcb}\trecovered pages' mass / ref mass")
rows.append(f"O\ta\trepr_query\tminmax_miss_pages\tcount\t{len(mm_missed)}/{len(sel_o)}\t{srcb}\tsymmetric: minmax misses MORE pages than avg")
rows.append(f"O\ta\trepr_query\tminmax_miss_mass_frac\tfrac\t{mm_miss_frac:.4f}\t{srcb}\tbut LESS ref mass than avg ({miss_frac:.4f}): page-count vs mass criteria differ")
rows.append(f"O\ta\trepr_query\tavg_extra_pages\tcount\t{len(avg_extra)}\t{srcb}\tselected outside mass-rank reference")
rows.append(f"O\ta\trepr_query\tminmax_extra_pages\tcount\t{len(mm_extra)}\t{srcb}\tselected outside mass-rank reference")
rows.append(f"O\ta\tall_queries\tmiss_mass_frac_avg_vs_minmax\tfrac\t{ams_mean:.4f},{mms_mean:.4f}\t{srcb}\tmean over 128 queries; minmax misses less ref mass")
if spike:
    rows.append(f"O\ta\trepr_query\tspike_page_top1_token_share\tfrac\t{spike_top1:.4f}\t{srcb}\tpage@{spike['pos_lo']}")
rows.append(f"O\ta\tall_queries\tfar_top5pct_bin_concentration\tfrac\t{conc_mean:.4f}\t{src}\tbins")
for x, msv, s in zip(pos_c, mass_a, sel_mask):
    rows.append(f"O\ta\trepr_strip\tbin_mass\tfrac\t{msv:.4e}\t{src}\tpos={x:.0f},sel={int(s)}")
for rid, am, mm, nref in reg_all:
    rows.append(f"O\tb\t{rid}\tmiss_mass_by_region\tfrac\tavg_near {am[0]:.4f},avg_far {am[1]:.4f},mm_near {mm[0]:.4f},mm_far {mm[1]:.4f}\t{srcb}\tn_ref={nref},n_miss_avg_near {am[2]},n_miss_avg_far {am[3]},n_miss_mm_near {mm[2]},n_miss_mm_far {mm[3]}")
rows.append(f"O\tb\tall\tmiss_mass_block_avg\tnear,far\t{avg_near_m:.4f},{avg_far_m:.4f}\t{srcb}\tmean over {len(reg_all)} queries; region attribution = aligned-bin absolute interval [q-d_hi, q-d_lo+1) vs region_bounds.near_lo")
rows.append(f"O\tb\tall\tmiss_mass_minmax\tnear,far\t{mm_near_m:.4f},{mm_far_m:.4f}\t{srcb}\tsame caliber; avg misses far (2/3 of its miss), minmax misses near (2/3 of its miss): complementary")
rows.append(f"O\tb\tall\tmiss_mass_totals\tavg,mm\t{ams_mean:.4f},{mms_mean:.4f}\t{srcb}\tmean over {len(ams_all)} queries (region-unresolved included)")
rows.append(f"O\tb\tall\tmiss_pages_block_avg\tnear,far\t{avg_near_p:.2f},{avg_far_p:.2f}\t{srcb}\tmean missed pages; page-count criterion differs from mass criterion")
rows.append(f"O\tb\tall\tmiss_pages_minmax\tnear,far\t{mm_near_p:.2f},{mm_far_p:.2f}\t{srcb}\tmean missed pages")
rows.append(f"O\tb\tretired\tquota_floor_panel\t-\t-\t-\t2026-10-21 fourth revision: old far-quota paired-arm panel withdrawn (nt demoted; nb/nl absolute-quota caliber). Hybrid near-budget scan (offline, diagnostic plane): best 0.137@near-budget-1280 still above global minmax 0.121; budget split is the E155d grid axis (task-accuracy caliber), not simulated here (mass != e2e)")
for a in arms:
    for mm, v in zip(mets_all, arm_val[a]):
        rows.append(f"O\tc\t{a}\t{mm}\tfrac\t{v:.4f}\t{src}\to3.{a}.{mm}.mean")
for k, row in zip(ROW_ORDER, M):
    for li, v in enumerate(row):
        rows.append(f"O\td\t{k}\tfar_mass\tfrac\t{v:.4e}\te6b_far_profiles.json\tlayer={li}")
rows.append(f"O\td\tall\tlayers_median_lt_0.01\tcount\t{n_light}\te6b_far_profiles.json\t-")
rows.append(f"O\td\tall\tlayers_lt_0.01_per_prompt\trange\t{int(skip_rng.min())}-{int(skip_rng.max())}\te6b_far_profiles.json\trecomputed per-row from the same 16x36 heatmap (single source; GPT-review fix: e6_layer_skip.json 10-prompt summary withdrawn)")
with open(os.path.join(OUT, "O_observation_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported O_observation_figure_data.tsv")
print("DONE")
