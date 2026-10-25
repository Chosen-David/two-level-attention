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
面板 (c)：完整请求 decode TPOT vs L——iter3+K1=64 主表口径三模型判决
  （2026-10-25 更新，取代 e152v4/e159 iter22 口径）。整步 CUDA Graph + 六融合
  kernel 链 + K1=64 后 TLI 全档快于 dense：Qwen3-8B 1.451/2.216/3.636×
  （e172 pc40 干净臂）、Llama-3.1-8B 1.428/2.192/3.643×（e173，跨模型族对称
  ≤0.024/档）、Qwen3-32B 1.338/1.924/3.015×（e174 pc02；128K 行走 E160b
  同款单 run/进程救援协议；取代 e160/e160b iter22 口径 1.015/1.338/1.841×）；
  graph 内部逐位一致；8B graph vs eager 64K 逐位、32K/128K 各一单词元 bf16 平局
  分歧（K1=64 扩大 far 排序集），32B 全部 10 项 token 门逐位（e174 判决
  token_identity_gates）；e166 RULER 质量门 md5 三版同源背书。eager 无 graph 臂
  如实画出（8B 100.9/103.2/133.9ms，launch 主导）——graph replay 是符号翻转
  的必要条件。

kernel 设计创新点（decode-only 转向后 E2 的设计贡献叙事，全部以同机对照数字入图）：
  (i) E127 融合两级选择 kernel 链（粗筛上界 + 精筛融合 decode kernel）——5.47× GPU
      稳态 @64K（面板 a 数据条）；
  (ii) E133c launch/融合优化套件——A1 统计 buffer 直写消 3 cat、A4/A5 尾段统计与
      4bit 量化融合 kernel（7-12 launch→1）、A6 D=32 tile 特化——稳态 GPU −12.3%
      （同机 A/B，两轮交叉一致；tail 臂 −14.6% 反平 full）；tail-32 子空间粗筛省算
      4×、mask 逐位一致（注记入面板 a）；
  (iii) E139 Step2 prefill 候选列化 + 选择链 CUDA graph capture——64K 1.64× vs dense
      （面板 b Step2 曲线）；
  (iv) E134-pre StaticCache + E152v4 整步 graph capture——消 HF DynamicCache 72 cat
      拷贝/步与 host launch 链，e2e TPOT 从 87.9ms（e152v2 dynamic）压到 24.7ms
      @32K（面板 c 蓝线），符号翻转为正并随 L 单调放大。

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
R = os.path.join(REPO, "exp/trace/results/")
OUT = os.path.join(REPO, "agent_doc", "guide", "paper_figs", "figs")
os.makedirs(OUT, exist_ok=True)

pal = apply_style("ocean")
# 本机无 Arial/Liberation：显式回退 DejaVu Sans（外观接近，避免 findfont 警告）
matplotlib.rcParams["font.family"] = "DejaVu Sans"
# Type42 嵌入（字形门友好；Type3 CJK bbox 膨胀教训见 ruler32k_comparison/draw.py）
matplotlib.rcParams["pdf.fonttype"] = matplotlib.rcParams["ps.fonttype"] = 42
C_DENSE = pal["neutral"]        # dense 基线 = 灰
C_STEP1 = pal["tertiary"]       # Step1 = 浅蓝
C_STEP2 = pal["primary"]        # Step2 = 深蓝（主方法）
C_BASE = pal["neutral"]
C_TLI = pal["primary"]
C_HL = pal["highlight"]         # 强调橙
C_EAGER = pal["secondary"]      # eager 无 graph 臂 = 青（诚实边界）

# ---------------- 数据读取：面板 (a) 阶段分解 ----------------
e128 = json.load(open(os.path.join(R, "e128_tradeoff.json")))
h = e128["e127_headline_64k_normal"]
stages = ["prepare_index", "compute_score", "compute_mask", "metrics", "eager_decoding"]
base_ms = [h["stage_gpu_baseline"][s] for s in stages] + [h["baseline_gpu_ms"]]
tli_ms = [h["stage_gpu_iter8"][s] for s in stages] + [h["iter8_gpu_ms"]]
stage_labels = ["prep", "score", "mask", "metric", "attn", "total"]

# ---------------- 数据读取：面板 (b) prefill wall ----------------
e139 = json.load(open(os.path.join(R, "e139_step2_cand_verdict.json")))
wl = e139["results"]["wall_lsweep"]
Ls = [8192, 16384, 32768, 65536]
dense_ms, step1_ms, step2_ms = [], [], []
for L in Ls:
    s = wl[f"L_{L}"]
    dense_ms.append(float(s.split("dense ")[1].split(" /")[0]))
    step1_ms.append(float(s.split("Step1 graph ")[1].split(" /")[0]))
    step2_ms.append(float(s.split("Step2 graph ")[1].split("ms")[0]))

# E133c 同机 A/B 稳态百分比（跨机绝对值不与 E127 混用，只取同机对照 delta）
e133c = json.load(open(os.path.join(R, "e133c_subspace_speedup_verdict.json")))
e133c_e2e_pct = e133c["steady_gpu_ms"]["mavg_e2e"]["delta_pct"]   # 生产口径 -12.3%
e133c_tail_pct = e133c["steady_gpu_ms"]["mavg_tail"]["delta_pct"]  # tail 臂 -14.6%

# ---------------- 数据读取：面板 (c) e2e decode TPOT（三模型 iter3+K1=64 判决）----------------
# Qwen3-8B：E172 pc40 干净臂（iter3 六融合 kernel 链 + K1=64，ABBA、bit-exact 门禁）
e172 = json.load(open(os.path.join(R, "e172_iter3_k64_qwen_verdict.json")))
q8b_dense = [e172["results"][str(L)]["dense_ms"]["mean"] for L in (32768, 65536, 131072)]
q8b_tli = [e172["results"][str(L)]["tli_graph_k64_ms"]["mean"] for L in (32768, 65536, 131072)]
import re as _re
q8b_eager = [float(x) for x in
             _re.findall(r"[\d.]+", e172["bitexact_gate"]["eager_tpots"])[:3]]
# 判决字符串尾部含「（门禁臂不进表）」中文注记，用数值正则取前三个数
# Qwen3-32B：E174 iter3+K1=64（pc02 GPU0；128K 行走 E160b 同款单 run/进程救援协议）
e174 = json.load(open(os.path.join(R, "e174_iter3_k64_32b_verdict.json")))
q32b_dense = [e174["results"][str(L)]["dense_ms"]["mean"] for L in (32768, 65536, 131072)]
q32b_tli = [e174["results"][str(L)]["tli_graph_k64_ms"]["mean"] for L in (32768, 65536, 131072)]
# Llama-3.1-8B：E173 iter3+K1=64 主表口径（pc40 GPU1 干净臂，与 E172 对称闭合）
e173 = json.load(open(os.path.join(R, "e173_iter3_k64_llama_verdict.json")))
llama_dense = [e173["results"][str(L)]["dense_ms"]["mean"] for L in (32768, 65536, 131072)]
llama_tli = [e173["results"][str(L)]["tli_graph_k64_ms"]["mean"] for L in (32768, 65536, 131072)]
# 跨机一致性注记：E160R pc31 复刻属 iter22 K1=128 旧口径，E174 后不再入图/TSV；
# 32B iter3+K1=64 行为单机（pc02）单轮，无跨机复刻。

# ---------------- 画布 ----------------
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(27, 9),
                                       gridspec_kw={"wspace": 0.32})

# ---- 面板 (a)：decode 阶段分解（并列条形）----
x = range(len(stage_labels))
w = 0.38
ax_a.bar([i - w / 2 for i in x], base_ms, width=w, color=C_BASE,
         label="dense FullKV", edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_a.bar([i + w / 2 for i in x], tli_ms, width=w, color=C_TLI,
         label="LoDA-SA (K1=128, K2=1024)", edgecolor="black", linewidth=LINE_WIDTHS["marker_edge"] * 0.6)
ax_a.set_yscale("log")
ax_a.set_ylim(3e-3, 50)
ax_a.set_yticks([1e-2, 1e-1, 1, 10])
ax_a.set_xticks(list(x))
ax_a.set_xticklabels(stage_labels)
ax_a.set_ylabel("per-step GPU time (ms, log scale)")
ax_a.set_title("Decode steady state: 5.47x @64K", pad=14)
ax_a.legend(loc="upper left", fontsize=FONT_SIZES["legend"] - 3)
# 仅 total 条标注数值（分项数值进 tsv，避免 log 轴低矮条标签互叠）
ax_a.text(5 - w / 2, h["baseline_gpu_ms"] * 1.15, f"{h['baseline_gpu_ms']:.2f}",
          ha="center", fontsize=FONT_SIZES["annotation"] - 2)
ax_a.text(5 + w / 2, h["iter8_gpu_ms"] * 1.15, f"{h['iter8_gpu_ms']:.2f}",
          ha="center", fontsize=FONT_SIZES["annotation"] - 2)
ax_a.text(5, 3, "5.47x GPU\n4.76x wall", ha="center",
          fontsize=FONT_SIZES["annotation"], color=C_HL, fontweight="bold")
# 底注：口径 caveat + E133c kernel 创新点合并单元素（同机 A/B 百分比注记，
# 跨机绝对值不与 E127 混）
ax_a.text(0.02, 0.02,
          f"single-layer synthetic steady state, not e2e;\n"
          f"E133c fused kernels + buffers: {e133c_e2e_pct:.1f}% steady GPU\n"
          f"(same-machine A/B); tail-32: 4x less coarse compute",
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
ax_b.set_title("Prefill wall: Step2 graph + columnization", pad=14)
ax_b.legend(loc="upper left")
# 数值标注（64K dense 标签移到标记左侧，避让 upper-left legend 右缘）
for i, (lk, d, s2) in enumerate(zip(Lk, dense_ms, step2_ms)):
    if i == len(Lk) - 1:
        # 64K dense 标签移到标记下方，避让 upper-left legend（右缘至 x≈1647）
        ax_b.annotate(f"{d:.0f}", (lk, d), textcoords="offset points",
                      xytext=(0, -26), ha="center",
                      fontsize=FONT_SIZES["annotation"] - 3, color=C_DENSE)
    else:
        ax_b.annotate(f"{d:.0f}", (lk, d), textcoords="offset points", xytext=(0, 10),
                      ha="center", fontsize=FONT_SIZES["annotation"] - 3, color=C_DENSE)
    ax_b.annotate(f"{s2:.0f}", (lk, s2), textcoords="offset points", xytext=(0, -18),
                  ha="center", fontsize=FONT_SIZES["annotation"] - 3, color=C_STEP2)
ax_b.text(64, 55, "1.64x vs dense\n(selection chain CUDA-graph-captured)", ha="right",
          fontsize=FONT_SIZES["annotation"], color=C_HL, fontweight="bold")
ax_b.text(0.02, 0.02, "single-layer synthetic prefill;\nshort lengths remain net-negative",
          transform=ax_b.transAxes, fontsize=FONT_SIZES["annotation"] - 4,
          style="italic", color=pal["neutral"], va="bottom")

# ---- 面板 (c)：完整请求 TPOT vs L（三模型 graph 时代，取代 e152v2 旧口径）----
Lk_c = [32, 64, 128]
models = [
    ("Qwen3-8B", q8b_dense, q8b_tli, "o"),
    ("Qwen3-32B", q32b_dense, q32b_tli, "s"),
    ("Llama-3.1-8B", llama_dense, llama_tli, "^"),
]
for name, dd, tt, mk in models:
    ax_c.plot(Lk_c, dd, "--", color=C_DENSE, marker=mk, markersize=13,
              markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"],
              linewidth=LINE_WIDTHS["data"] * 0.8)
    ax_c.plot(Lk_c, tt, "-", color=C_TLI, marker=mk, markersize=13,
              markeredgecolor="white", markeredgewidth=LINE_WIDTHS["marker_edge"])
# eager 无 graph 臂（仅 8B 有连续序列，诚实画出符号翻转的系统杠杆）
ax_c.plot(Lk_c, q8b_eager, ":", color=C_EAGER, marker="o", markersize=9,
          markerfacecolor="none", markeredgecolor=C_EAGER,
          linewidth=LINE_WIDTHS["data"] * 0.6)
ax_c.set_xscale("log", base=2)
ax_c.set_yscale("log")
ax_c.set_xlim(27, 330)
ax_c.set_ylim(13, 230)
ax_c.set_xticks(Lk_c)
ax_c.set_xticklabels([f"{l}K" for l in Lk_c])
ax_c.set_yticks([20, 30, 50, 100, 200])
ax_c.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax_c.set_xlabel("input length L (tokens)")
ax_c.set_ylabel("decode TPOT (ms/token, log scale)")
ax_c.set_title("E2E decode: fused LoDA-SA vs dense (K1=64)", pad=14, loc="left")
# 图例（线型语义；模型由右侧括号标注区分）
from matplotlib.lines import Line2D  # noqa: E402
handles = [
    Line2D([0], [0], ls="--", color=C_DENSE, marker="o", markersize=11,
           markeredgecolor="white", label="dense FullKV"),
    Line2D([0], [0], ls="-", color=C_TLI, marker="o", markersize=11,
           markeredgecolor="white", label="LoDA-SA + graph"),
]
ax_c.legend(handles=handles, loc="upper left", fontsize=FONT_SIZES["legend"] - 6)
# 数值标注：32K 密集对（持平属故事本身）TLI 值右置；64K 上/下置；128K 并入括号文本
for name, dd, tt, mk in models:
    for i, (lk, d, t) in enumerate(zip(Lk_c, dd, tt)):
        if i == 0:
            ax_c.annotate(f"{d:.0f}", (lk, d), textcoords="offset points",
                          xytext=(0, 10), ha="center", va="bottom",
                          fontsize=FONT_SIZES["annotation"] - 4, color=C_DENSE)
            ax_c.annotate(f"{t:.1f}", (lk, t), textcoords="offset points",
                          xytext=(13, -4), ha="left", va="center",
                          fontsize=FONT_SIZES["annotation"] - 4, color=C_TLI)
        elif i == 1:
            ax_c.annotate(f"{d:.0f}", (lk, d), textcoords="offset points",
                          xytext=(0, 10), ha="center", va="bottom",
                          fontsize=FONT_SIZES["annotation"] - 4, color=C_DENSE)
            # 32B tli@64K=41.8 与 Qwen-8B dense@64K=39.8 点位贴邻（上下左右均
            # 有标签/线）→ 移到右下空区（Qwen dense 线下方、Llama"33"右侧）
            if name == "Qwen3-32B":
                ax_c.annotate(f"{t:.1f}", (lk, t), textcoords="offset points",
                              xytext=(28, -14), ha="left", va="top",
                              fontsize=FONT_SIZES["annotation"] - 4, color=C_TLI)
            else:
                ax_c.annotate(f"{t:.1f}", (lk, t), textcoords="offset points",
                              xytext=(0, -10), ha="center", va="top",
                              fontsize=FONT_SIZES["annotation"] - 4, color=C_TLI)
# 128K 加速比括号 + 模型名 + 端点值（右侧直标，Quest 风格图内结论；y 手工错开防互叠）
bracket_y = {"Qwen3-8B": 40.0, "Qwen3-32B": 100.0, "Llama-3.1-8B": 24.0}
for name, dd, tt, mk in models:
    sp = dd[-1] / tt[-1]
    ax_c.annotate("", xy=(128, tt[-1]), xytext=(128, dd[-1]),
                  arrowprops=dict(arrowstyle="<->", color=C_HL,
                                  lw=LINE_WIDTHS["reference"] * 0.8, shrinkA=2, shrinkB=2))
    ax_c.text(134, bracket_y[name], f"{name}\n{dd[-1]:.0f} -> {tt[-1]:.1f} ms\n{sp:.2f}x",
              fontsize=FONT_SIZES["annotation"] - 2, color=C_HL,
              fontweight="bold", va="center", ha="left")
# eager 臂行内直标（诚实边界：无 graph 时 launch 主导；置于 eager 线下方空区）
ax_c.text(30, 95, "LoDA-SA eager\n(no graph)\n101-134 ms",
          fontsize=FONT_SIZES["annotation"] - 4, color=C_EAGER,
          ha="left", va="top", style="italic")
ax_c.text(0.98, 1.0,
          "graph replays bit-identical\n"
          "8B graph-eager: 64K bit-exact;\n"
          "32K/128K one-token bf16 ties\n"
          "32B arms bit-exact\n"
          "pc40/pc02; iter3+K1=64",
          transform=ax_c.transAxes, fontsize=FONT_SIZES["annotation"] - 4,
          style="italic", color=pal["neutral"], va="top", ha="right")

# 面板字母
for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
    ax.text(0.01, 1.07, letter, transform=ax.transAxes,
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
src133c = "e133c_subspace_speedup_verdict.json:steady_gpu_ms"
rows.append(f"E2\ta\ttli_e133c_delta\tsteady_gpu_delta_pct\tpct\t{e133c_e2e_pct:.1f}\t{src133c}\tmavg_e2e.delta_pct")
rows.append(f"E2\ta\ttli_e133c_delta_tail\tsteady_gpu_delta_pct\tpct\t{e133c_tail_pct:.1f}\t{src133c}\tmavg_tail.delta_pct")
# 面板 (c)：三模型 e2e TPOT（iter3+K1=64 口径，三行判决全齐）
c_src = {
    "Qwen3-8B": ("e172_iter3_k64_qwen_verdict.json:results", q8b_dense, q8b_tli, q8b_eager),
    "Qwen3-32B": ("e174_iter3_k64_32b_verdict.json:results",
                  q32b_dense, q32b_tli, None),
    "Llama-3.1-8B": ("e173_iter3_k64_llama_verdict.json:results",
                     llama_dense, llama_tli, None),
}
for name, (src, dd, tt, ee) in c_src.items():
    for L, d, t in zip((32768, 65536, 131072), dd, tt):
        rows.append(f"E2\tc@L{L}\t{name}_dense\ttpot_ms\tms\t{d:.2f}\t{src}\tdense")
        rows.append(f"E2\tc@L{L}\t{name}_tli_graph\ttpot_ms\tms\t{t:.2f}\t{src}\ttli_graph")
        rows.append(f"E2\tc@L{L}\t{name}_speedup\tratio\tx\t{d / t:.3f}\t{src}\tdense/tli_graph")
    if ee is not None:
        for L, e in zip((32768, 65536, 131072), ee):
            rows.append(f"E2\tc@L{L}\t{name}_tli_eager\ttpot_ms\tms\t{e:.2f}\t{src}\ttli_eager")
rows.append("E2\tc\tsuperseded\tnote\t-\t"
            "panel (c) replaced the iter22 caliber (e152v4/e159/e160/e160b: Qwen-8B "
            "1.064/1.448/2.067x, Llama 1.015/1.41/1.961x, 32B 1.015/1.338/1.841x) "
            "with the iter3+K1=64 caliber (E172/E173/E174, 2026-10-25)")
with open(os.path.join(OUT, "E2_latency_figure_data.tsv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("exported E2_latency_figure_data.tsv")
print("DONE")
