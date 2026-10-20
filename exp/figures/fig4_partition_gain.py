# Fig 4 (2026-10-19 部署口径重写): 分区收益 + 长上下文 baseline 对比双 panel
# (a) RULER 长度梯度 32/64/128K: FullKV / PSI β.0625 主臂 / PSI 同预算单池
#     —— 支撑「分区增益随长度单调增长 +0.93/+3.58/+6.47」(32K 同机配对)
# (b) RULER 三档 baseline 对比: PSI / FullKV / Quest / AsyncTLS 分档 grouped bars
#     —— 支撑「Quest/AsyncTLS 随长度崩塌、PSI 全档贴住 FullKV」
# 数据快照（落袋 formal JSON，勿手改数字）：
#   exp/trace/results/e146c_series_verdict.json           FullKV + β.125 三档
#   exp/trace/results/e147c_r32k_formal.json              β.0625 32K 59.29
#   exp/trace/results/e147h_r64k_formal.json              β.0625 64K 48.85
#   exp/trace/results/e147h_r128k_formal.json             β.0625 128K 45.73
#   exp/trace/results/e146ca_r{32,64,128}k_formal.json    同预算单池三档
#   exp/trace/results/e119_ruler{32,64,128}k_formal_quest_vendor.json
#   exp/trace/results/asynctls_r32k_formal.json
#   exp/trace/results/e119_ruler65536_formal_asynctls_vendor_64k.json
# house style: okabe_ito、白底、全英文、矢量 PDF+PNG
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TR = "/home/wangyuanshuo02/two-level-attention/exp/trace/results"
OUTS = ["/home/wangyuanshuo02/sglang/paper/figures",
        "/home/wangyuanshuo02/two-level-attention/exp/figures"]

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F", "sky": "#56B4E9"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------- 数据（全部读 formal JSON） ----------
def cell_avg(path):
    d = json.load(open(path))
    for tasks in d["scores"].values():
        return float(np.mean(list(tasks.values())))

fullkv = []
b125 = []
d = json.load(open(f"{TR}/e146c_series_verdict.json"))["three_lengths"]
for L in ["L32768", "L65536", "L131072"]:
    fullkv.append(d[L]["fullkv"])
    b125.append(d[L]["sink4"])

b0625 = [cell_avg(f"{TR}/e147c_r32k_formal.json"),
         cell_avg(f"{TR}/e147h_r64k_formal.json"),
         cell_avg(f"{TR}/e147h_r128k_formal.json")]
single = [cell_avg(f"{TR}/e146ca_r32k_formal.json"),
          cell_avg(f"{TR}/e146ca_r64k_formal.json"),
          cell_avg(f"{TR}/e146ca_r128k_formal.json")]
quest = [cell_avg(f"{TR}/e119_ruler32k_formal_quest_vendor.json"),
         cell_avg(f"{TR}/e119_ruler64k_formal_quest_vendor.json"),
         cell_avg(f"{TR}/e119_ruler128k_formal_quest_vendor.json")]
async_tls = [cell_avg(f"{TR}/e119_ruler32768_formal_asynctls_vendor_fix.json"),
             cell_avg(f"{TR}/e119_ruler65536_formal_asynctls_vendor_fix.json"),
             cell_avg(f"{TR}/e119_ruler131072_formal_asynctls_vendor_fix.json")]  # 排序修复版 32/64/128K

LENS = ["32K", "64K", "128K"]
delta = [b - a for a, b in zip(single, b0625)]

# ---------- 画布 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})

# (a) 分区收益长度梯度
ax = axes[0]
x = np.arange(3)
ax.plot(x, fullkv, "s-", color=OI["gray"], lw=1.6, ms=5, label="FullKV")
ax.plot(x, single, "^--", color=OI["orange"], lw=1.6, ms=5.5, label="PSI single-pool (same budget)")
ax.plot(x, b0625, "o-", color=OI["blue"], lw=1.9, ms=5.5, label="PSI partitioned ($\\beta{=}.0625$)")
for i, (a, b) in enumerate(zip(single, b0625)):
    ax.annotate(f"{b - a:+.2f}", (i, (a + b) / 2), textcoords="offset points",
                xytext=(14, 0), fontsize=7.6, ha="left", color=OI["blue"])
ax.set_xticks(x)
ax.set_xticklabels(LENS, fontsize=8.5)
ax.set_xlabel("RULER context length", fontsize=8.5)
ax.set_ylabel("11-task average score", fontsize=8.5)
ax.set_ylim(35, 62)
ax.set_title("(a) partition gain grows with length\n+0.93 / +3.58 / +6.47 at 32/64/128K",
             fontsize=8.6)
ax.legend(fontsize=7.2, frameon=False, loc="lower left")
ax.tick_params(labelsize=7.5)

# (b) 长上下文 baseline 对比（分档 grouped bars）
ax = axes[1]
systems = [("FullKV", fullkv, OI["gray"]),
           ("PSI", b0625, OI["blue"]),
           ("Quest@1024", quest, OI["orange"]),
           ("AsyncTLS", async_tls, OI["purple"])]
n = len(systems)
w = 0.19
for j, (name, vals, color) in enumerate(systems):
    xs, ys = [], []
    for i, v in enumerate(vals):
        if v is not None:
            xs.append(i + (j - (n - 1) / 2) * w)
            ys.append(v)
    ax.bar(xs, ys, w * 0.92, color=color, label=name)
    for xx, vv in zip(xs, ys):
        ax.annotate(f"{vv:.1f}", (xx, vv), textcoords="offset points",
                    xytext=(0, 3), fontsize=6.2, ha="center", color="#333333")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(LENS, fontsize=8.5)
ax.set_xlabel("RULER context length", fontsize=8.5)
ax.set_ylabel("11-task average score", fontsize=8.5)
ax.set_ylim(0, 66)
ax.set_title("(b) Quest collapses with length, PSI and AsyncTLS keep quality\nJSSA 58.7 at 32K (budget$=$1024) sits between PSI and single-pool",
             fontsize=8.6)
ax.legend(fontsize=7.2, frameon=False, loc="upper right", ncol=2)
ax.tick_params(labelsize=7.5)

fig.tight_layout(w_pad=2.4)
for out in OUTS:
    fig.savefig(f"{out}/fig4_partition_gain.pdf")
    fig.savefig(f"{out}/fig4_partition_gain.png")
print("saved fig4_partition_gain (deployment caliber)")
print("lens:", {L: dict(fullkv=round(f, 2), b0625=round(m, 2), b125=round(s, 2),
                       single=round(c, 2), quest=round(q, 2),
                       asynctls=None if t is None else round(t, 2))
               for L, f, m, s, c, q, t in zip(LENS, fullkv, b0625, b125, single, quest, async_tls)})
print("partition deltas:", [round(v, 2) for v in delta])
