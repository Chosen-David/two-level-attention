# E1 主图两轴 baseline 结构与 FASA 臂接入（deepseek 轨 2026-10-16）

> 用户确认（2026-10-16）的两轴结构落地记录。E1_quality 主文 P0 图 panel (b)
> 现在有 4 条臂：dense FullKV（上界）+ Quest（结构轴）+ FASA（表示轴）+ TLI（主方法）。

## 1. 两轴结构（方法轴各自最强同类）

| 轴 | 代表 baseline | 对仗点 | 理由 |
|---|---|---|---|
| 结构近邻（检索/候选选择） | Quest | L1 页粗筛 + minmax 上界 | 同为两级选择器、页预算、minmax 评分 |
| 表示近邻（紧凑评分） | FASA | 校准选 dominant RoPE components 做低维评分 | 同为「RoPE 坐标子集代替全维评分」 |

主图只放这两轴各一个最强同类，SparQ/MInference/ClusterKV/MoBA 走附录/全景图
（`ref/figs/baseline_comp/fig1_accuracy_matrix.py` 升级 8 方法）。

## 2. E1 panel (b) 现臂与数值（程序化读取）

| 臂 | LBv1 16 | LBv2 503 | RULER 128K | 来源 |
|---|---|---|---|---|
| dense FullKV | 48.11 | 32.21 | 46.43 | e116b/e146 |
| Quest vendor | 41.35 | 30.22 | 19.33 | main_table draft + e119 |
| **FASA vendor** | **33.31** | **11.93** | **—（未测）** | fasa_lb_16.json + pred_fasa_1024_lbv2 |
| TLI (mavg) | 48.90 | 32.21 | 23.98 | e116b/e146 |

- FASA RULER 128K 未测（方法局限：仅 RULER 32K 6/11 任务、s1=0），图与 tsv 留空，
  不做外推；tsv 有 fasa_ruler128k_missing note。
- 改动文件：`agent_doc/guide/paper_figs/figs/fig_e1_quality.py`（数据段 + panel b 4 组 bar
  + tsv fasa 行 + 脚注压缩至 5 行），lint overlaps=0 clipped=0。

## 3. 表示轴论证的 E133 呼应材料（论文主文可直接引用）

FASA 用校准挑的 RoPE 坐标都保不住质量（LBv1 33.31，检索类全线掉 10-20 分），审稿人
可能反问「同为 RoPE 子空间评分，你凭什么说 TLI 的 tail32 共享表示行？」。现成辩护：

- **E133 子空间消融**（Qwen3-8B LBv1 5 任务 AVG5）：
  - ① full+sub（冠军默认）：47.91
  - ② sub+sub（L1/L2 均 tail32）：47.588，Δ-0.322，95% CI[-0.510, -0.106]（显著小幅下降）
  - ③ sub+full（L1 tail32 / L2 全维）：47.666，Δ-0.244，CI[-0.636, +0.148]（未显著）
  - 判决：L1 粗筛子空间引入可检测精度损失（②③ vs ① 均负，② 显著），L2 全维/子空间
    影响甚微 → **论文方法维持 full+sub（mavg 冠军默认）**。
  - 落盘：`exp/trace/results/e133_subspace_verdict.json`（E1 图 panel b 脚注已含 tail32
    AVG5 47.59, Δ-0.32 CI[-0.51,-0.10]）。
- 呼应话术建议：「紧凑表示保质量的代价是 -0.32（CI 不含 0 但仅 0.3 分）；FASA 的校准
  坐标同样压缩表示却掉 -16.08，说明**坐标选择方式**而非**压缩本身**决定质量」。

## 4. 待办（数据回来补）

- ClusterKV（本地 GPU0/1 在跑）：RULER 32K 100 样本 + LongBench 16 任务 → E1 图加臂。
- MoBA（.187/.251 重跑中）：RULER 32/64/128K + LongBench → E1 图加臂。
- SparQ 64K/128K（.187 GPU6/7）→ 全景图补齐。

## Opus 核验与 sink4 数值替换通报（2026-10-17 09:1X）

核验：两轴结构（Quest 结构轴 + FASA 表示轴）与 E133 子空间判决引用（47.91/47.588/47.666，CI 值与 e133_subspace_verdict.json 一致）确认无误；FASA LBv1 33.31 / LBv2 11.93 接受为 vendor 实测口径。「坐标选择方式而非压缩本身决定质量」话术与 E133 数据匹配。

**关键替换（sink4 主口径转向，2026-10-16 18:0X 用户指令）**：上表 TLI 两列已被 sink4 新数据取代，E1 图换数点：

| 臂 | 旧值（sink128/β0.375 混杂） | sink4 新值 | 来源 |
|---|---|---|---|
| TLI LBv1 EN 13 | 48.90（e116b 16任务 sink128） | **51.25**（vs FullKV 50.36 → +0.89） | e145c_en_mavg_verdict.json（本轮 09:01 打分） |
| TLI LBv1 ZH 8 | —（E145 β0.375 崩溃） | **46.004 vs dense 46.003（Δ+0.001）** | e145c_zh_verdict.json |
| TLI RULER 128K | 23.98（E146 β0.375 far=0 崩溃臂） | **46.02（vs FullKV 46.43，-0.41）** | e146c_r128k_formal.json（三档系列判决 e146c_series_verdict.json） |
| TLI RULER 32K/64K | 59.65/49.94（e116b sink128） | **58.30/48.05** | e146c_r32k/r64k_formal.json |

注意：fig_e1_quality.py 当前 WIP 的 L61-63 仍读 e146_ruler128k_verdict.json 旧档（TLI=23.98）；LB 面板读 e116b（sink128 48.90）。替换须等本图提交边界后进行，且 LBv1 任务集构成（e116b 16 任务 vs E145c 13 EN+8 ZH）需先统一口径再换格——两套任务集不可直接横比。FASA/Quest/FullKV 臂不受 sink4 转向影响（vendor/dense 无 sink 参数），数值保持。
