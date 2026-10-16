# O_observation 图元数据（随图同步更新）

> 规则：每次修改 `fig_o_observation.py`、任一数据源或面板叙事后，必须同步更新本文件、
> `O_observation_figure_data.tsv`（由渲染脚本自动重写）、
> `../observation/e140_validity.json`（reviewed 哈希更新），并运行
> `python -m exp.paper.e140_asset_provenance` 校验。

## 渲染

```sh
python3 agent_doc/guide/paper_figs/figs/fig_o_observation.py
# 输出：O_observation.{pdf,png,svg} + O_observation_figure_data.tsv（同目录）
```

版式：2×2，figsize 9.5×6.9in，对 IEEE 双栏 \textwidth≈7.16in 缩放 ≈0.75，
有效字号 ≈ title 11.3 / label 9.8 / tick 8.6pt。
导出前经 `figure_lint`（0 overlap；2 条 clipped 为 tight-bbox 前的 colorbar tick，导出件内无裁切）。

**2026-10-16 版式约束（用户指示）**：图内不堆说明文字（note box 已全部移除），
面板数字与结论并入手稿 caption。图内仅保留：坐标轴/刻度、面板 (a) 的带名 ytick、
far/near 区域标签、紧凑选择图例（灰=未选 / 蓝=avg 选中 / 红=oracle 选中但 avg 漏，
✓=minmax 救回）与尖峰页 ▼ 标记（page@7936）；面板 (b) 仅图例 + worst 0.00→0.26 标注。

## 数据源（全部为 pin 住的真实采集）

| 面板 | 数据源 | 说明 |
|---|---|---|
| (a)(b)(c) | `exp/trace/results/e140_o_trace_v2.json` | E140 v2 重采（修复采集器：实际 q_norm/k_norm 输出 + 实际 RoPE）；4 源序列 × L8/L24 × 16 query；64-token bin 质量（GQA 组均概率） |
| (a) 叠加 | `exp/trace/results/e140b_pooling_anatomy.json` | E140b pooling 解剖 probe（2026-10-16，本机 H20，transformers 4.56.2 / torch 2.8.0）；同口径 import 复用 v2 采集器数学；页边界与 o1 bin 逐一对齐断言 |
| (d) | `exp/trace/results/e6b_far_profiles.json` + `e6_layer_skip.json` | E6 逐层 far mass（16 序列 × 36 层）与可跳过层统计 |

O4 v2（`e140_o4_select_trace_v2.json`）有部分末页缺陷，任何面板均不读取
（见 `../observation/E140_VALIDITY.md`）。

## 代表 query（面板 a 条带）

固定为 **musique sample1, layer 24, q_pos=14212**（非预登记规则产出，披露如下）：

- far 份额 0.498 ≈ 128 条 query 中位数 0.496（位置分段有代表性，非极端样本）；
- 在同等代表性的候选中，其 avg 稀释机理最清晰（尖峰页 page@7936：单 token 占页质量 100%）；
- 身份与 dac_manuscript 正文/caption 一致；手稿正文须保持同一口径。

区域口径：sink/swa 各 128 保送；near := 未保护 mid 中距 query 最近的 25%（α=.25，
floor 页对齐）；q=14212 时 near_len=3525，far 占 mid token 的 75.7%。

## 面板定义与关键数字（当前渲染）

### (a) Avg pool misses sparse far spikes — 分区 + 异构打分器动机

三条带（x = key 绝对位置，query 在最右端）：

1. **attn mass**：逐 64-token bin 的真实注意力质量（log 色标，vmin 2e-4 / vmax 2e-2）；
2. **oracle pool**：按真实 bin 质量排序的单池 top-2048 token（任何打分单池的上界）；
3. **avg pool**：共享 32 维 tail 子空间（[48,64)∪[112,128)）token 分的页均值单池
   top-2048；**红 = oracle 选中但 avg 漏掉的页，✓ = 其中被 Quest minmax 上界救回**。

关键数字（repr query）：avg 漏 13/33 oracle 页（漏掉质量占 oracle 质量 59%）；
minmax 救回 6 页（其中 far 3/4）；尖峰页 page@7936 单 token 占页质量 100% →
avg 页均分被 63 个邻居稀释而落选，minmax 上界 ≥ 该 token 分而保留。
128 query 汇总：avg 平均漏 7.88/~30 oracle 页（漏质量均值 24.9%）。
→ 结论：near 均匀用 block-avg；far 稀疏尖峰用 minmax 无漏上界 + 精筛。

诊断口径披露：token 分为 32 Q-head 均值（生产按 KV 组内 4 head 逐头选页）；
漏筛同时含 32 维表示误差与 pooling 稀释，二者未在图内分离。

### (b) Quota floors far capture — far 保底配额（β）动机

128 条 query 按单池 far 捕获率升序排序；配额臂（far 区内部 oracle top-1024，
总预算 2048 的 50%）按同一 query 顺序叠加，竖直间隙 = 逐 query 保底收益。

关键数字：单池 far 捕获均值 0.69（near 同池 0.93）；弥散 far（share≤0.4）0.41；
最差 0.00 → 配额后 0.26；p10 0.20→0.34；弥散 0.41→0.52；均值仅 0.69→0.72。
→ 结论：配额是**防挤出下限保底，不是均值增益**；e2e 增益归主表 β 让渡消融
（mass ≠ e2e，不在本图下任务质量结论）。

### (c) Which 32 of 128 dims — 共享子空间动机（未变）

等宽 32/128 维预算：tail32（低频完整对）0.84 / random32（非成对）0.51 /
high32（高频）0.24。主指标 = 选中质量 / 参考 top-K 质量（非全部注意力质量）。

### (d) Far mass by layer — 逐层 far 门控（D′）动机

16 序列 × 36 层 prefill far mass 热图（log 色标）：24/36 层逐样本中位 <0.01；
每序列 12–32 层 far 极轻。prefill 统计，不约束 decode（正文第五节有反例：
prefill <.05 → decode .1678）。

## 诚实约束（任何改写不得违反）

1. 4 个源序列；128 条 query 是重复观测，非独立样本。
2. (a) oracle/avg/minmax 条带是诊断性模拟（oracle = 质量排序上界；avg/minmax =
   32 维共享子空间诊断打分器），非生产 near-avg/far-minmax 逐 KV 头实现。
3. 质量捕获 ≠ 任务精度；e2e 证据属于评估章节（β 让渡消融）。
4. (d) 是 prefill 统计；门控有效性不得由本图断言。
5. (b) 配额臂是 1024/2048 示意性划分，非调参结果。

## 下游同步清单

- `agent_doc/advice/dac_manuscript/figures/o_observation.pdf`（从本目录拷贝）
- `dac_manuscript/source/paper_{en,zh}.tex` 的 `fig:observation` caption 与引言段
- `dac_manuscript/MANIFEST.json` 哈希
- `dac_manuscript/OBSERVATION_UPDATE_2026-10-16.md`（远端 GPT 交接）
