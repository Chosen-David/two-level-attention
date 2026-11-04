# O_observation 图元数据（随图同步更新）

> 规则：每次修改 `fig_o_observation.py` 或任一数据源后，必须同步更新本文件并
> 运行 `python3 agent_doc/guide/paper_figs/figs/fig_o_observation.py` 重渲染。

## 渲染

```sh
python3 agent_doc/guide/paper_figs/figs/fig_o_observation.py
# 输出：O_observation.{pdf,png,svg} + O_observation_figure_data.tsv（同目录）
# 中文：O_LANG=zh python3 ...（输出 O_observation_zh.*）
```

## 口径（本地 NFA 几何，2026-11-04 起）

生产几何（`sparse_attn/indexer/nfa_layer_batch_selector.py`，
`sparse_attn/arguments.py` 默认参数）：

- `sink`  = `nfa_sink_tokens` = 128（正交强制区）
- `swa`   = `nfa_swa_size`    = 128（尾窗）
- `near`  = `nfa_near_len`    = 4096（页对齐 64）
- `page`  = `tia_block_size`  = 64
- 区域（绝对 token 位置，causal 长度 t）：
  - sink = `[0, sink_hi)`，`sink_hi = min(128, t)`
  - swa  = `[t-128, t)`
  - `far_hi = floor((t-128-4096)/64)*64`（夹在 `[sink_hi, t-128]`）
  - far  = `[sink_hi, far_hi)`
  - near = `[far_hi, t-128)`
- 打分：far 池 L1 粗筛 = **minmax**（块 min/max 上界）；near 池 L1 粗筛 = **avg**（块均值）。

此口径取代旧「sink=128 / near_len=65 / near_lo」短序列诊断口径；图 (a) 数据源
由 `e140_o_trace_v2.json` + `e140b_pooling_anatomy.json`（短序列）切换为
`e140b_nfa_region_anatomy.json`（长序列 12k–32k，本地口径页级解剖）。

## 面板

1×3 网格，figsize 10.2×3.9in，`width_ratios [1.30, 0.78, 0.76]`。

- **(a) 分区+异构打分**：代表 query `lb_hotpotqa_0:layer03`（t=16956）的
  far/near 选中页数分组柱。oracle（真实质量）far 23 / near 7；avg far 0 / near 30
  （far 稀疏尖峰被均值稀释漏选）；minmax far 21 / near 10（救回 far）。
  全 576 样本（16 序列 × 36 层）：minmax 在 far 选更多页占比 86.1%，
  avg 在 near 选更多页占比 83.2%。far 质量捕获均值 avg 0.179 vs minmax 0.737。
- **(b) 共享子空间**：tail32 0.84 / random32 0.51 / high32 0.24
  （`selected_mass_over_reference_topk_mass`），数据源 `e140_o_trace_v2.json` 的 o3。
- **(c) 近远区质量**：near/far 中位注意力质量柱（log 轴）。far 中位 0.0018、
  near 中位 0.0005（16 样本 × 36 层，本地口径）。far 区宽度远大于 near（几万 vs 4096
  token），因此 far 总 mass 大但单位 token 密度更低。

## 数据源

- `exp/trace/results/e140b_nfa_region_anatomy.json`：本地口径页级 avg/minmax 选择解剖
  （诊断简化：全 128 维子空间，GQA 组均，块 64，预算 2048 token）。
- `exp/trace/results/e140_o_trace_v2.json`：共享子空间三臂（图 b）。
- `exp/trace/results/e6b_near_far_profiles.json`：本地口径 near/far mass（图 c）。

## 诚实约束

1. 图 (a) 是诊断性模拟（页级 mass/avg/minmax 打分，非生产逐 KV 头实现）；
   集合差异是候选访问差异，不判定打分器质量优劣。
2. 质量捕获 ≠ 任务精度；e2e 证据属于评估章节。
3. 图 (c) 是 prefill 统计；门控有效性不得由本图断言。
4. far/near mass 为区域总质量（受区域宽度影响），跨区域对比需结合宽度解读。
