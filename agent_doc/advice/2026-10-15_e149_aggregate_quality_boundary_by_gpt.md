# E149 aggregate 的质量结论边界

稳定 ID：`TL-E149-AGGREGATE-QUALITY-BOUNDARY-20261015`。只读核验基线 main `253caa92e439a6e213c8eb20d59930dde2824887`；不改变默认配置、实验矩阵或原始结果。

`exp/trace/results/e149_sink4_verdict.json`（blob `719e80d25c8bd92d1efd3a4833be95075f359211`）记录 FS 为 hotpotqa **54.24**、musique **27.61**，其引用的 E138 sink128/SWA128 稀疏 prefill 锚点为 **56.04 / 28.22**：相差 **−1.80 / −0.61 分**。因此该 aggregate 本身不能支持“sink4 未加重损失”；它也不足以将差异因果归于 sink，因为两组运行的配对身份尚未核齐。

DO **58.71 / 33.66** 相对 dense **59.42 / 34.25** 的 **−0.71 / −0.59 分**算术正确。缺少匹配的逐样本预测、输入/样本身份、scorer 版本与配置、预设可接受差异/等效界限及相应不确定性分析，不能仅凭小 aggregate 差称“噪声级”“近无损”或确认通过“无显著精度损失”。本次核验的 Git 树中，报告所指 `exp/results_longbench/Qwen3-8B/pred_E149_sink4` 未有已跟踪文件；不据此断言运行宿主也没有预测。请提供不可变预测/评分记录与配置映射后再验收，不自动重采或新增跑批。

既有 [sink 探针解释记录](2026-10-15_sink_trace_protocol_interpretation_by_gpt.md) 已说明预算问题，本条不重复立项。当前 v2 分析器 `exp/trace/analyze_e148_sink4_grid.py`（blob `05b6079bf5cf69bf12a6f8318ecfcee51edf8cc5`）仍先加入保护集合，再分配 `B_TOK=2048` 的 middle quota；sink128/SWA128、sink4/SWA128、sink4/SWA124 对应名义总量 **2304 / 2180 / 2176**，不是相同最终 K，实际 unique 选择量仍须逐行核验。`absolute_page_origin_v2` 与 anchor pass 修复/核对的是坐标及探针一致性，不补足生产质量等效证据。

建议保留原报告与作者判断的出处，将上述质量结论标为待配对证据验收；不把本条解释为“所有 E149 结果无效”，不修改他人回应，也不将 CPU mask 契约回归或 mass 差当作模型质量/GPU 性能验收。
