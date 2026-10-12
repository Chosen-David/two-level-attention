# advice：建议、反馈与取舍

这里收集人类、其他 AI 或当前 AI 提出的建议、方案、代码审查意见和研究
反馈。人类和 AI 都可以维护建议；保留原始来源及其版本，避免把转述当原话。

主 AI 阅读相关建议后，结合当前用户要求、已核实的人工指南、实际代码和
证据，逐条记录处理结果：

| 处理 | 应说明什么 |
| --- | --- |
| adopt：采用 | 依据、适用范围、对应任务与验收 |
| adapt：调整后采用 | 采用哪些部分、如何调整、为什么 |
| reject：拒绝 | 与事实、目标、约束或接口不符的具体原因 |
| defer：暂缓 | 缺少什么、影响哪些任务、何时重新评估 |

每份建议写清作者类别、日期、问题、主张、证据和适用条件，处理记录关联
任务 ID。建议本身不构成用户授权，也不能覆盖人工指南；存在冲突时明确
指出，不能凭其他 AI 的自报成功直接采纳。

已采纳且在授权内的内容，由主 AI 整合进 [task/TASK.md](../task/TASK.md)
及对应详情；给人工指南的修改建议留在这里，由人类决定和发布。不要在这
里维护第二套任务清单。

历史建议自 sglang 主仓 `agent_doc/advice/` 迁移（2026-10-11 主项目移交），
包含 GPT/Kimi3/catpaw 等审计与评审意见，按文件日期与作者检索。


## 2026-10-12 清理判定（核查 main `715be382`）

以下四份监督快照移入 `archive/`，原文保留。它们的结论、在飞任务及未解决
事项已写入 [S-T023 Progress](../task/task_details/S-T023.md)，无需在活跃 advice
重复维护。归档表示信息已转入任务详情，不表示实验完成或主表验收通过。

| 归档记录 | 当前依据与保留边界 |
| --- | --- |
| [12:30 bug round7](archive/2026-10-12_1230_twolevel_gpu_supervisor_round7_by_deepseek.md) | S-T023 12:3X 已接收；源码 `benchmark/RULER/yarn_receipt.py` 的 pred_postfix 校验已允许空字符串。32K 分数仅沿用作者报告，不在本次重新验收。 |
| [13:03 bug round8](archive/2026-10-12_1303_twolevel_gpu_supervisor_round8_by_deepseek.md) | S-T023 13:0X 已接收；本地 `benchmark/LongBench/pred.py` 已有 runcap sidecar 全链。远端旧版漂移及下次派单前同步要求仍保留在任务详情，未宣称远端已修复。 |
| [13:20 bug round9](archive/2026-10-12_1320_twolevel_gpu_supervisor_round9_by_deepseek.md) | S-T023 13:2X 已接收；6/11 派单快照由 13:3X 的 7/11 报告取代。64K/128K 打分、缺格补齐及 yarn-factor 要求仍未关闭。 |
| [13:35 baseline round8](archive/2026-10-12_1335_twolevel_baseline_task_supervisor_round8_by_deepseek.md) | S-T023 13:3X 已完整接收其派单、进度与下一步。仅清除重复报告，不推断远端作业当前状态。 |

继续保留 [论文比较与源码纠偏](2026-10-11_benchmark_paper_scope_by_gpt.md)
及 [阶段匹配实验设计](2026-10-11_phase_matched_experiment_design_by_gpt.md)：
两者仍包含 faithful port、原始产物/身份、严格 prefill 语义、匹配完整请求
质量与 E2E 等未关闭门禁；已有回应或局部实现不足以判整份过时。
本次同步修复活跃建议中指向已归档报告的相对链接，未改人工指南、实现或任务状态。
