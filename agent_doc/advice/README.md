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


## 2026-10-12 第二次清理判定（核查 main `35b39bd8`）

再次核对活跃 advice、S-T023 Progress、当前 Quest adapter/LongBench producer、
主表初稿及 eager prefill/indexer 源码。以下四份监督快照移入 `archive/`，
原文逐字保留；它们已重复或被后续状态取代，待办继续由 S-T023 维护。

| 归档记录 | 判定依据与未关闭边界 |
| --- | --- |
| [14:25 baseline round9](archive/2026-10-12_1425_twolevel_baseline_task_supervisor_round9_by_deepseek.md) | S-T023 14:2X 已接收；64K JSON/raw 已交付，两份活跃建议中的窄核验继续适用。128K .24 派单及旧 ETA 被 round12 删除/重派报告取代；不以“符合论文趋势”作为验收。 |
| [14:55 baseline round10](archive/2026-10-12_1455_twolevel_baseline_task_supervisor_round10_by_deepseek.md) | S-T023 14:5X 已接收；2/11 及 qmsum 58% 是历史进度，round12 已报告 7 格回收及 qmsum 收齐。未独验的产物身份门禁继续保留。 |
| [15:55 GPU round10](archive/2026-10-12_1555_twolevel_gpu_supervisor_round10_by_deepseek.md) | S-T023 15:5X 已完整登记 pylibraft、gcc/pybind/flashinfer、kernel、Qwen3 适配阻塞及 MoBA 暂缓。机器删除使旧资源表失效；不宣称 ClusterKV 已接入或 sklearn 替换等价。 |
| [17:15 baseline round12](archive/2026-10-12_1715_twolevel_baseline_task_supervisor_round12_by_deepseek.md) | 已将删除事件、重派身份、作者收官记录及待打分事项补入 S-T023 17:1X。归档只表示不再重复维护建议；~20:1X 是作者历史 ETA，不代表实际完成。 |

两份 2026-10-11 GPT 建议继续活跃：Quest dense/norm 修复已有源码证据，
但 faithful port、原始身份/评分、整请求 E2E、严格 prefill 前缀语义等门禁
尚未全部闭合；阶段设计还包含 E130/E131 配置及产物绑定待核事项。
未修改人工指南、实现、实验分数或任务完成状态。


## 2026-10-13 清理判定（核查 main `cac0238`）

核对活跃 advice、S-T023、MInference/Quest 适配层、LB/RULER 入口、YaRN
声明边界及 E119 重判 JSON。以下九份监督快照归档，原文 SHA-256 逐份一致；
未完成事项与撤回判定整合到 [S-T023 §10](../task/task_details/S-T023.md)。
归档不等于独立实验验收。本次未运行 GPU 或复测远端依赖/网络。

| 归档记录 | 判定依据及保留边界 |
| --- | --- |
| [0125 报告](archive/2026-10-13_0125_twolevel_gpu_supervisor_round11_by_deepseek.md) | GPU round11：旧全空/收官快照被后续 MInference 接入取代；ClusterKV 更正与阻塞链补入 S-T023 §10，未判依赖已解。 |
| [0208 报告](archive/2026-10-13_0208_twolevel_baseline_task_supervisor_round13_by_deepseek.md) | baseline round13：vendor 通道已有源码，接入及下一步已进入 S-T023；Qwen3 pattern 仍须独验，不沿用旧层/头数猜测。 |
| [0505 报告](archive/2026-10-13_0505_twolevel_baseline_task_supervisor_round14_by_deepseek.md) | baseline round14：Qwen3 胶水、搜索与两任务作者 sanity 已入 S-T023；后续全量记录取代派单建议，faithful port 门禁未关闭。 |
| [0520 报告](archive/2026-10-13_0520_twolevel_baseline_task_supervisor_round15_by_deepseek.md) | baseline round15：makedirs 幂等修复在源码存在；任务名/跨机检查要点保留，全量在飞快照已被后续汇总取代。 |
| [0550 报告](archive/2026-10-13_0550_twolevel_gpu_supervisor_round12_by_deepseek.md) | GPU round12：RULER MInference 分支在源码存在，漂移事件已入 S-T023；32K 待补 vt/旧进度由 round16 作者汇总取代。 |
| [0700 报告](archive/2026-10-13_0700_twolevel_baseline_task_supervisor_round16_by_deepseek.md) | baseline round16：LBv1/LBv2/32K 作者分数及 64K 派单已入 S-T023，64K 在飞快照被 round13/18 取代；非独立主表验收。 |
| [0725 报告](archive/2026-10-13_0725_twolevel_gpu_supervisor_round13_by_deepseek.md) | GPU round13：64K 50.29 与 128K 派单摘要补入 S-T023 §10；08:00 ETA 与 GPU 快照失效，32B 待办保留。 |
| [0740 报告](archive/2026-10-13_0740_twolevel_baseline_task_supervisor_round17_by_deepseek.md) | baseline round17：在飞/ETA 被后续汇总取代，32B 前置、Quest 代表任务和 ClusterKV 阻塞继续留在 S-T023。 |
| [0840 报告](archive/2026-10-13_0840_twolevel_baseline_task_supervisor_round18_by_deepseek.md) | baseline round18：三臂作废及 mavg/aavg 重跑被 Opus 11:57 判决明确撤回；S-T023 §10 标注撤回，保留原文供追溯，不再作为执行指令。 |

继续保留两份 10-11 GPT 建议、E134 子空间判决与 E119 11:57 重判：前两份
尚有 faithful port/身份/整请求 E2E/严格 prefill 等未关闭门禁；E134 待扩大
质量确认与 prefill 省算兑现；E119 是撤回 round18 误判的当前权威作者记录。
不根据论文趋势关闭验收，不修改人工指南、方法默认值、分数或任务完成状态。


## 2026-10-13 第二次清理判定（核查 main `b787a372`）

以下两份 YaRN 重判已完整收录于 [S-T023 Progress 与 §10](../task/task_details/S-T023.md)
及 [E119 判决 JSON](../../exp/trace/results/e119_128k_yarn_factor_reverdict.json)，
移入 `archive/`，逐字保留。归档表示重复建议已整合，不重新验收实验或改变作者判决。

| 归档记录 | 依据与保留边界 |
| --- | --- |
| [Opus 11:57 E119 重判](archive/2026-10-13_1157_E119_128k_yarn_factor_reverdict_by_opus.md) | round18 撤回、三臂锚点恢复及 mavg/aavg 重跑取消已有 S-T023 §10 和判决 JSON；源码 `benchmark/RULER/run_ruler_e109.sh`、`pred_ruler.py` 与 `yarn_receipt.py` 支持 auto factor/声明边界。未重跑逐位对拍，不把当前脚本推断当成全部旧 producer 回执。 |
| [DeepSeek 12:10 重判复核](archive/2026-10-13_1210_yarn_factor_round18_verdict_revisit_by_deepseek.md) | fullkv 重评分、Opus 交叉判决、minf 分任务观察及 open_question_minf 已入 S-T023 12:1X/13:3X。旧安装与派单快照不再作为当前进度；稀疏聚焦是待验证解释，不能据分数方向关闭 faithful port。 |

当前四份活跃建议继续保留：

- 两份 10-11 GPT 建议：整请求质量/实测 E2E、faithful port、原始身份与严格
  prefill 等门禁未全部关闭；Quest Q/K norm 在当前源码已修，任务索引相应纠正，
  不再将该局部缺陷列为待修。
- E134 子空间判决：紧凑实现和更新表已存在，更广任务/128K 质量确认与
  prefill 省算兑现仍待证，不以局部实现完成归档整份建议。
- E137 gate 组合建议：当前 `_resolve_skip` 仍会按非 null 静态集合重写
  `skip_far`，静态默认开、动态默认关；布局修复已有，但 gate 组合及实际运行
  flags/有效 mask 绑定、修复后标定与开销证据未关闭，继续活跃。

归档路径在任务详情同步更新，历史 Progress 原意保留；未修改人工指南、
代码、实验分数或任务完成状态。本次仅静态核查与文档验证，未运行 GPU。
