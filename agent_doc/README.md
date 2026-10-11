# two-level-attention 项目 agent_doc：任务、指南、建议与结果

这是 **two-level-attention 主项目**（TLI/PSI 稀疏注意力索引器研究主战场，
独立 git 仓库）的项目文档区，与 `docs/`、`exp/` 等普通文档/数据分开，遵循
agent 库的 [项目文档治理](https://github.com/Chosen-David/agent) 布局。

工作流源码与模板从 `~/agent-research-workflows`（agent 库，当前同步至
`39c6d42`）读取，本目录只放 two-level-attention 项目自身的任务、建议与
结果，不复制 agent 库的指南或历史记录。

## 接入关系

| 项 | 指向 |
| --- | --- |
| agent 库（远端） | `git@github.com:Chosen-David/agent.git`（本地 `~/agent-research-workflows`） |
| 本仓库远端 | `git@github.com:Chosen-David/two-level-attention.git` |
| 人类指南（映射） | `/home/wangyuanshuo02/sglang/TASK.md`（用户私人任务设定，AI 只读）与 `/home/wangyuanshuo02/sglang/论文indexer.md`（论文口径权威定义，异议写反馈） |
| 历史治理区 | `/home/wangyuanshuo02/sglang/agent_doc/`（主项目迁移前的治理存档，内容已迁移到本目录） |

本目录 2026-10-11 建立：task/advice/results 内容自 sglang 主仓
`agent_doc/` 迁移而来（主项目地位移交），后续由主 AI 在本目录维护，不再
双清单并行。

## 目录职责

| 目录 | 主要内容 | 谁维护、AI 如何处理 |
| --- | --- | --- |
| [guide](guide/README.md) | 人类目标、优先级、硬约束（含 sglang 根两个映射文件） | 人类发布；AI 只读，优先据此编排 |
| [advice](advice/README.md) | 人类、其他 AI、当前 AI 的建议与反馈 | 人类或 AI 记录；主 AI 核验证据后 adopt/adapt/reject/defer |
| [task](task/README.md) | AI 整合后的唯一任务索引与详情 | 主 AI 串行维护 `task/TASK.md` |
| [results](results/README.md) | 数据、运行元数据、产物与独立验证 | 生产者记录，独立验证者核验 |

处理顺序：当前用户决定与权限 → 根 TASK.md/论文indexer.md 等人类指南
→ advice 核验取舍、已有 results 适用性检查 → AI 整合任务 → 执行与独立
验收 → 按要求逐项汇报。

## 既有资产的映射

- 根 `CODEMAP.md` = 实验代号→脚本→数据→论文引用映射表（代码库已有）；
- `agent_doc/task/TASK.md`（AI）= 执行索引层，登记的每项任务须能追溯到
  用户指令或 sglang 根 TASK.md 条目；
- `exp/trace/results/*.json` = 历史结果落袋区（E 系列 JSON），
  `agent_doc/results/` 登记新 run 的索引与验证状态（JSON 仍落
  `exp/trace/results/`，不搬家）；
- `research/docs/` = 审查判决与研究文档（gpt_kimi3_verdict 等判决 JSON
  保持原路径——注：部分判决文档位于 sglang 主仓 `research/docs/`）。

模板见 agent 库 [templates](https://github.com/Chosen-David/agent)；
任务格式见 `~/agent-research-workflows/templates/TASK.md`。
