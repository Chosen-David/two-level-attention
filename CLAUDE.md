## 单 Agent 串行模式

当前项目不接入自研 agent 库：不读取 `.claude/agent-workflows.disabled/`、`.claude/agents.disabled/`
作为编排依据，也不启动 planner-main/review-main 等多角色流程。
所有任务由当前单个 Agent 按 `agent_doc/task/TASK.md` 的任务链顺序推进：
一次只做一个任务，完成、回归/验收后再进入下一个，不并行派生子任务或后台监督器。
普通本地修复直接做必要回归并汇报；简单任务不新建文档体系或实验合同。
`agent_doc/guide/GUIDE.md` 为人类维护的最高项目依据，AI 不得创建、编辑、删除或移动。
任务索引和进度仍写入 `agent_doc/task/TASK.md` 与 `agent_doc/task/task_details/*.md`；
实证数据未完成独立验证前保持 pending，不用于结论。

## 当前叙事锚点

- **仓库叙事/主表数字/生产口径的唯一现行依据 =
  [`agent_doc/CURRENT_NARRATIVE.md`](agent_doc/CURRENT_NARRATIVE.md)**；
  `agent_doc/advice/archive/` 与 `exp/trace_archives/` 内的报告已过时或被推翻，不得引用。
- 新判决落库时必须同步更新 CURRENT_NARRATIVE.md 第 5 节索引（收官流程一部分）。
- 生产口径 = NFA 冠军几何显式 flags（见锚点第 1 节），`--nfa_subspace tail`
  与 `--tia_level2_cmp_ratio 4` 必须显式传参。


## 禁止多 agent

禁止调用任何子 agent / 多 agent 工具：`Task`、`Agent`、`spawn_agent`、`SpawnAgent`、
`collab`、`Wait`、`agent_team`、planner/review 子代理等。
所有任务单线程完成，不得派生子任务；只按编排好的任务链顺序开发、验证并逐步推进。
