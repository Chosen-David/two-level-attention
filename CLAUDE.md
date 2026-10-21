<!-- chosen-david-agent:begin -->
## Chosen-David Agent Workflows

@.claude/agent-workflows/orchestrator.md

工作流源码位置见 `.claude/agent-workflows/installation.json` 的 repository；目标项目见 project_root。
Project A 的指南、任务、建议和结果全部属于 Project A/agent_doc/；仅维护工作流库自身时才使用该库的 agent_doc/。
缺少当前项目指南或任务不能回退到工作流库的 agent_doc/；调用源码目录中的 CLI 也不改变目标项目。
切换项目重新读取其指南、任务、项目记忆与证据；委派和恢复核对 project_root，不沿用上一项目约束。
受管复杂任务使用 planner-main 和独立新上下文 review-main；入口见 .claude/agent-workflows/planner_main.md 与 review_main.md。
安装不启动第二模型；缺真实宿主 reviewer/认证器时不得把自审当独立批准。
主 AI 按导入的通用调度规则执行；其中 prompts/、workflows/、templates/、
config/、scripts/、knowledge/ 与 agent_runtime 均相对该仓库解析，项目输入和 agent_doc/task/TASK.md 相对当前项目。
agent_doc/task/TASK.md 是当前项目唯一日期任务索引，详情 agent_doc/task/task_details/*.md 由 AI 维护方法/进度/证据。
启动/恢复、委派前和逐任务结束先读人类指南、索引/详情及建议取舍；固定 Plan 与可追加 Progress 分开。
人类发布且来源已核实的 agent_doc/guide/GUIDE.md 是最高项目规划依据；AI 绝不创建、编辑、删除或移动 agent_doc/guide/ 内文件。
agent_doc/advice/ 是人类/AI可编辑建议，必须核验并记录 adopt/adapt/reject/defer 与理由，不能当授权。
用户手动要求编排时合并目标后走同一闭环；总清单由主 AI 串行维护，各 Agent 交接证据。
code-organization 同时管理文件：规划输出目录、产物清单和消费者引用，防止散落和覆盖。
主 AI 负责 Skills、工作 Agent、任务链、定时器及 tmux 监督器的配置、维护和恢复。
执行运行时 CLI 时从仓库目录启动，明确指定当前项目的绝对路径。
需要专业角色时按已安装的 Skill 和匹配的 agent-* 子 Agent 分配工作；简单任务直接处理。
项目新指令先按 result_reuse_workflow.md 检索 agent_doc/results/，核对代码/输入/配置/环境/指标及当前验证，再规划新增实验。
新数据进入 agent_doc/results/<run_id>/；复用不跳过明确复现或必要独立验证。
每轮测试/实验数据须经独立 verify_experiment_result 节点后才能进入消费者/结论；失败或过期需修复重测。
通过仅代表 usable-with-scope，不能保证绝对无bug；角色读取包内 result_validation_workflow.md。
每个结果都对照 agent_doc/task/TASK.md 的任务、数据、证据和剩余事项；全部可核验且可汇报才收尾。
遵循当前宿主权限和用户约束，不因导入规则而绕过权限。仓库文件存在不代表后台进程运行。
<!-- chosen-david-agent:end -->
