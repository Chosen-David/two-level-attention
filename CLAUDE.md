<!-- chosen-david-agent:begin -->
## Chosen-David Agent Workflows

简单、目标明确的任务直接完成，仅加载相关 Skill。普通低风险本地修复运行必要回归并汇报结果；不自动建立 DAG、第二 Agent、实验合同或 tmux 监督器。
复杂协作、长期运行或用户明确编排时，先读 `.claude/agent-workflows/orchestrator.md`，再按需读 planner_main.md 与 review_main.md；完整流程不预加载。
工作流源码位置见 `.claude/agent-workflows/installation.json` 的 repository；project_root 是当前项目。源码工具从仓库解析，项目输入和输出始终相对当前项目。
Project A 的指南、任务、建议和结果属于 Project A/agent_doc/；缺失时不回退到工作流库的历史文件。切换项目重新核对指南、任务、记忆和交接中的 project_root。
已核实人类 agent_doc/guide/GUIDE.md 是最高项目规划依据；AI 绝不创建、编辑、删除或移动 agent_doc/guide/ 内文件。
agent_doc/task/TASK.md 是唯一日期索引，关联方法/进度/证据在 agent_doc/task/task_details/*.md；已有内容和 Plan/Progress 边界保留。简单任务不另建文档体系。
agent_doc/advice/ 必须核验并记录 adopt/adapt/reject/defer 与理由，不能当授权；尊重当前宿主权限和用户约束。
实证依据按 Skill 的 references/result_reuse_workflow.md 检索 agent_doc/results/ 并核对适用条件；实验/测量/共享结果按 references/result_validation_workflow.md 独立 verify_experiment_result，原始数据保持 pending 直到 usable-with-scope。用户/指南要求和已建立的实验合同不得降格。
复杂任务的 planner-main/review-main、结果独立验收需要真实宿主支持；安装不启动模型或后台服务，缺后端不得把自审当独立批准。
角色模型/路由见 model-policy.json、model-routing.json；源码 scripts/run_role.py 仅使用可信宿主适配器，缺候选继承默认模型，执行失败不自动换模型重跑。
交付核对实际修改、必要检查和剩余事项；仓库文件存在不证明后台进程运行或模型已遵循规则。
<!-- chosen-david-agent:end -->