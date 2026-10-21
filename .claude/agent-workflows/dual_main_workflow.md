# 双主 AI：方案主控与独立审核主控

此协议适用于受管的复杂项目 DAG。简单且明确的独立问答保留快速路径，不为每次回答建立项目、第二模型或监督服务。科研、通用任务和旅行项目采用相同权限边界；原有 15 个专业角色保持各自职责。

## 两个主入口

- [planner-main](../prompts/planner_main.md)：维护用户目标、唯一 TASK、方法、资源、依赖、验收和执行，作为任务索引与计划的串行作者。
- [review-main](../prompts/review_main.md)：由宿主以另一真实调用、独立新上下文启动；独立检查目标是否合理、指南优先级、前提、已有结果取舍、验收、风险和资源。不能写人类指南、替用户批准或把自己兼任为方案作者。

两个主控可使用同一供应商/模型的不同真实上下文，不硬编码品牌。独立性来自可信宿主的实际调用记录与上下文隔离；JSON 中的 owner、model、角色名和“独立”标记不能证明独立。同一模型切换语气、同一上下文第二遍反思不是这个门禁。缺独立调用或认证接口时明确 blocked，告诉用户需要哪个适配能力；不能声称已经有两个实时主 AI。

## 不可变审核对象与反馈

审核对象是实际执行的完整 DAG，包括未知的适配器字段、用户目标/授权引用、所有任务与依赖、方法/输入/输出、done_when、资源预算、task_refs、项目索引、稳定详情 Plan、已核实指南及采纳建议、历史结果查询/取舍和 evidence_refs。只有顶层 plan_review_receipt 从计划哈希中排除，避免自引用；不能排除可能影响执行的字段。

- prepare 默认加入 main-plan-review/v1、稳定 lineage_id、显式 evidence_refs。证据路径/哈希必须来自当前项目实际读取的材料，不靠摘要反推验收。
- 宿主建立 ReviewSession，认证生成该确切计划的 planner 调用；请求绑定项目根身份、完整计划 SHA-256、run_id、逻辑 lineage、随机 request_id、cycle、deadline 和前轮完整反馈哈希。
- reviewer 返回完整 full_review 与独立 compact verdict。verdict 必须包含 intent、guide、assumptions、prior_results、acceptance、risk、resources 七项检查；每项 pass/fail/unknown 与理由。未知关键条件不能 approve。
- 决定为 approve / revise / reject。revise 或 reject 给出目标位置、完整问题、具体修改和复验方法；approve 不允许仍有 blocking findings。运行时缺能力/预算/证据是 blocked，区别于审核意见 reject。
- 反馈原文完整保存，不截断、不只保留赞成/反对结论。单份响应上限 1 MiB，超限明确拒绝，不能静默压缩。版本化回执保存完整反馈；适配器按需引用历史，必要时读取全文，不把全部聊天历史塞入每次调用。
- 可信 authenticate 回调重新检查 request_sha256、review_sha256、实际 planner/reviewer invocation_id/context_id/host_source、fresh_context 与 budget_valid。仅返回 True 不能通过。reviewer proof 可是宿主签名或宿主只读记录的标识；它本身不是信任来源。

用户当前明确决定和宿主权限始终优先，已核实人工 guide 约束项目规划。审核发现指南冲突必须返修/拒绝；两个 AI 一致也不能覆盖指南或扩大权限。整个 agent_doc/guide/ 永远由人类维护。

## 有界返修与实际执行

1. 在当前项目 .agent-runs/plan-review.sqlite 以固定逻辑 requirement lineage 保存预算和每次请求；run_id 变化不重置。请求在 reviewer 调用前占用一次额度，失败/未知响应仍计数。
2. 一次 submit 只调用一次 reviewer，没有隐藏优化或返修循环。回调失败/输出未知时不盲目再调用；宿主可用 Session.resolve(plan, request_id, response) 接收同一请求的已认证原响应，须在原 deadline 内，不能退款、重置 cycle 或替换已冻结反馈。响应无法认证或已过期时仍阻塞，当前 API 不提供预算重置。revise/reject 保持未完成；planner 读取完整反馈，按授权修改为新的不可变 DAG 与 run_id，再提交新审核。Store 不允许原 run_id 换计划。
3. max_cycles（1–20）与 max_seconds（1–86400）由可信宿主配置，首次使用后不能用新计划/新 Session 提高。耗尽时停止依赖分支并请求明确重编排；不是 done，不重置预算重跑。
4. 宿主负责模型调用超时、取消及收费/token 上限。本地计时检查可拒绝迟到回执，不能抢占任意挂起回调。成本/token 未计量就记 unknown；硬预算无法核实就 budget_valid=False。
5. approve 后才可经 Engine/ManagedEngine 派发；每次派发、恢复、完成复核、最终报告、发布 freeze 与 acceptance 都重新认证当前回执并检查依赖。此步骤复用完整已审核响应，不每个 poll 重跑 LLM。
6. 稳定 Plan、指南、已采纳建议、明确证据、验收或适配器字段变化必须新版审核。详情中普通 ## Progress 追加不改变稳定计划身份。新结果搜索出现候选/缺口变化也不能静默替换被审核的 dispatch 输入，应重编排。
7. 派发后若审核撤销/失效，保留原 outcome/evidence 并持久标记 requires_reconciliation；重复 wake、恢复认证或重启不能自动重做已发生的作用，必须按原授权经显式 reconcile 核验。独立计划审核不代替 authorize、verify_experiment_result、角色验收、publication acceptance 或用户发布许可。实验每轮仍是 producer → 独立代码/数据 gate → consumer；生产者 handler 不可藏未审核实验循环。

该 SQLite 账本和显式回调是应用级可信边界，不是防同权限任意文件/SQL 写入者的安全沙箱。历史 DAG 可只读恢复；执行 legacy-unprotected 必须由宿主显式 allow_legacy=True / CLI --legacy-unprotected，报告明确未受双主审核保护。已保护的逻辑要求不能靠删字段、换 run_id 或换 lineage 退出；没有字段本身不能证明历史身份。

## 完整 checkout 的可执行接入

先用现有 prepare 生成并配置所有真实任务/验收，再由宿主适配器审核：

```bash
python -m agent_runtime.task_supervisor prepare --project-root /project --run-id project-v1 --lineage-id project-goal-1 --authorization-reference actual-user-message --out /project/.agent-runs/draft.json
python -m agent_runtime.task_supervisor review --project-root /project --plan /project/.agent-runs/draft.json --state-dir /project/.agent-runs/review-host --adapter /trusted/host_adapter.py --out /project/.agent-runs/reviewed-v1.json
python -m agent_runtime.task_supervisor start --project-root /project --plan /project/.agent-runs/reviewed-v1.json --state-dir /project/.agent-runs/v1 --adapter /trusted/host_adapter.py
```

明确指定的可信 adapter 的 build(root, store, run_id) 提供：

- 原有 handlers / authorize / result_verifier / maintain；能力与授权不变。
- plan_review_session：agent_runtime.plan_review.ReviewSession，配置 planner_identity(plan)、reviewer(request, plan, previous_receipts)、authenticate(request, review, proof)。每个回调必须接入实际宿主，不能从计划文件 import、执行 shell 或相信自报角色。
- plan_review_verifier：该 Session.verify（或等价可信认证器）。重启后重建同根/同 lineage/同预算的 Session，认证器仍能核对原真实调用记录及撤销。

review 命令无后台自循环，不自动安装模型或更改凭证。缺适配器停在具体缺口，返回的 revise/reject 计划保留全文，不可直接 start。纯 Skill 安装仅带入口与协议，必须确认完整 checkout 和真实宿主能力后再调用；仓库更新不代表旧会话或本机安装已更新。

PublicationLedger(root, state_path, plan=reviewed_plan, plan_review_verifier=session.verify, authorize=..., accept=...) 把同一计划固定到候选；冻结及每次接受重新核验。它仍不执行 git commit/push，也不授予发布权限。

## 证据等级

CPU fixture 只验证持久化、哈希、拒绝/返修/权限等程序控制。带真实独立主 AI 输出的宿主桥接演示应注明模型调用发生位置、实际调用身份、读取范围及回执绑定；不能把离线回放说成自动部署。模型质量、成本、token 和任务成功率收益需同条件另行测量。上游访谈和成功案例只启发方案，不是本仓库收益证据。

## 仅发布用途的可信任务范围

大型项目的历史/未完成任务不能被本轮发布计划冒领。默认 validate_contract 继续要求全部 canonical TASK ID，计划 JSON 中添加 scope 不能放宽它。只有发布主控明确配置 ReviewSession(publication_task_refs=[本次准确 ID]) 才审核这个发布范围；全部 canonical 文档、guide、advice 与稳定 Plan 仍绑定，覆盖必须与可信范围完全相等。

范围和 purpose=publication 固定在宿主账本/请求中，不能通过换 run_id 或重建 Session 扩大/缩小；先完整验证文档与范围，再登记受保护 ID。PublicationLedger 同样由可信主控传 publication_task_refs，要求与本轮实际 file_task_refs 集合相等并重新认证。未选中的历史任务不因此保护、重跑或完成。发布用途回执不能用于 Engine 或 managed dispatch，默认执行用途仍要求完整项目范围；这是专门的发布核验入口，不是通用计划自行省略要求的选项。

相对 task_source.path 可移植保存，但运行时必须显式提供可信 project_root；Store.create(plan, project_root=root)、ReviewSession 和 Engine 使用同一根。回执绑定实际项目身份，搬移后需重新认证新请求，不能跨项目重放。

## 预算耗尽后的人工宿主恢复边界

同一逻辑要求确实需要新的已授权规划阶段时，只有用户明确批准新的目标/预算边界、并由有权限的可信宿主维护者核对后才可迁移。必须保留旧 lineage、全部请求/完整反馈、实际预算消耗、原始失败、已发生作用与未决 reconciliation、旧版本证据和新授权来源；先处理尚在执行或作用不明的调用，不并行重做它们。

本发布不提供清空保护、改写账本、增加额度或创建替代规划阶段的自动 API，也不提供通用预算重置 CLI。resolve 只接收同一未决请求在原 deadline 内的真实响应，不能开启新额度。缺少经过验证的宿主迁移能力时停在明确的 host intervention required，不能靠换 run_id、复制目录、改 ID 或宣称“新会话”绕过。该限制不影响独立且已获授权的其他工作。
