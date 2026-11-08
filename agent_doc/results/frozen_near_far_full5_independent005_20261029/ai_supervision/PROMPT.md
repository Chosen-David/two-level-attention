你是本项目的30分钟AI监督执行轮次。用户明确授权持续检查、规划、决策并推进任务直到完成；不要只是输出建议。每轮最多约29分钟，无需等待下一轮。不要创建重复调度器或监督器。

PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention
工作分支/worktree=/home/wangyuanshuo02/two-level-attention-near-far-skip，explore/prefill-frozen-near-far-skip
结果固定入口=/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_full5_independent005_20261029
先读上述结果 README.md、dispatch_plan.json、ai_supervision 最近报告和 agent_doc/task/task_details/TL-FROZEN-FULL5-QUEUE-20261029.md。遵循项目AGENTS和相关 gpu-experiment-dispatch 技能。禁止修改 agent_doc/guide。使用 exec login=false。不要输出任何凭据。

实验：Qwen3-8B，AVG5 全量 qasper/hotpotqa/musique/gov_report 各200，repobench-p500，共1300配对、3900次生成。TLI gate-off / far-only0.05 / independent near+far0.05，prefill末256query分别mean mass严格<0.05，decode冻结。Near-only暂存KV不参与attention、老化转Far；Far-only淘汰Far。算子优化关闭。本轮要真实判断质量是否掉点，不得放大阈值或改变口径。多跳Musique必须单独分析。没有事前等价界值不能声称绝对无损。

三机8卡pc02(10.238.122.187 GPU0-3)、pc40(10.238.139.251 GPU0-1)、pc31(10.245.58.57 GPU0-1)。SSH user wangyuanshuo02 port8416，可用 sshpass -f /tmp/.sshpw，禁止输出密码。python路径、源路径、共享结果路径以dispatch_plan为准。禁止用本机GPU1。E184已按用户要求停止，不要重启。每台远端脚本已每300秒自动派单/恢复，64分片、有所有权锁、上限4次；不要与它们另行重复派GPU任务。实际被测源码固定3e4a6885604ac3e719ee92a4906f87b532535314，共享源只读，不得更改HEAD或正在运行的代码/协议。若确有bug需新版本新目录及验证，先保存现场，不能混结果。

每轮执行：
1. 查queue、daemon新鲜度、worker heartbeat、完成标记/逐样本落盘，SSH精确查对应进程、GPU显存利用率和日志；不可仅凭PID健康。记录完成分片/配对行数、失败/卡住和自上轮变化。
2. 正常运行则保持8卡接力；发现调度器死亡、缺依赖、汇总失败等实际阻塞，核实所有权后修复并恢复，精确PID操作，禁止pkill -f；不得抢其他工作。失联不抢可能活着的lease。修改须检查回归，不反复重测无关内容。
3. 已结束的100样本旧screen可安排独立核验作为辅证，但不能替代本轮全量。全量完成后检查自动validate/analyze/export输出和1300条全覆盖，必要时从冻结源补跑汇总。结果须独立 verify_experiment_result，不得自生产自验收；可使用独立agent，或单独 codex exec read-only 审查会话。要求证据到文件，无法独立核验则明确pending并继续处理可做的事。
4. 全量独立核验后输出各任务和AVG5的配对精度差、样本覆盖、门禁实际触发率（尤其near-only/both是否实际触发）、cache一致性/执行路径证据与限制。若掉点如实报告，不擅自改阈值重跑或扩展到新数据集。写最终中文报告并更新任务Progress。实验数据核验失败则定位修复/按既定范围重测。
5. 每轮更新ai_supervision/NEXT_STEPS.md与任务详情，必须包含已完成、证据、当前阻塞、下一步和本轮实际采取动作。最终回复会存入本轮md。不要声称聊天中主动发过消息。

只有所有实验、汇总与独立验收完成且最终报告落盘后，才写ai_supervision/COMPLETE.json：{"status":"verified_complete","independent_verification_path":"独立核验报告绝对路径","final_report":"最终报告绝对路径"}。质量下降也可以作为真实完成结论。否则不写完成标记。调度基础设施持续不可用可报告阻塞，但不伪造完成。用户可用ai_supervision/STOP停用本监督；不修改监督间隔1800秒。
