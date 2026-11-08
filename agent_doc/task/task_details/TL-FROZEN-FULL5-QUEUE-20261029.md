# TL-FROZEN-FULL5-QUEUE-20261029

## Plan v1
用户要求：三台远端各自常驻轮询空闲 GPU 并接力跑完现有 AVG5 全量，无需助手持续监督。保留独立 Near/Far 0.05 prefill 冻结与 decode 分区缓存，不改算子优化。1300 配对样本、3900 次生成、64 分片。结果固定到当前项目 agent_doc/results。

## Progress v1
实现位于分支 explore/prefill-frozen-near-far-skip，worktree /home/wangyuanshuo02/two-level-attention-near-far-skip。脚本 scripts/remote_frozen_queue.py 与 scripts/run_remote_frozen_queue.sh；说明 docs/frozen_full_queue.md。独立代码审阅和相关 63 项回归已执行；部署跨机 flock 检验发现不互斥，已暂停点火并改为原子文件系统锁，须通过三机验证后启动。

结果入口：[README](../../results/frozen_near_far_full5_independent005_20261029/README.md)。deployment_receipt.json 与 preflight/receipts.json 保存实际点火与环境身份。新全量结果须独立 verify_experiment_result；不得由调度完成推导精度无损。

## Progress v2 — 已部署
冻结源码 3e4a6885604ac3e719ee92a4906f87b532535314。跨机锁改为预写 owner + 原子 symlink 发布，同机 dead-PID 恢复串行化。实际三机并发 90 次锁内计数全部保留，模型/数据哈希一致；相关 68 项回归通过。三机 watchdog 已启动，PID 见 deployment_receipt.json；实际工作进程/GPU/日志证据见 launch_live_check.json。无需本机监督器或助手在线继续派单，轮询 300 秒，审计 1 小时。

## Progress v3 — 用户改为30分钟AI监督
已部署 scripts/supervise_frozen_ai.py，每1800秒启动独立 Codex exec 轮次（不并发叠加），读取交接/实况，自主诊断、决策、推进和落盘；远端300秒派单保留，不改冻结被测代码与协议。探针真实成功，调度生命周期检查通过。状态与报告在结果目录 ai_supervision/，NEXT_STEPS.md持续交接，只有全量独立验收及最终报告落盘后结束。创建 ai_supervision/STOP 可停止监督但不停止GPU任务。该监督依赖本机运行与Codex服务可用；AI请求失败下一周期重试。旧固定8rank小时监督只覆盖已结束screen，不重启。

## Progress v4 — 2026-10-29 22:35 首轮AI监督

首轮无历史md，按当前用户授权执行；主项目唯一TASK已补登本任务。三机watchdog/daemon/worker精确SSH、GPU、日志、heartbeat和逐样本多次核验，确认原300秒自动接力健康。22:35快照16/64分片完成、8运行、40待领取、0失败；425/1300完整配对、1279/3900生成；本轮较初始272/825净增153配对/454生成。数据集完整数80/80/77/58/130。冻结HEAD三机一致，72个源码哈希全匹配。没有生产修改、重派、停车、本机GPU1使用或E184重启。

独立后端collaboration失败及bwrap失败均保留；实际独立Codex只读CPU会话完成旧100配对screen源码/输入/缓存/评分审阅，官方300评分复算缺口已在pc02既有环境闭合。固定预测宏AVG5差−0.510086pp；Musique20条差0；Near-only/Both各1/3600层—请求。整体验收仍pending（参考边界/生成复现及完整历史资产/合同缺口），不可用于全量结论。补充主控CPU参考检查32步迁移、5门控边界、8query mass通过，尚未经独立接收；pytest缺失失败回执保留。

证据：[本轮报告](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T141704/ROUND_REPORT.md)、[独立补充](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T141704/independent_screen/SCORER_FOLLOWUP.md)、[下一步](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/NEXT_STEPS.md)。当前阻塞仅全量未齐与独立验收缺口；继续原8卡接力，全量到齐核自动汇总并独立验收，单列Musique与Near/Both覆盖，不改阈值。未写COMPLETE.json。

## Progress v5 — 2026-10-29 22:59 第二轮AI监督

三机精确SSH/进程/GPU/日志/heartbeat/逐样本盘点确认原300秒自动接力健康，pc40 rank22→31、pc31 rank24→32已观察到。末次25/64完成、8运行、31待领取、0失败；620/1300配对、1864/3900生成，较上轮末+195/+585，本轮首末+65/+190。任务完整数106/104/104/100/206。无生产修改、额外派单、停车、本机GPU1使用或E184重启。

实际独立Codex会话完成CPU边界补验（396缓存用例、400门控组合、20mass对照）和固定500配对评分审查（1500分误差0，548文件/72源码hash及54000层记录不变量通过）。固定子集限定usable-with-scope：Qasper Far/Dual均−1.227905pp；Musique80条+0.002874pp，0降/1升/79不变；子集宏Far+0.002692pp、Dual−0.005635pp，不是最终AVG5。GovReport98 Near-only请求Dual对Far单样本−2.997513pp；Dual Near-only15/18000层—请求（10/500请求）、Both0。不能称无损或外推全量。

证据：[本轮报告](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T144704/ROUND_REPORT.md)、[子集中文质量报告](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T144704/QUALITY_PARTIAL_ZH.md)、[独立评分](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T144704/independent_partial/REPORT.md)、[独立边界](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T144704/independent_boundary/REPORT.md)。

当前阻塞：余下680配对未齐，自动汇总和全量独立验收pending；无调度基础设施故障。下一步保持原8卡接力，完整核1300/3900后独立复算与门控/缓存审查，单列Musique和GovReport98/Near/Both覆盖，按既定协议输出最终中文报告。NEXT_STEPS已更新；未写COMPLETE.json。

## Progress v6 — 2026-10-29 23:31:46 CST 第三轮AI监督

三机多轮精确SSH/进程/GPU/日志/heartbeat/逐样本盘点确认原300秒自动接力健康。末次40/64 queue完成、8运行、16待领取、0失败，done标记41待收割；922/1300配对、2777/3900生成，较上轮+302/+913，本轮首末+135/+412。无生产修改、额外GPU派单、停车、本机GPU1使用或E184重启。

两个独立Codex只读CPU会话完成固定680配对2040官方评分（误差0）、680输入重建、2040输出解码和73440末步TLI回执核验；746文件/72源码hash通过。子集宏Far−0.053844pp、Dual−0.059447pp，Qasper−0.957999pp、RepoBench-p−0.059524pp；非最终AVG5，不称无损。Musique107条0降/1升/106平，+0.002148pp但唯一变化row82仍答错。Near-only27/24480层—请求（13/680请求），Both1/24480（Musique56层11）；GovReport98 Dual对Far−2.997513pp，记录8个64-token页老化转Far。原始GPU张量/生成复现、完整权重及逐步选择集合等限制保留。

证据：[轮次报告](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T151704/ROUND_REPORT.md)、[质量观察](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T151704/QUALITY_PARTIAL_ZH.md)、[独立输入/路径](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T151704/independent_inputs/REPORT.md)。

剩378配对未齐，自动汇总与全量独立验收pending，无实际调度阻塞。NEXT_STEPS已更新；继续原8卡接力，完成后全量独立复算与最终中文报告。未写COMPLETE.json。

## Progress v7 — 2026-10-30 00:14:06 CST 第四轮AI监督

三机多轮精确SSH/进程/GPU/log/heartbeat/逐样本核查，正常接力至末批。60/64完成、4运行、0待领取、0失败；1271/1300配对、3818/3900生成，较上轮+349/+1041，本轮+204/+607。全部64分片已领取，无生产修改、额外GPU派单/停车、本机GPU1使用或E184重启。

两个实际独立Codex会话分别验1227输入/3681输出/132516回执和固定1241配对3723官方评分（误差0）；绑定与日志已落盘。Qasper全200 Far/Dual−0.683604pp，11降2升187平、两CI不含0；HotpotQA+0.25pp。Musique全200 0降1升199平、+0.001149pp，但唯一row82仍答错，不能称推理提升/无损。固定子集宏Far−0.080125pp、Dual−0.083231pp，不是最终AVG5。GovReport98 Dual−Far−2.997513pp；评分子集Near-only33/44676层请求（19/1241请求）、Both2/44676（2/1241请求）。

证据：[轮次报告](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T154704/ROUND_REPORT.md)、[质量报告](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T154704/QUALITY_PARTIAL_ZH.md)、[独立评分](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T154704/independent_scores/REPORT.md)、[独立路径](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T154704/independent_paths/REPORT.md)。

当前剩29配对及自动汇总/全量独立验收pending，无实际调度故障；不同独立覆盖不可混称全量。NEXT_STEPS已更新，保持原接力，完整1300/3900后独立核最终汇总并写最终中文报告。未写COMPLETE.json。


## Progress v8 — 2026-10-30T00:33:46.499749+08:00 第五轮全量验收收尾

64/64分片、1300/1300配对、3900/3900生成全部完成，200/200/200/200/500。自动validate/analyze/export于00:26:50完成；三机watchdog/daemon自然退出，8GPU释放。较上轮+29配对/+82生成，本轮首末+9/+23；无生产修改、额外派单、停车或E184重启。

两个实际独立Codex会话完成同一1300集合验收：3900官方评分误差0，全部bootstrap/导出一致，1300输入/3900输出/140400回执/421200缓存快照检查通过；1533资产字节绑定无变化。限定usable-with-scope，不代表GPU生成复现或绝对无bug。

最终AVG5 Far−Full−0.082408pp、Dual−Full−0.085406pp，区间含0不等价；Qasper−0.683604pp且CI不含0。Musique200条0降1升199平、+0.001149pp，row82仍答错。GovReport98 Dual−Far−2.997513pp。Near-only33/46800层请求、19/1300请求；Both2/46800、2/1300。

证据：[最终中文报告](../../results/frozen_near_far_full5_independent005_20261029/FINAL_REPORT_ZH.md)、[独立验收索引](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T161704/VERIFICATION_INDEX.md)、[本轮动作](../../results/frozen_near_far_full5_independent005_20261029/ai_supervision/round_20261029T161704/ROUND_REPORT.md)。无本次范围阻塞；原始GPU张量/逐步选择/完整权重历史及kernel环境限制保留。不扩大阈值或数据，负结果按既定协议交付。NEXT_STEPS已更新；收尾核对后写COMPLETE.json。

收尾 2026-10-30T00:35:38.318071+08:00：所有最终证据再次核对通过；文档完成记录提交f3d4d2a8dc18cc3597778b9902c58ca63717c045并推送/核验bee和origin，生产冻结源不变。ai_supervision/COMPLETE.json已按全部实验、汇总、独立验收与最终报告完成条件写入。

## 用户接受标准更新 — 2026-10-30
用户接受下降3个百分点以内。现有全量AVG5及所有单任务均值下降均在该范围内，工程质量可接受；此为事后用户容忍度，不追溯称事前等价检验或无损。见结果 USER_ACCEPTANCE.md。全量指现有五数据集1300条，不扩为LongBench全部任务。
