# 2026-10-29 首轮30分钟AI监督执行报告

状态：**全量运行中，独立验收 pending，未完成**。截至 2026-10-29T22:34:41.536143+08:00，425/1300 条完整三臂配对、1279/3900 次生成已落盘；16/64 分片完成，8 运行，40 待领取，0 失败。无 FINISHED 或 COMPLETE 科学验收完成声明。

## 已完成与本轮实际动作

- 读取固定README、dispatch_plan、verification_plan、任务详情、人类GUIDE和GPU派单/文档/结果验证/复用规范；GUIDE未修改。主项目TASK索引原缺此任务，已补登同一任务链接，未另建任务索引。
- 多次SSH逐机精确检查watchdog、daemon及当前worker的PID/父进程/命令；同步核对GPU显存利用率、heartbeat、日志、逐样本JSON和完成标记。没有只凭PID判断健康。
- 初始完整落盘快照272配对/825生成，末次425配对/1279生成，净增153配对/454生成；队列完成8→16。首轮无前轮md，因此以本轮首次快照为增量基线。
- 核实完成worker释放GPU后，由原有300秒daemon自动接上下一批；接力后的8个worker均真实加载并继续生成。未新派任何GPU任务、未重启/停止进程、未抢lease、未改重试次数或调度周期。
- 三机实际源码HEAD均为3e4a6885604ac3e719ee92a4906f87b532535314；manifest绑定的72个源码文件哈希全匹配。未更改共享源、门限或生成协议，未用本机GPU1，未重启E184。
- 旧screen独立审查通过单独Codex会话执行。collaboration后端缺rollout、首个只读沙箱bwrap权限失败均保留；后续使用宿主现有权限但严格只读CPU操作完成源码/数据审阅及评分补验。没有创建新的调度器/监督器。
- 本机缺rouge/jieba的评分缺口已利用pc02既有Python环境解决；主控及独立会话分别复算300/300官方评分，误差≤1e-12。无需安装依赖、无新生成。
- 为参考边界缺口尝试针对性pytest，因远端缺pytest未执行（保留失败回执）；随后运行只读CPU参考脚本，32步Near staging跨页可见性、5个严格阈值用例、8个query的显式mass对照通过。此为主控回归证据，尚未经独立接收，不能把screen整体pending改为通过。

## 全量当前覆盖

| 数据集 | 完整配对 | 目标 | 已保存生成 |
|---|---:|---:|---:|
| Qasper | 80 | 200 | 240 |
| HotpotQA | 80 | 200 | 240 |
| Musique | 77 | 200 | 231 |
| GovReport | 58 | 200 | 178 |
| RepoBench-p | 130 | 500 | 390 |
| 合计 | 425 | 1300 | 1279 |

GovReport另有4条部分完成请求。快照为遍历时点观察，队列/逐文件可有秒级时差；中途完成标记先于300秒队列更新是正常接力间隙，后续已确认推进。末次三机daemon年龄约58–76秒，worker heartbeat均近期更新，最老约36秒；最终GPU显存17508–25838 MiB、利用率37%–97%，三机SSH均成功。首版snapshot以开始时刻减读取期间新写心跳出现约−1.5秒，属采集时间差；后续采集脚本已用逐次读时计龄，原始快照保留不改。

全量质量判断仍需余下875条配对及完整独立验收。未根据部分样本输出全量AVG5或Musique掉点判断。

## 旧100配对screen独立核查事实（仅辅证）

独立报告：independent_screen/REPORT_access_retry.md；官方评分补充：independent_screen/SCORER_FOLLOWUP.md。旧screen源码a1cdb2f...与全量3e4a688...不同，核心gate/cache/adapter相同，但全量抽样入口有变化；不混结果。

| 任务 | 样本 | gate-off | Far-only / dual | 配对差（百分点） |
|---|---:|---:|---:|---:|
| Qasper | 20 | 56.100883 | 54.042059 | −2.058824 |
| HotpotQA | 20 | 65.523810 | 65.523810 | 0 |
| Musique | 20 | 45.083333 | 45.083333 | 0 |
| GovReport | 20 | 35.227082 | 34.835478 | −0.391604 |
| RepoBench-p | 20 | 72.050000 | 71.950000 | −0.100000 |
| 宏AVG5 | 100 | 54.797022 | 54.286936 | −0.510086 |

这批固定预测的评分确有下降；**Musique这20条未下降**，不外推剩余样本。Far与dual的100条生成序列相同，不证明算法等价。后续审查提示曾误写Musique下降，已由独立核查纠正，并保留followup_prompt_correction.md。无事前等价界值，不能声称无损。

实际dual门控共3600个层—请求决策：keep3272、far-only326、near-only1、both1；后两类各占0.02778%，各仅涉及1个Musique请求。唯一near-only请求没有跨页老化，因此screen缺少此分支的实际迁移覆盖。审查确认代码严格独立<0.05、末256query mean、decode冻结、Near staging attention不可见/Far删除，且记录cache守恒一致；有限记录不能证明所有逐步执行或GPU分支普遍正确。full是TLI gates-off，不是dense；KV节点容量/attention token计数不是速度或实际峰值显存收益。

## 当前阻塞与下一步

没有发现需要恢复的调度故障。全量尚未完成；旧screen整体独立验收仍pending，参考边界、生成复现、权重/历史依赖绑定及事前完整合同缺口如独立报告所述；不事后伪造事前计划、不把缺项标N/A。新CPU参考检查可交下一独立会话核查，但不消除真实8B分支覆盖限制。

继续保持现有8卡自动接力。全量到齐后核查自动validate→analyze→export、1300唯一覆盖/3900生成；独立verify_experiment_result后输出各任务和宏AVG5配对差、置信区间及实际Near-only/Both覆盖，Musique单列，保存中文最终报告。汇总故障按实际锁所有权修复，不能另行争写。执行清单见../FULL_VERIFICATION_HANDOFF.md。

本轮未写COMPLETE.json，也没有最终实验报告；下一轮从../NEXT_STEPS.md继续，监督间隔1800秒保持不变。

## 证据入口

- snapshot_initial.json、snapshot_final.json及中间快照：覆盖/daemon/heartbeat/日志/SSH实况。
- full_source_hash_check.json：72个源码哈希。
- independent_screen/session_access_retry.log、scorer_followup.log及对应两份报告：实际独立会话与检查证据。
- screen_official_rescore.py/json：主控评分复算及逐文件SHA256清单。
- targeted_boundary_regression.json：缺pytest的真实失败；targeted_boundary_reference.py、targeted_boundary_reference_result.json：替代CPU参考检查。
- context_review.md：项目依据、复用取舍与后端恢复过程。
