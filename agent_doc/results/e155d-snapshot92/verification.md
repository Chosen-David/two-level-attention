# TL-E155D-SNAPSHOT-20261020 验证记录

2026-10-20；基线 `b418e220aae5cd86d4ba4ce1833ea79d4ad6bcde`。
本轮重新fetch main未变化，继续此前干净独立检出中的本任务补丁；
无AGENTS.md，已读根TASK、AI任务索引及人类guide，未编辑人类guide。
实验分支e138-ablation为`2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，未合并。
实现与优化技能用于固定输入契约、复现与验证，沿用可用标准库路线，未新增依赖。

## 证据绑定

- analyzer SHA256：`c719089b3c4e279abc1a42430d116802b901929054f936ec52100abbd4f44658`
- snapshot单测：`22adfabe8bb2ac0c17f6c3876e75b103e948b37ee88db3567a7eca1550496b50`
- 全入口差分脚本：`570d2cfcd49ef883e2985f7340a81be9c6243a86dedaefe2f04f007db89dcd34`
- 差分合成输入SHA256：`3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9`

保留[修前失败原始日志](before.log)，9项为4失败、3错误；其中KeyError反映
旧实现缺少新增身份字段，不能把这些错误全部当独立真实生产触发。
源替换反例直接观察到 `AssertionError: -50.0 != 25`；删除源出现FileNotFoundError。
最终源码及测试哈希与[既有独立复核](independent-review.md)完全一致。
独立复核实际执行过替换、Unicode/三种换行和读取中覆盖验证；本轮只重跑主控回归，
不虚称再次启动独立审查者。

## 本轮命令与原始短日志

```text
python -m unittest discover -s tests -p 'test_e155d*.py'
Ran 41 tests in 0.094s / OK
python -O -m unittest discover -s tests -p 'test_e155d*.py'
Ran 41 tests in 0.077s / OK
python tests/check_e155d_snapshot.py
PASS: 24 full-entry baseline differential cases; input_sha256=3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9
python -O tests/check_e155d_snapshot.py
PASS: 24 full-entry baseline differential cases; input_sha256=3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9
python tests/check_e155d_score_units.py
PASS: 24 baseline differential cases; actual aggregate scorer scale
python -O tests/check_e155d_score_units.py
PASS: 24 baseline differential cases; actual aggregate scorer scale
python -m unittest discover -s tests -p 'test_explicit_quota*.py'
Ran 10 tests in 0.089s / OK
python -O -m unittest discover -s tests -p 'test_explicit_quota*.py'
Ran 10 tests in 0.090s / OK
python tests/check_explicit_quota_modes.py
legal_state_equal=181; baseline_conflicts_accepted=15; candidate_conflicts_rejected=15
python tests/check_explicit_quota_oracle.py
legal_state_equal=212; baseline_conflicts_accepted=9; candidate_conflicts_rejected=9
```

41项含新增9项与既有收集/配对/量纲回归，不把多模式运行重复计为新增用例。
补充固定配额构造差分只是既有保护的回归，181/212不是本补丁新收益。
测试覆盖空收集、短行、不完整/冲突副本、ID错配、源替换/删除、暂存修改、
稳定输入、并列分、UTF-8与换行；不涉及张量batch/head/dtype/stride或prefill/decode状态。

## 环境及未验证边界

Linux6.18.55 x86_64/glibc2.39，CPython3.12.14；此前本机CPU探测AMD EPYC9V74。
本轮Torch/Triton均不可导入，nvidia-smi不在PATH；未访问历史GPU服务器，
没有GPU架构/驱动/kernel配置可报告。本任务纯标准库CPU验证，无编译kernel。
真实文件系统及完整aggregate入口已执行，但eval子进程与逐例metric用数值替身；
不能等同真实LongBench scorer、sidecar、模型/GPU全链路验收。
置信区间反例从错误-50恢复+25仅是合成正确性结果，不是质量提升。
没有速度、峰值内存、kernel数或host同步性能数据；测试耗时不是基准。

下一步：真实稳定T2输入重汇总、核对原始预测/sidecar；另行处理并发暂存与
跨格answers/input身份。固定配额near-width及T7依赖已授权且实际可用GPU，仍未完成。

## 发布前续接复验

2026-10-20再次获取main，仍为b418e22；e138-ablation仍为2ed6a1f，未合并。
独占本任务锁，已有脏树逐项与本检查点匹配，未涉及其他执行者修改。
源码与两份新测试SHA逐项等于上列独立复核输入；继续复用该范围验收，
没有冒称本次重新调用独立上下文。

本次普通/-O各41项E155d通过（0.077/0.085s），各24组全入口差分与
各24组量纲差分通过；输入hash不变。固定配额相邻普通/-O各10项通过
（0.129/0.105s），181/212合法构造差分保持一致。原始短日志另存本地
`exp/local_checkpoints/snapshot92-current*.log`，不将重复运行计作新用例。
本机Python3.12.14；Torch/Triton仍缺，nvidia-smi不可用，未联系任何远端作业。
本次仅恢复尚未发布的有界修复；不存在新增GPU或真实评分性能结论。
