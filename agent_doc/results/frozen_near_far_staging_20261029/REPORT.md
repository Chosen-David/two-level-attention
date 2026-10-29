本轮已完成：5 个海选任务各 20 条、100 条配对请求、500 组生成；这不是每任务 200/500 条的全量海选。

- 双向 skip，阈值 0.1：AVG5 54.0548，相对 Full +0.1759 pp；95% bootstrap CI [-0.2380, +0.6534]。Near 在 23/100 条请求中触发；层状态为 Near-only 92、Far-only 795、双 skip 4。
- 双向 skip，阈值 0.15：AVG5 53.3765，相对 Full -0.5024 pp；95% bootstrap CI [-2.4623, +1.4752]。Near 在 93/100 条请求中触发；层状态为 Near-only 378、Far-only 1033、双 skip 133。

平均分没有下降也不能声称逐任务无损或总体等价；各任务和逐条变化必须分别审阅。CI包含0不证明等价。
Near-only 的 KV 暂存继续向 Far 迁移，所有 Near-only 层的驻留 KV 比例为 1；其收益仅体现在 attention 读取量。表中的 KV/attention 比例是节点容量与 key 数量口径，不是整体 GPU 显存或实测速率。
双向策略按层选择 Near/Far/两者/都保留，不是所有层都丢 Near+Far；共同质量预算也会改变部分 Far 决策，因此 dual-vs-far 不是纯粹的 Near 增量消融。
结果来自独立 eager 分区缓存，不包含生产 TLI L1/L2 选择器。已完成本地可执行复核，未表示独立第三方审阅或论文主表验收完成。

多跳单独核查，musique ε=0.1：n=20，均分变化 +0.0000 pp，逐条下降 0、改善 0，Near实际触发 9 条。
多跳单独核查，hotpotqa ε=0.1：n=20，均分变化 +0.0000 pp，逐条下降 0、改善 0，Near实际触发 4 条。
多跳单独核查，musique ε=0.15：n=20，均分变化 +2.5000 pp，逐条下降 0、改善 1，Near实际触发 19 条。
多跳单独核查，hotpotqa ε=0.15：n=20，均分变化 -1.0000 pp，逐条下降 1、改善 0，Near实际触发 20 条。
小样本不推翻此前大样本中可能掉点的观察。

<!-- frozen-summary:end -->
# Frozen near/far cache exploration

Status: complete; paired requests 100/100.

| arm | AVG5 | near-active requests | far-active requests | frozen KV / full-node KV | attention tokens / full |
|---|---:|---:|---:|---:|---:|
| full | 53.8789 | 0 | 0 | 1.0000 | 1.0000 |
| far_0.1 | 54.0283 | 0 | 84 | 0.8842 | 0.8870 |
| dual_0.1 | 54.0548 | 23 | 84 | 0.8865 | 0.8815 |
| far_0.15 | 54.3552 | 0 | 84 | 0.8208 | 0.8250 |
| dual_0.15 | 53.3765 | 93 | 84 | 0.8164 | 0.7799 |

Per-task scores (0–100):

| arm | qasper | hotpotqa | musique | gov_report | repobench-p |
|---|---:|---:|---:|---:|---:|
| full | 53.2381 | 65.5238 | 45.0833 | 35.2494 | 70.3000 |
| far_0.1 | 53.5136 | 65.5238 | 45.0833 | 34.4710 | 71.5500 |
| dual_0.1 | 53.5136 | 65.5238 | 45.0833 | 34.6035 | 71.5500 |
| far_0.15 | 53.3870 | 65.5238 | 45.4833 | 34.2818 | 73.1000 |
| dual_0.15 | 49.9215 | 64.5238 | 47.5833 | 33.1539 | 71.7000 |

Paired differences (percentage points):

- far_0.1-vs-full: +0.1494; bootstrap 95% CI [-0.25498600852840336, 0.6426625120610266]
- dual_0.1-vs-full: +0.1759; bootstrap 95% CI [-0.23802125091472612, 0.6534001593515305]
- far_0.15-vs-full: +0.4763; bootstrap 95% CI [-0.3067285665853365, 1.3919100377080886]
- dual_0.15-vs-full: -0.5024; bootstrap 95% CI [-2.46230475626965, 1.4752271649095376]
- dual_0.1-vs-far_0.1: +0.0265; bootstrap 95% CI [-0.03696811219982862, 0.09264855777079198]
- dual_0.15-vs-far_0.15: -0.9787; bootstrap 95% CI [-2.8614597082401345, 0.891600178161849]

Exploratory paired request bootstrap, 5000 repeats, task-stratified macro AVG5; CI containing zero is not proof of equivalence. Near-only retains staging KV and skips attention work only; far-only/both physically evict nodes. No TLI champion selection or performance claim.

KV ratios measure allocated cache-node capacity, relative to the same full-node cache; they exclude model weights, gather temporaries and allocator behavior. Attention-token ratios are a pre-decode work proxy, not measured speedups.
Dual gates share a combined calibrated mass budget: adding near gating may replace an eligible far decision. Dual-vs-far is a policy comparison, not a pure near-only ablation.
Near-active and far-active request counts include both-skip states.
