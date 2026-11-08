自动生成结果：机械核验已通过，独立结果审阅待完成。

本轮已完成 16 个任务的全量数据：narrativeqa 200 条、qasper 200 条、multifieldqa_en 150 条、hotpotqa 200 条、2wikimqa 200 条、musique 200 条、gov_report 200 条、qmsum 200 条、multi_news 200 条、trec 200 条、triviaqa 200 条、samsum 200 条、passage_count 200 条、passage_retrieval_en 200 条、lcc 500 条、repobench-p 500 条；共 3750 条配对请求、11250 组生成。

- 双向 skip，阈值 0.05：AVG16 49.3257，相对 Full -0.1372 pp；95% bootstrap CI [-0.2430, -0.0338]。Near 在 105/3750 条请求中触发；层状态为 Near-only 319、Far-only 13734、双 skip 5。

平均分没有下降也不能声称逐任务无损或总体等价；各任务和逐条变化必须分别审阅。CI包含0不证明等价。
Near-only 的 KV 暂存继续向 Far 迁移，所有 Near-only 层的驻留 KV 比例为 1；其收益仅体现在 attention 读取量。表中的 KV/attention 比例是节点容量与 key 数量口径，不是整体 GPU 显存或实测速率。
Near/Far gate独立比较各自mass；同阈值下加入Near gate不会改变Far gate判定。
结果复用真实 TLI tail32 + L1/L2 选择器；full 代表 TLI 双 gate 关闭，非 Dense FullKV。当前为 eager 精度参考适配器，不作速度或总显存结论。已完成本地可执行复核，未表示独立第三方审阅或论文主表验收完成。

多跳单独核查，musique ε=0.05：n=200，均分变化 +0.0011 pp，逐条下降 0、改善 1，Near实际触发 7 条。
多跳单独核查，hotpotqa ε=0.05：n=200，均分变化 +0.2500 pp，逐条下降 0、改善 1，Near实际触发 4 条。
本轮覆盖所列任务的全部样本；结论仍限于本模型、数据集与生成配置。

<!-- frozen-summary:end -->
# Frozen near/far cache exploration

Status: complete; paired requests 3750/3750.

| arm | AVG16 | near-active requests | far-active requests | frozen KV / full-node KV | attention tokens / full |
|---|---:|---:|---:|---:|---:|
| full | 49.4629 | 0 | 0 | 1.0000 | 1.0000 |
| far_0.05 | 49.3301 | 0 | 2258 | 0.9624 | 0.9644 |
| dual_0.05 | 49.3257 | 105 | 2258 | 0.9624 | 0.9636 |

Per-task scores (0–100):

| arm | narrativeqa | qasper | multifieldqa_en | hotpotqa | 2wikimqa | musique | gov_report | qmsum | multi_news | trec | triviaqa | samsum | passage_count | passage_retrieval_en | lcc | repobench-p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 27.1290 | 48.4861 | 53.8867 | 59.1649 | 43.2877 | 35.9615 | 33.6261 | 23.9801 | 24.8741 | 71.5000 | 90.7119 | 44.6805 | 2.0000 | 100.0000 | 68.1580 | 63.9600 |
| far_0.05 | 26.9033 | 47.8025 | 53.1626 | 59.4149 | 42.7211 | 35.9627 | 33.6405 | 24.1097 | 24.8094 | 71.5000 | 90.7119 | 44.4465 | 2.0000 | 100.0000 | 68.1300 | 63.9660 |
| dual_0.05 | 26.8385 | 47.8025 | 53.1626 | 59.4149 | 42.7211 | 35.9627 | 33.6255 | 24.1097 | 24.7595 | 71.5000 | 90.7119 | 44.5064 | 2.0000 | 100.0000 | 68.1300 | 63.9660 |

Paired differences (percentage points):

- far_0.05-vs-full: -0.1329; bootstrap 95% CI [-0.23543353793211627, -0.0327706415602608]
- dual_0.05-vs-full: -0.1372; bootstrap 95% CI [-0.2429747325561665, -0.033831761865107286]
- dual_0.05-vs-far_0.05: -0.0044; bootstrap 95% CI [-0.013650619548307192, 0.004735721348378953]

Attention baseline: tli-tail32-frozen-regions-reference. Arm full means both region gates off.

Exploratory paired request bootstrap, 5000 repeats, task-stratified macro AVG16; CI containing zero is not proof of equivalence. Near-only retains staging KV and skips attention work only; far-only/both physically evict nodes. TLI tail32 L1/L2 reference adapter; full denotes gates off, not dense attention. No performance claim.

KV ratios measure allocated cache-node capacity, relative to the same full-node cache; they exclude model weights, gather temporaries and allocator behavior. Attention-token ratios are a pre-decode work proxy, not measured speedups.
Near and Far gates compare their own mass independently; adding Near gating does not change Far decisions.
Near-active and far-active request counts include both-skip states.
