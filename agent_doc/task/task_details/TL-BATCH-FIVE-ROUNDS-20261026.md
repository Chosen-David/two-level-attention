# TL-BATCH-FIVE-ROUNDS-20261026

用户要求：继续寻找性能浪费并完成五轮迭代，更新同一个 PR #7。
基线：d317956；既有 B=1 compact 与 full-batch 候选边界不变。

五轮实现和本地对拍均完成，独立 source snapshots、逐轮日志和配对结果见
`agent_doc/results/tli_batch_five_rounds_20261026/`。
具体保留项、否决项、运行命令和限制见 `docs/tli_batch_five_rounds.md`。

新增 Triton movement fusions 需 `TLI_BATCH_FUSIONS=1`，默认 0；必须配合
`TLI_BATCH_SCORE=triton`，在 capture 外预热。旧默认路由不变。

本任务未标完成的原因：目标 GPU 测量与质量验收没有执行条件。
解释器、CPU timing 和 offline cubin 均不能替代 CUDA runtime、完整模型
TPOT/throughput/memory、E127/batch gates、RULER/生成词元质量。
尚无独立 reviewer 验收；不把五轮代码迭代写成五轮 GPU 提速。
