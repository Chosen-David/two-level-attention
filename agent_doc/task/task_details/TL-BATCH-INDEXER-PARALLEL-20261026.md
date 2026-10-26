# TL-BATCH-INDEXER-PARALLEL-20261026

用户要求：将逐 bs prepare_mask 改为接收完整 batch 的 indexer，并沿用 PR #7 交付。
基线：PR head 9dbc6e9；未修改 main，未合并 PR。

## 已实施

- `TLIIndexer.prepare_mask_batched`：batch 维贯穿增量页/新 token 维护、L1/L2、mask。
- Qwen staged eager / Graph 两条 B>1 路由；长度元数据 eager 跨层复用，Graph 设备 buffer。
- 可选 Triton `(tile, B*Hkv)` 打分，候选 gather 融合；Torch oracle 保留。
- 旧路径默认不变；显式 `TLI_BATCH_INDEXER=parallel` / 可选 `TLI_BATCH_SCORE=triton`。
- 对照工具分别比较旧 eager 全量重建与新路径、同算法串行与批量；记录源码、输出和 profiler。
- 详细契约与命令：`docs/tli_batch_indexer.md`。

## 验证范围

本地 CPU 回归、Triton CPU interpreter、SM80/90 离线编译完成，原始记录及源码摘要在
`agent_doc/results/tli_batch_parallel_20261026/`。结果仅限 CPU/编译器检查，没有 GPU 实测。
不是独立 reviewer 验收，也没有替代 E127 / batch CUDA 门。

## 待完成

目标 GPU CUDA correctness / BF16 / changing-length Graph replay；旧 Graph vs 新 Graph；
B=2/4/8 × 32K/64K/128K 性能与显存（按硬件容量）；完整模型生成与 RULER 质量。
CPU 更快不能判定远端 eager 回退已解决。本任务保持 open。

## 限制

固定显式 near 几何、free K2、静态子空间、D=128、无投影/动态门/async。
当前批量路径保留完整 KV；strict far compact eviction 仍 B=1，不能将本次改动说成批量 KV 淘汰。
固定槽 softmax/topk 不承诺所有 tie / shape 与旧 eager 位级一致；短序列 padding 和重复 bypass 做安全处理。
跨请求 refill / reorder 必须 clear + Graph recapture，不复用旧图。
