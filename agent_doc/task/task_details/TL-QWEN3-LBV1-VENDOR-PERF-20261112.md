# [TL-QWEN3-LBV1-VENDOR-PERF-20261112] Qwen3-8B LongBench v1 三 baseline 同预算性能

Task-ID: TL-QWEN3-LBV1-VENDOR-PERF-20261112
Date: 2026-11-12

## Plan

### 目标与验收

按用户追加指令，在本机 pc33 两张 H20 上补齐 Quest、JSSA、AsyncTLS 的 Qwen3-8B × LongBench v1 英文16任务同预算性能；远端其他机器只跑 LongBench v2 精度，本机双卡专跑性能。最终 token 预算统一1024，AsyncTLS 粗筛128×64；Quest/JSSA 使用已核验的官方算子路径，AsyncTLS 使用论文语义复现。

固定与既有 FullKV/NFA 可比的计时协议，分别报告 kernel/microbenchmark 与完整请求 prefill/decode TPOT/e2e；不得用局部 kernel 收益代替完整收益。每个计时进程独占一张 GPU，clean e2e 与 profiler/nsys 分离，测量前后检查 `nvidia-smi`。新数据在不同 owner 独立验收前保持 `pending-independent-verification`。

### 实现方法

- 复用 `sparse_attn/baseline_runtime.py` 的请求级 host-wall 边界：计时包含 cache/index 工作和完成同步，不是 kernel-only 时间。
- 先盘点现有 LongBench v1 入口及 FullKV/NFA 计时产物，固定样本、prompt、max-new、warmup、重复次数和统计量，再实现最小性能 runner；不直接把 RULER `nfa_vendor_graph_perf_20261111` 当 LongBench v1 证据。
- 两卡可运行不同 arm/分片，但同卡不得与精度生成或其他计时任务并发。微基准和 e2e 排序若反转须同时如实报告。

## Progress

### 进度与证据

- 2026-11-12：任务已登记，三臂语义和官方算子前置核验及真实 Qwen3-8B smoke 由 `TL-QWEN3-LBV2-VENDOR-20261112` 完成。
- 既有 `agent_doc/results/nfa_vendor_graph_perf_20261111/` 是 RULER `niah_multikey_2` Graph 性能矩阵，且其中 Quest/AsyncTLS 是独立 PyTorch Graph 语义复现，不能替代本任务的官方 Quest kernel 与 LongBench v1 完整请求测量。
- 本机双卡在 LongBench v2 三臂 smoke 结束后实测均仅 1 MiB、无 compute process；尚未启动正式 LongBench v1 性能计时。

### 阻塞与下一步

- 先完成三 baseline 代码推送和远端 LongBench v2 精度派单，再按本任务协议在本机双卡启动性能测量。
- 当前没有 LongBench v1 新性能数字，不能预填或推断加速比。
