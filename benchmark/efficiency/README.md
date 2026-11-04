# NFA decode 性能验收口径

## 目标与范围

使用真实 Qwen3-8B 与 RULER 长上下文，对比 **FlashInfer Dense** 和 **NFA Sparse 动态 Skip** 的完整 decode 性能。NFA 的 near/far skip 只缩小 selector、`gather_indices` 与 sparse gather attention 的逻辑可见集合；所有 KV 物理页在请求生命周期内连续分配并常驻，不做 prefill/decode Evict & Reclaim。

性能功能只在以下 canonical 入口中 in-place 更新，不新增一次性 benchmark：

- `benchmark_nfa_end_to_end.py`：单配置真实模型 runner。
- `benchmark_nfa_vs_fa2.py`：独立 subprocess、配对顺序与结果汇总 driver。

## 完整 decode 主表

测试矩阵：

- 上下文：32K、64K、128K，默认使用原生 RULER `niah_multikey_1` sample 0 与 `gate-epsilon=0.05` 自然动态 skip。
- Batch：1、2、4；同一 prompt 复制到各 batch 行，仅用于性能测量。
- 执行：eager、CUDA Graph，分别判决。
- 方法：FlashInfer Dense、NFA Sparse。
- 生成长度：512 token，即 511 个 decode step；保留 warmup 与 repeat。

主表只报告：

| 方法 | 上下文 | BS | 模式 | TPOT (ms) | 吞吐量 (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FlashInfer Dense | 32K / 64K / 128K | 1 / 2 / 4 | Eager / Graph | 实测 | 实测 |
| NFA Sparse | 32K / 64K / 128K | 1 / 2 / 4 | Eager / Graph | 实测 | 实测 |

`TPOT` 使用 steady median；每行生成吞吐量按 `1000 / TPOT` 计算。若报告整批聚合吞吐，必须另标为 `batch_throughput = 1000 * BS / TPOT`，不得与每行吞吐混用。

计时边界覆盖完整 decode step：输入与位置更新、scheduler/cache planning、KV write、NFA L1/L2/union 或 FlashInfer paged metadata、模型全部层、sparse/dense attention 与 argmax。禁止使用 replay-only、attention-only closure、post-selection SDPA 或每步全量 KV 布局复制代替生产路径。

## 纯 CUDA kernel 补充表

BS=1，分别测 eager/Graph 的真实 device kernel time；Graph 为主表，eager 为补充。该表只用于定位算子瓶颈，不能替代完整 decode TPOT。

| 原始 KV 长度 | L1 (ms) | L2 (ms) | main Attention (ms) | NFA 算子总耗时 (ms) | FlashInfer Dense (ms) | FlashInfer / NFA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 32K | 实测 | 实测 | 实测 | L1+L2+Attention | 实测 | 实测 |
| 64K | 实测 | 实测 | 实测 | L1+L2+Attention | 实测 | 实测 |
| 96K | 实测 | 实测 | 实测 | L1+L2+Attention | 实测 | 实测 |
| 128K | 实测 | 实测 | 实测 | L1+L2+Attention | 实测 | 实测 |

32K/64K/128K 使用原生 RULER；96K 使用 `repeated-needle-body-extension` 并保留 provenance。纯 kernel 采用 3 warmup + 5 repeat，只聚合 CUDA device time，不含 host、其他模型层或 Graph replay 间隙。

## 测量隔离与正确性

- 每个 arm 使用独立 subprocess，正式对照采用 ABBA 或镜像配对顺序。
- 起跑前目标 GPU used <500 MiB 且无其他计算进程；采数期间每 30 秒旁路监控，污染臂标记 `polluted-rerun` 并重跑。
- latency、profile、traffic、correctness 分进程测量；latency 热循环不做同步、即时 event 读取或 profiler。
- 优化前后 NFA 在相同输入和 execution 下逐 token 一致，并检查 NFA eager/Graph 一致性。NFA 与 Dense 的 token 分叉作为稀疏近似结果单独报告。
- Graph capture 必须覆盖真实 L1、L2、final union、KV write 与 main Attention；自然 skip 样本用于验证 skip 消费链，`gate-epsilon=0` 只作为手动 no-skip 对照。

## 正式结果与重画

2026-11-04 的完整 18 配置 e2e、32K no-skip 对照、32K/64K/96K/128K 纯算子 ABBA 原始臂及最终图归档于 `results/nfa_flashinfer_20261104/`。图表数值全部由归档 JSON 实时计算：

```bash
python benchmark/efficiency/results/nfa_flashinfer_20261104/plot_performance.py --mode all
```

输出位于该目录的 `figures/`，包括独立 PNG（300 dpi）、矢量 PDF 和 `stats.json`。96K 仅为扩展输入，不与原生 RULER 混称；跨方法 token 分叉与自然 skip 层均保留在 receipt 中。
