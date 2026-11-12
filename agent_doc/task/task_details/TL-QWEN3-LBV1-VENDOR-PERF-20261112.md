# [TL-QWEN3-LBV1-VENDOR-PERF-20261112] Qwen3-8B LongBench v1 三 baseline 同预算性能

Task-ID: TL-QWEN3-LBV1-VENDOR-PERF-20261112
Date: 2026-11-12

## Plan

### 目标与验收

按用户追加指令，在本机 pc33 两张 H20 上补齐 Quest、JSSA、AsyncTLS 的 Qwen3-8B × LongBench v1 英文16任务同预算性能；远端其他机器只跑 LongBench v2 精度，本机双卡专跑性能。最终 token 预算统一1024，AsyncTLS 粗筛128×64；Quest/JSSA 使用已核验的官方算子路径，AsyncTLS 使用论文语义复现。

固定与既有 FullKV/NFA 可比的计时协议，分别报告 kernel/microbenchmark 与完整请求 prefill/decode TPOT/e2e；不得用局部 kernel 收益代替完整收益。每个计时进程独占一张 GPU，clean e2e 与 profiler/nsys 分离，测量前后检查 `nvidia-smi`。新数据在不同 owner 独立验收前保持 `pending-independent-verification`。

### 实现方法

- `benchmark/LongBench/perf_vendor.py` 复用 `pred.py` 的真实 prompt、截断、chat、question split 与生产 greedy 循环；`HostWallRequestTimer` 记录 host-wall TTFT、首 token 后 decode wall/TPOT 和完整请求 e2e，包含 cache/index、argmax、逐 token host readback 与边界同步，不是 kernel-only 时间。
- 正式协议固定 B1 eager、英文16任务各 source sample0、每样本1次完整 warmup、3次测量、max-new=33（首 token + 32 decode）；EOS 只记录不早停，effective cap=31500。逐任务取3次 median，再报告 macro-task mean 与 all-request median。该范围是16个确定性请求的性能矩阵，不外推为3750请求分布。
- Quest 固定1024/16/16/40960并调用官方 CUDA；JSSA 固定1024/rank32/page64并调用官方 Triton；AsyncTLS 固定1024、block64、top_kb128及 per-group channel。三臂显式关闭不消费的 NFA layer-skip。
- clean-e2e 与 operator-profile 使用不同输出路径，后者只供外部 nsys 单独运行；两卡可运行不同 arm，但同卡不得与精度生成或其他计时任务并发。微基准和 e2e 排序若反转须同时如实报告。

## Progress

### 进度与证据

- 2026-11-12：任务已登记，三臂语义和官方算子前置核验及真实 Qwen3-8B smoke 由 `TL-QWEN3-LBV2-VENDOR-20261112` 完成。
- 新增 `benchmark/LongBench/perf_vendor.py` 与 `run_longbench_v1_vendor_perf.sh`，并在 `_manual_generate`、`get_pred` 和官方 JSSA 完整生成入口加入可选计时/定长参数；默认 `stop_on_eos=True` 的精度行为不变，samsum 的 JSSA/AsyncTLS 保持完整 prompt 路由。
- 聚焦回归：`tests/test_longbench_empty_handoff.py` + `tests/test_longbench_vendor_perf.py` 共12 passed；`tests/test_qwen3_jssa_official.py` CUDA 4 passed，覆盖真实 EOS 下 `stop_on_eos=False` 仍生成完整3 token。
- 真实 Qwen3-8B 三臂均在 qasper/samsum 完成 warmup+定长3 token smoke，输入身份与计时回执完整；smoke 产物 `/tmp/qwen3_lbv1_perf_smoke_20261112/{quest,jssa,asynctls}.json`，不作为正式性能结果。
- 既有 `agent_doc/results/nfa_vendor_graph_perf_20261111/` 是 RULER `niah_multikey_2` Graph 性能矩阵，且其中 Quest/AsyncTLS 是独立 PyTorch Graph 语义复现，不能替代本任务的官方 Quest kernel 与 LongBench v1 完整请求测量。
- 正式 clean-e2e 已按 GPU0=Quest→AsyncTLS、GPU1=JSSA 串行独占启动，输出 `agent_doc/results/qwen3_lbv1_vendor_perf_20261112T01/`；尚未收官，当前数字不得作为最终结论。

### 阻塞与下一步

- 等三臂48个测量请求全部完成后，核对48/48 cells、输入身份、generated_tokens=33、EOS不早停与哈希，再汇总正式结果；随后只在空闲独占卡上运行代表性外部 nsys profile。
- 所有正式性能数据在不同 owner 独立验收前保持 `pending-independent-verification`，不以 smoke 或 kernel 局部收益推断完整请求加速比。
