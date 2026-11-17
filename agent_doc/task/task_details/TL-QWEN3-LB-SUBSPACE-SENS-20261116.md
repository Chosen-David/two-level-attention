# TL-QWEN3-LB-SUBSPACE-SENS-20261116

## 目标
- 代码基线：本地 `main` HEAD `9f3e8057acd528a904aa8883b830d541b389e39f`（= `origin/main`），运行于干净 worktree。
- 模型：`/home/wangyuanshuo02/Qwen3-8B`。
- 数据：`/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench`；远端缺数据节点仅同步 `gov_report.jsonl`。
- 协议：`python -m benchmark.efficiency.benchmark_longbench_quality --method nfa --execution eager --nfa-subspace tail|full --nfa-subspace-dim 16|32|64|128 --nfa-gate off --visible-len 8192`，每任务完整 200 请求，单进程单 GPU。
- 分组：tail 维度 `cmp_ratio=128/dim`；dim128 为 `full`。均 `near_tokens` 默认关闭，不共享 GPU。

## 单元
| cell | task | subspace | dim | machine | gpu | out |
|---|---|---|---|---|---|---|
| hotpotqa_dim16 | hotpotqa | tail | 16 | local | 0 | qwen3_lb_subspace_20261116/out/hotpotqa_dim16 |
| hotpotqa_dim32 | hotpotqa | tail | 32 | local | 1 | qwen3_lb_subspace_20261116/out/hotpotqa_dim32 |
| hotpotqa_dim64 | hotpotqa | tail | 64 | pc40 | 0 | qwen3_lb_subspace_20261116/out/hotpotqa_dim64 |
| hotpotqa_dim128 | hotpotqa | full | 128 | pc40 | 1 | qwen3_lb_subspace_20261116/out/hotpotqa_dim128 |
| gov_report_dim16 | gov_report | tail | 16 | pc02 | 0 | qwen3_lb_subspace_20261116/out/gov_report_dim16 |
| gov_report_dim32 | gov_report | tail | 32 | pc02 | 1 | qwen3_lb_subspace_20261116/out/gov_report_dim32 |
| gov_report_dim64 | gov_report | tail | 64 | pc31 | 0 | qwen3_lb_subspace_20261116/out/gov_report_dim64 |
| gov_report_dim128 | gov_report | full | 128 | pc31 | 1 | qwen3_lb_subspace_20261116/out/gov_report_dim128 |

## 状态
- 2026-11-16 18:07：8 单元全部点火，GPU 占用与日志健康检查通过。
- 原始 `quality_manifest_s0of1.json` 及 `pred/*.jsonl` 均先 pending；独立验收前不采信。

## 结果
- 完成时间：2026-11-17 02:05，8/8 manifest 完整 200/200。
- 官方 scorer 对每份 `pred/*.jsonl` 单独跑 `benchmark/LongBench/eval.py --expect-count 200`，`strict_validated=1`。
- 原始与配对结果目录：`agent_doc/results/qwen3_lb_subspace_sensitivity_20261116/`。
- 相对 FullKV（FlashInfer canonical，hotpotqa 59.42 / gov_report 33.48）：
  - hotpotqa：dim16 59.3171（-0.0991）、dim32 59.4213（+0.0051）、dim64 59.3716（-0.0446）、dim128 59.1240（-0.2922）
  - gov_report：dim16 33.3949（-0.0871）、dim32 33.5313（+0.0494）、dim64 33.5026（+0.0206）、dim128 33.6269（+0.1449）
- 配对 bootstrap CI 见 `sensitivity_paired_vs_fullkv.csv`。
- 状态仍为 `pending-independent-verification`，独立验收通过前不采信为结论。

## RULER MultiKey2 扩展
- 代码基线同上（`9f3e8057`），模型 `/home/wangyuanshuo02/Qwen3-8B`。
- 数据：`/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/{32768,65536,131072}/niah_multikey_2.jsonl`，每档 100 行。
- 协议：`python -m benchmark.efficiency.benchmark_ruler_quality --method nfa --execution eager --nfa-subspace tail|full --nfa-subspace-dim 16|32|64|128 --nfa-gate off --task niah_multikey_2 --context-length <L> --visible-len 8192`；`level2_topk=1024`（K2）、`level1_topk=128`、`near_tokens off`。
- 分布：
  - 32768：dim16/32 @pc40，dim64/128 @pc31
  - 65536：dim16/32/64/128 @pc02
  - 131072：dim16/32/64 @pc02，dim128 拆 s0of2@pc02 / s1of2@pc31
- 完成：2026-11-17 12:45，12 单元全部 `quality_manifest` 100/100；128K dim128 两片索引无重叠且 0..99 无缺失。
- 独立验收：本地拉回后重算 `string_match_all`，每行与 `official_score`、`answers_sha256`/`_answers_sha` 一致；manifest head/level2_topk/level1_topk 复核通过。
- 分数（官方 string_match_all，越高越好）：
  - 32768：dim128 88.0 / dim64 86.0 / dim32 83.0 / dim16 81.0
  - 65536：dim64 83.0 / dim128 80.0 / dim32 80.0 / dim16 59.0
  - 131072：dim128 69.0 / dim32 64.0 / dim64 60.0 / dim16 32.0
- 归档：`agent_doc/results/qwen3_lb_subspace_ruler_multikey2_20261117/`（`summary.json`、`sensitivity_raw.csv`、`raw/`、`independent_verify.py`）。
- 状态：`pending-independent-verification`；独立验收通过前不作为最终结论。

## Full128 性能（L1/L2 算子与 clean TPOT）
- 代码基线同上（`9f3e8057`），`run_nfa_perf_matrix.py` 全四臂：`nfa-operator-profile / nfa-clean-e2e / flashinfer-operator-profile / flashinfer-e2e`，`subspace full`、`l2-subspace full`、Graph、BS1，RULER `niah_multikey_2` sample-index2、扩展输入 32768/65536/131072/262144/524288。
- 五长度全部 `gpu_hygiene=clean`、nsys 证据齐全，`summary.json` 5 records / 0 infeasible。
- 关键数值（ms/step，operator 为 profiled CUDA kernel 均值；clean TPOT 为无 profiler 的中位 TPOT）：
  - 32768：NFA clean 12.5845 / FlashInfer 9.7847；L1 总 1.0907，L2 总 3.8465，算子总 5.9577。
  - 65536：NFA clean 13.1071 / FlashInfer 10.9789；L1 总 1.1605，L2 总 4.3194，算子总 6.5023。
  - 131072：NFA clean 13.3069 / FlashInfer 13.6684；L1 总 1.4071，L2 总 4.3038，算子总 6.7315。
  - 262144：NFA clean 13.5235 / FlashInfer 18.8132；L1 总 1.7599，L2 总 4.2940，算子总 7.0728。
  - 524288：NFA clean 14.1596 / FlashInfer 30.8154；L1 总 2.3934，L2 总 4.2835，算子总 7.6853。
- FlashInfer/NFA clean TPOT 加速比：32K 0.778、64K 0.838、128K 1.027、256K 1.391、512K 2.176；纯 attention kernel 加速比 512K 约 23.28×，selection+attention 路径 512K 约 3.05×。
- 归档：`agent_doc/results/qwen3_lb_subspace_full_perf_20261117/`（`summary.json`、`perf_summary.csv`、四臂 raw JSON/nsys/sqlite）。
- 状态：`pending-independent-verification`。
