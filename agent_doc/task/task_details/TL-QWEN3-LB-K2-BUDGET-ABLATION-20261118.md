# TL-QWEN3-LB-K2-BUDGET-ABLATION-20261118

## 目标
- 代码基线：本地 `main` HEAD `8736a965e903fd461ef5591347933205e18adee0`（已合入 `origin/main`），工作区仅运行质量入口，不携带源码改动。
- 模型：`/home/wangyuanshuo02/Qwen3-8B`。
- 数据：LongBench `hotpotqa` / `gov_report`，每任务完整 200 请求。
- 协议：`python -m benchmark.efficiency.benchmark_longbench_quality --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 --nfa-gate off --nfa-level2-topk {256,512,1024,2048} --visible-len 8192`；`level1_topk=128`、`page_size=64`、`near_tokens off`。
- 分组：HotpotQA + GovReport × K2 四档，共 8 单元，单进程单 GPU，本机两卡不占用。

## 分布
| cell | task | k2 | machine |
|---|---|---|---|
| hotpotqa_k2_256 | hotpotqa | 256 | pc02 |
| hotpotqa_k2_512 | hotpotqa | 512 | pc40 |
| hotpotqa_k2_1024 | hotpotqa | 1024 | pc40 |
| hotpotqa_k2_2048 | hotpotqa | 2048 | pc32 |
| gov_report_k2_256 | gov_report | 256 | pc32 |
| gov_report_k2_512 | gov_report | 512 | pc40 |
| gov_report_k2_1024 | gov_report | 1024 | pc40 |
| gov_report_k2_2048 | gov_report | 2048 | pc32 |

## 状态
- 2026-11-18：8 单元全部生产完成并收割至本地 `raw/`，每单元 `quality_manifest_s0of1.json` 完整 200/200。
- 独立验收：`independent_verify.py` 逐单元复核 manifest schema/verification/head/source diff、runcap 完整性、逐行输入身份，并调用官方 `benchmark.LongBench.eval --expect-count 200` 重算逐样本分数；官方分数与独立复算均值一致，配对 bootstrap 相对 FlashInfer FullKV 基线完成。结果 `PASS 8 cells`。

## 结果
- 归档：`agent_doc/results/qwen3_lb_k2_ablation_8736a965_20261117/`（`independent_verify.json`、`k2_ablation_summary.csv`、`raw/`、`_dispatch/`）。
- 官方分数（QA-F1 / gov_report-ROUGE-L，200 请求）与相对 FullKV（hotpotqa 59.4162 / gov_report 33.4820）的配对差值：
  - hotpotqa：k256 58.94（-0.48，CI95 [-1.85, 0.60]）、k512 59.38（-0.03，CI95 [-0.88, 0.75]）、k1024 59.12（-0.29，CI95 [-0.90, 0.02]）、k2048 59.12（-0.29，CI95 [-0.90, 0.01]）
  - gov_report：k256 33.45（-0.03，CI95 [-0.48, 0.43]）、k512 33.64（+0.15，CI95 [-0.31, 0.63]）、k1024 33.63（+0.14，CI95 [-0.27, 0.57]）、k2048 33.47（-0.01，CI95 [-0.41, 0.37]）
- 全部 8 个 CI 跨 0，任一 K2 档相对 FullKV 均无统计显著精度差异；hotpotqa 在 k512 最接近 FullKV（-0.03），gov_report 在 k512/k1024 略高于 FullKV（+0.15/+0.14，仍跨 0）。
- 状态：`pending-independent-verification`；独立验收通过前不作为最终结论。
