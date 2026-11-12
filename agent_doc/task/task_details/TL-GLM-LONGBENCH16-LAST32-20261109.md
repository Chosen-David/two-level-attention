# TL-GLM-LONGBENCH16-LAST32-20261109

## 目标与口径

- 按用户选定的 `glm_nope_last32`（seed=0）运行 GLM-4.7-Flash × LongBench v1 英文 16 任务全量，BS=1，`seed=42`、`context_cap=31500`。
- 闭集为 3750 请求：13 个任务各 200 行、`multifieldqa_en=150`、`lcc=500`、`repobench-p=500`。每请求串行运行 `full`（NFA full-region）与 `native_full`（HF SDPA + DynamicCache），共 7500 次单样本生成。
- 本任务是 **full-region 质量臂**：`thresholds=[]`，47 层全部 keep，near/far/both skip 均为 0；NFA decode 开启 CUDA Graph。它不是 threshold=0.1、真实 near/far skip 的完整 NFA 新口径，两者产物和审计协议不得混用。
- 冻结快照 `/tmp/tli-glm-jssa-prod-20261106`，登记 GIT_HEAD=`6760cc495f554ffa85e8eb6d6e552ce02911f457`，run identity=`e477ffa900372d6adb1ab2c7160e12cb2574db566057a64c2eb266f1430f3087`。

## Progress

### 2026-11-09 至 2026-11-12：六分片全量完成

- rank0/1=pc40 GPU0/1，rank2/3=pc31 GPU0/1，rank4/5=pc32 GPU0/1；采用 manifest 中 seed=42 permutation 后的全局位置 `%6` 确定性互斥分片，每 rank 625 请求。
- 六份 `worker_done` 与 complete heartbeat 均落盘，最终 3750/3750 请求完整；各 Python 自然退出，本轮使用的 GPU 显存释放。未重启旧 random32、旧子空间重测或 Qwen，未向 pc02/pc33 派单。
- 三机原始目录分别通过 tar 收割到独立目录 `~/ak_verify_results/glm_longbench_16en_nope_last32_20261109_merged/`，不与旧 random32 或后续 skip 实验混池。

### 2026-11-12：独立只读重算 PASS，结果上报

审计器重新读取 3750 请求，核验 identity、源码/数据集/输入 SHA、确定性 rank 归属、双臂完整性、生成 token IDs、预测、官方逐样本分数，以及每请求 47 层 `keep=47, near=0, far=0, both=0` 回执。结果为 `status=PASS`、`failures=[]`，六 rank 各 625 请求。

| 数据集 | NFA full-region | native FullKV |
|---|---:|---:|
| narrativeqa | 29.3690 | 27.9430 |
| qasper | 36.5197 | 38.0132 |
| multifieldqa_en | 58.0622 | 57.5857 |
| hotpotqa | 59.2644 | 62.4985 |
| 2wikimqa | 44.8778 | 44.5786 |
| musique | 32.9106 | 35.3598 |
| gov_report | 32.6207 | 33.4201 |
| qmsum | 23.5067 | 24.2964 |
| multi_news | 26.4797 | 26.6471 |
| trec | 78.0000 | 77.0000 |
| triviaqa | 91.9083 | 91.9917 |
| samsum | 42.5192 | 43.1685 |
| passage_count | 5.5000 | 5.0000 |
| passage_retrieval_en | 100.0000 | 100.0000 |
| lcc | 66.5960 | 66.8500 |
| repobench-p | 60.5000 | 63.4360 |
| **macro AVG16** | **49.2896** | **49.8618** |
| **micro AVG3750** | **51.4540** | **52.2038** |

macro 差值（NFA−native）为 **−0.5721 分**，micro 差值为 **−0.7498 分**。这是所述 full-region 闭集上的描述性结果，不作统计等价或 skip 口径外推。

- 精简结果：`agent_doc/results/glm_longbench_16en_nope_last32_20261109/summary.json`。
- 完整审计：同目录 `audit.json`；六份 manifest、六份 done、六份 heartbeat、逐请求 SHA256 清单及复现脚本均已归档。
- 请求清单 SHA256=`a14eee4ed6a5f5983390df1d95a419a8010d585a24d6e67bb8b668161b144b5f`；收割清单 SHA256=`11c73464c02986587d521c2037e9b52f0e071cc760597eec54ab75afa079d74c`；审计 JSON SHA256=`9449efaa204a2197eb7c6c66b26bda8687c94c3977ec379d6a7d9348ab082b35`。
- 原始逐请求 JSON 保留在上述 raw 目录；版本库只保存精简结果、协议、脚本和哈希清单。

## 证据边界

本次是由生产会话执行的独立只读重算，状态继续保持 `pending-independent-verification`，不冒充不同 owner 的最终验收。NFA `wall_seconds` 包含 prefill、decode、47 层审计和逐请求 GraphRunner/capture 等开销；`native_full.wall_seconds=NaN`，因此不能由本批数据计算 NFA/native 的精确耗时比或宣称加速。后续 threshold=0.1 + near/far skip + 子空间 + CUDA Graph 的 BS=1 全量重跑另立任务、另用快照/identity/输出目录和适配的审计门禁。
