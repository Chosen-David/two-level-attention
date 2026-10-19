# 论文 vendor baseline 对比写入蓝本（2026-10-18 deepseek 轨）

用户指令（2026-10-17/18）：四 baseline（JSSA/FASA/Quest/AsyncTLS）接入 → 出数据 → 写论文相关部分对比。
本蓝本汇总已齐数据 + 待补数据，供合入 `sglang/paper/TLI_paper_en.tex`。

## ⚠️ 2026-10-19 AsyncTLS 列重跑（排序修复版）——32K 已替换，LongBench/128K 待替换
- **发现**：AsyncTLS 复现存在「预算拉满≠dense」锚点 FAIL（decode logits diff 11.8→0.55）。
  根因 = Stage-3 `sel_idx` 用 topk 降序索引乱序取值，bfloat16 matmul 对 token 排列敏感，
  36 层累积放大（排序后与 dense **逐位一致 diff=0.0**）。已在 `qwen3_asynctls_patch.py`
  修复（`sel_idx = torch.sort(cand_idx[sel_local])[0]`，数学等价仅稳定数值路径），锚点 PASS。
- **影响**：排序修复改变真实生成序列（L=8192 下 sorted vs shuffled logits diff=0.34，
  8 步生成第 5 token 起分歧）→ **乱序版数据不能进主表，须排序修复版重跑**。
- **已完成替换**：RULER 32K AVG11 **48.33**（2026-10-19 13:09 正式发布
  `e119_ruler32768_formal_asynctls_vendor_fix.json`，ts=10191101/2/3）。
- **进行中**：LongBench 16 任务重跑（.187 GPU1-6 ts=10191104，已 11/16）+ RULER 64K/128K
  重跑规划（本机 128K 乱序版 cwe/fwe 在跑，完成后 .187 GPU0/7 接重跑）。
- 旧乱序版数字（LongBench AVG13 46.51 / AVG16 44.97、RULER 32K AVG11 50.09）仅供参考。

## 预算口径（统一对齐主表 K2=1024）
- Quest: budget 1024（pred_qv_1024_16 批）
- AsyncTLS: token_budget 1024（LongBench）/ 512（RULER 论文口径），block 64、top-kb 128
- FASA: budget 1024、fasa_dataset qasper、t0.8、k16
- JSSA: budget 1024（每 kv head top-B token）、rank 32、校准基 bases/qwen3_8b.pt（32K 非 yarn）

## LongBench 13 任务表（统一 vendor 批，13 = 16 减 passage_count/trec/samsum）
| task | FullKV | Quest | AsyncTLS | FASA | JSSA |
|---|---|---|---|---|---|
| hotpotqa | 59.07 | 45.09 | 49.03 | 12.58 | **60.20** |
| 2wikimqa | 43.29 | 30.78 | 38.31 | 24.73 | **42.34** |
| musique | 35.44 | 23.12 | 20.99 | 4.15 | **34.25** |
| passage_retrieval_en | 100.00 | 88.25 | 95.50 | 60.12 | **100.00** |
| qasper | 47.47 | 36.43 | 45.41 | 34.85 | **48.16** |
| multifieldqa_en | 53.19 | 47.86 | 50.74 | 37.90 | **53.83** |
| gov_report | 33.36 | 32.32 | 32.38 | 28.35 | **33.58** |
| qmsum | 23.92 | 23.20 | 23.37 | 18.80 | **24.24** |
| multi_news | 24.78 | 25.46 | 24.65 | 25.33 | **24.64** |
| narrativeqa | 27.59 | 18.85 | 17.85 | 2.53 | **26.47** |
| triviaqa | 91.11 | 72.73 | 81.91 | 71.36 | **89.82** |
| lcc | 68.10 | 60.23 | 67.13 | 61.26 | **68.37** |
| repobench | 64.74 | 53.63 | 57.39 | 44.84 | **65.54** |
| **AVG13** | **51.70** | **42.92** | **46.51** | **32.83** | **51.65** |

来源：FullKV=`exp/results_longbench_vendor/Qwen3-8B/fullkv/pred/result.json`；Quest=`pred_qv_1024_16/pred/`；
AsyncTLS=`pred_asynctls/`；FASA=`exp/results_longbench_fasa/Qwen3-8B/pred_fasa_1024/pred/`；JSSA=
`exp/results_longbench_vendor/Qwen3-8B/pred_jssa_vendor_qwen3_8b_k1024_r32/result.json`（2026-10-19 03:36，
strict runcap 校验，16/16 任务）。
16 任务 AVG：FullKV 49.39 / Quest 41.35 / AsyncTLS 44.97 / FASA 33.31 / **JSSA 49.28**。
JSSA AVG13=51.65 ≈ FullKV 51.70（triviaqa 89.82 接近 91.11、passage_retrieval_en 100 持平、hotpotqa
60.20 反超 59.07）——**JSSA LongBench 与 dense 几乎持平**，远超 Quest/AsyncTLS/FASA。

## RULER 32K 表（AVG11，n=100）
| task | FullKV | TLI mavg | Quest | AsyncTLS | FASA | JSSA |
|---|---|---|---|---|---|---|
| niah_single_1 | 100.00 | 100.00 | 41.00 | 95.00 | 10.00 | **100.00** |
| niah_single_2 | 100.00 | 100.00 | 96.00 | 89.00 | 73.00 | **100.00** |
| niah_single_3 | 100.00 | 98.00 | 36.00 | 53.00 | 10.00 | **100.00** |
| niah_multikey_1 | 99.00 | 98.00 | 87.00 | 97.00 | 50.00 | **99.00** |
| niah_multikey_2 | 76.00 | 84.00 | 44.00 | 35.00 | 2.00 | **76.00** |
| niah_multikey_3 | 4.00 | 8.00 | 0.00 | 0.00 | 0.00 | **2.00** |
| niah_multiquery | 24.50 | 24.50 | 20.00 | 31.25 | 19.75 | **24.50** |
| niah_multivalue | 25.00 | 28.50 | 32.00 | 43.75 | 23.50 | **25.75** |
| cwe | 83.50 | 71.30 | 44.60 | 31.80 | 8.40 | **85.50** |
| fwe | 7.00 | 11.67 | 18.00 | 16.00 | 3.00 | **2.67** |
| vt | 34.20 | 32.20 | 52.80 | 39.80 | 5.40 | **35.00** |
| **AVG11** | **59.38** | **59.65** | **42.85** | **48.33** | **18.64** | **59.13** |

来源：FullKV/TLI=`e116b_ruler32768k_formal_{fullkv,mavg}.json`；Quest=`e119_ruler32k_formal_quest_vendor.json`；
AsyncTLS=**`e119_ruler32768_formal_asynctls_vendor_fix.json`（2026-10-19 13:09 正式发布，排序修复版重跑批 ts=10191101/2/3，
  11/11 格 100 行，producer 统一。排序修复=保持原始 token 序稳定数值路径，锚点预算拉满==dense 逐位一致；
  AVG11 48.33 vs 旧乱序版 50.09，个别任务 ±1-8 双向差异）**；FASA=`e119_ruler32768_formal_fasa_vendor_10182330.json`
（2026-10-19 08:23 正式发布，完整重跑批 ts=10182330 11/11 格 producer sha=af11de1c 统一、yarn=False 一致，
替换 legacy 合成 18.55 → **18.64**）；JSSA=`e119_ruler32768_formal_jssa_vendor.json`（正式发布 2026-10-18 21:45）。
*FASA 32K 旧 legacy 口径（fasa_32k_full11.json，跨批 6a19808b vs 6bde5310）已由 formal 完整重跑批取代——
  新 5 任务（multiquery 19.75/multivalue 23.50/cwe 8.40/fwe 3.00/vt 5.40）与 legacy 完全一致；旧 6 任务
  单格略有差异（重跑批实际分数：single_1 10/single_2 73/single_3 10/multikey_1 50/multikey_2 2/multikey_3 0）。

## 口径警示（写入必须注明）
- 论文 LongBench 主表 FullKV 50.36 / Quest 47.72 为 **PSI 实验批**，与 vendor 批（51.70/42.92）不同。
  写入时：vendor baseline 作为**独立统一批次**单独报告（同批 FullKV 作内参照），不与 PSI 主表混同。
- RULER 32K Quest 论文主表 32K†=47.63，vendor 批=42.85（批次差异）；TLI mavg 59.99（记忆）vs e116b 59.65 待与 Opus 核对统一口径。

## RULER 64K 表（新增，2026-10-19 04:3X，AVG11 n=100）
| task | AsyncTLS | FASA |
|---|---|---|
| niah_single_1 | 82.00 | 5.00 |
| niah_single_2 | 87.00 | 6.00 |
| niah_single_3 | 9.00 | 0.00 |
| niah_multikey_1 | 79.00 | 2.00 |
| niah_multikey_2 | 12.00 | 0.00 |
| niah_multikey_3 | 0.00 | 0.00 |
| niah_multiquery | 22.50 | 0.25 |
| niah_multivalue | 19.25 | 0.50 |
| cwe | 0.30 | 0.00 |
| fwe | 18.33 | 4.00 |
| vt | 45.80 | 1.20 |
| **AVG11** | **34.11** | **1.72** |

来源：`e119_ruler65536_formal_{asynctls_vendor_64k,fasa_vendor}.json`（2026-10-19 正式发布，--yarn）。
*口径：e116e data_root 已补 65536/131072 synthetic stub（仿 32768 两行格式）；64K/128K 打分需
  `--pred-postfix <唯一postfix>`（避开 L32768 残留 pred_fasa_vendor 跨长度门禁）+ `--yarn`。

## RULER 128K 表（新增，2026-10-19 10:1X，AVG11 n=100）
| task | FASA |
|---|---|
| niah_single_1 | 4.00 |
| niah_single_2 | 0.00 |
| niah_single_3 | 0.00 |
| niah_multikey_1 | 0.00 |
| niah_multikey_2 | 0.00 |
| niah_multikey_3 | 0.00 |
| niah_multiquery | 0.00 |
| niah_multivalue | 0.00 |
| cwe | 0.10 |
| fwe | 7.67 |
| vt | 3.60 |
| **AVG11** | **1.40** |

来源：`e119_ruler131072_formal_fasa_vendor.json`（2026-10-19 10:11 正式发布，--yarn，唯一 postfix
`_fasa128`；8 主批 ts=10181813 + 3 single 补跑 ts=10181852，11/11 格 n=100 全 100 行）。
*FASA 长度趋势：**32K 18.64 → 64K 1.72 → 128K 1.40**（质量崩塌一致）。AsyncTLS 128K 在飞
  （本机 GPU1，multivalue 93/100 在跑，剩 cwe/fwe/vt），全齐后同口径打分补本表列
  （--yarn + 唯一 postfix _asynctls128）。

## 性能对比表（新增，2026-10-19 05:1X，#29 第一批）
脚本 `exp/bench/vendor_e2e_latency.py`（复用 E152 计时框架 timed_request，vendor 挂载分支对齐
pred_ruler.py；合成 needle prompt，max_gen=128，warmup1+repeats2-3，.187 单卡 H20 143GB）。
预算口径对齐主表（Quest 1024 / JSSA k1024 r32 / FASA qasper b1024 / AsyncTLS kt512 b64 kb128）。
**口径警示**：均为 **eager torch 语义等价实现**（非官方 kernel），只表征方法语义的 eager 实现成本。

| method | 32K tpot(ms) | 64K tpot(ms) | 128K tpot(ms) | 128K wall(ms) |
|---|---|---|---|---|
| dense | 26.05 | 39.62 | 66.94 | 84026.6 |
| JSSA | 33.30 (+28%) | 39.68 (+0.2%) | **61.69 (−8%)** | 83297.2 |
| FASA | 75.64 | 133.00 | 265.91 | 108597.7 |
| AsyncTLS | 105.29 | 123.42 | 160.25 | 95778.8 |
| Quest | 177.04 | 317.38 | 621.37 | 154381.5 |

（32K wall：dense 11299.8 / JSSA 12189.8 / FASA 17455.6 / AsyncTLS 21298.3 / Quest 30385.5）
发现：**JSSA 固定预算 1024 稀疏 decode tpot 随 L 增长最缓（32→128K 1.85x vs dense 2.57x），
128K 反超 dense（61.69 < 66.94，−8%）**——与 TLI prefill 长上下文加速故事同构；Quest eager
全维 token gather 开销最大（128K 621ms，约 dense 9x）。

## 待补数据（2026-10-19 05:1X 更新）
1. ~~JSSA RULER 32K~~ → **已完成并正式发布 AVG 59.13**（e119_ruler32768_formal_jssa_vendor.json）。
2. ~~JSSA LongBench 16 任务~~ → **已完成并打分：AVG16=49.28 / AVG13=51.65**（新主机 GPU1 ts=10181840，
   2026-10-19 03:31 16/16 全齐，result.json strict runcap 校验；13 任务表 JSSA 列已填）。
3. ~~FASA RULER 32K 补 5 任务~~ → **全部完成并合成打分 AVG11=18.55**（fasa_32k_full11.json，
   legacy 基础打分非 formal，跨批 producer 门禁说明见上）。
4. ~~FASA/AsyncTLS RULER 64K~~ → **已完成并正式打分**（见上 RULER 64K 表：FASA 1.72 / AsyncTLS 34.11）。
   **RULER 128K 在飞**：FASA 128K single_1/2/3 补跑全完成（ts=10181852，各 100 行已同步本地），
   主批 cwe 81 在跑（GPU1 batch 循环自动接 fwe/vt）；AsyncTLS 128K 本机 multikey_3 49 在跑，
   剩 multiquery/multivalue/cwe/fwe/vt。
5. ~~JSSA/AsyncTLS/FASA 性能（#29 decode/prefill microbench）~~ → **32K/64K 已完成**（见下性能表，
   `exp/trace/results/perf_r{32k,64k}_{method}.json`，脚本 `exp/bench/vendor_e2e_latency.py`）；
   **128K 性能在飞**（.187 五卡 2026-10-19 05:1X，warmup1+repeats2）。
6. RULER 128K 全齐后同口径打分（--yarn + 唯一 postfix _fasa128/_asynctls128）。
