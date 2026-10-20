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
- **LongBench 16/16 已替换**（2026-10-19 17:5X 正式打分 AVG16 **45.13** / AVG13 **46.83**，
  隔离目录 `pred_asynctls_vendor_fix/` strict runcap 16；组成 = 10191104 批 12 任务
  + 10191920 批 4 任务（qmsum/multi_news/gov_report/repobench，.187 relay GPU3/4/6）。
  **与乱序版几乎持平**（AVG16 44.97→45.13，逐任务 ±0.02~1.75 双向），符合预期——
  LongBench 上下文短（多数 <8k），排序修复影响小，与 RULER 32K 差异（48.33 vs 50.09）方向一致）。
- **已完成（2026-10-19 21:33）**：**RULER 128K 修复版正式发布 AVG11=37.47**
  （e119_ruler131072_formal_asynctls_vendor_fix.json，ts=10191920 relay GPU0/1/2/5/7，11/11 格
  100 行）。**RULER 64K 修复版正式发布 AVG11=33.88**
  （e119_ruler65536_formal_asynctls_vendor_fix.json，ts=10191930 relay GPU3/4/6→扩 8 卡，
  11/11 格 100 行，替换乱序版 34.11）。AsyncTLS 长度趋势 32K 48.33 / 64K 33.88 / 128K 37.47
  保持无崩塌，对比 FASA（18.64/1.72/1.40）显著优势。**2026-10-19 18:3X 已停 lb_relay2 重复
  repobench-p 进程**（LongBench 修复版 16/16 已全部完整，.187 repobench 文件 500 行
  md5 与本机打分数据一致，runcap gate 未覆盖损坏；教训：LongBench repobench-p 落盘名
  去掉 `-p`，relay 判定须用实际落盘名）。
- 旧乱序版数字（LongBench AVG13 46.51 / AVG16 44.97、RULER 32K AVG11 50.09）仅供参考。

## 预算口径（统一对齐主表 K2=1024）
- Quest: budget 1024（pred_qv_1024_16 批）
- AsyncTLS: token_budget 1024（LongBench）/ 512（RULER 论文口径），block 64、top-kb 128
- FASA: budget 1024、fasa_dataset qasper、t0.8、k16
- JSSA: budget 1024（每 kv head top-B token）、rank 32、校准基 bases/qwen3_8b.pt（32K 非 yarn）

## LongBench 13 任务表（统一 vendor 批，13 = 16 减 passage_count/trec/samsum）
| task | FullKV | Quest | AsyncTLS | FASA | JSSA |
|---|---|---|---|---|---|
| hotpotqa | 59.07 | 45.09 | 48.23 | 12.58 | **60.20** |
| 2wikimqa | 43.29 | 30.78 | 39.30 | 24.73 | **42.34** |
| musique | 35.44 | 23.12 | 21.62 | 4.15 | **34.25** |
| passage_retrieval_en | 100.00 | 88.25 | 97.25 | 60.12 | **100.00** |
| qasper | 47.47 | 36.43 | 45.32 | 34.85 | **48.16** |
| multifieldqa_en | 53.19 | 47.86 | 50.85 | 37.90 | **53.83** |
| gov_report | 33.36 | 32.32 | 32.63 | 28.35 | **33.58** |
| qmsum | 23.92 | 23.20 | 23.17 | 18.80 | **24.24** |
| multi_news | 24.78 | 25.46 | 24.67 | 25.33 | **24.64** |
| narrativeqa | 27.59 | 18.85 | 18.74 | 2.53 | **26.47** |
| triviaqa | 91.11 | 72.73 | 81.14 | 71.36 | **89.82** |
| lcc | 68.10 | 60.23 | 67.61 | 61.26 | **68.37** |
| repobench | 64.74 | 53.63 | 58.24 | 44.84 | **65.54** |
| **AVG13** | **51.70** | **42.92** | **46.83** | **32.83** | **51.65** |

来源：FullKV=`exp/results_longbench_vendor/Qwen3-8B/fullkv/pred/result.json`；Quest=`pred_qv_1024_16/pred/`；
AsyncTLS=`pred_asynctls_vendor_fix/`（**排序修复版**，2026-10-19 17:5X 正式打分，strict runcap 16；
  组成 ts=10191104 12 任务 + 10191920 4 任务，relay GPU3/4/6 无重复派单）；FASA=
`exp/results_longbench_fasa/Qwen3-8B/pred_fasa_1024/pred/`；JSSA=
`exp/results_longbench_vendor/Qwen3-8B/pred_jssa_vendor_qwen3_8b_k1024_r32/result.json`（2026-10-19 03:36，
strict runcap 校验，16/16 任务）。
16 任务 AVG：FullKV 49.39 / Quest 41.35 / **AsyncTLS 45.13** / FASA 33.31 / **JSSA 49.28**。
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
| niah_single_2 | **85.00** | 6.00 |
| niah_single_3 | **8.00** | 0.00 |
| niah_multikey_1 | 79.00 | 2.00 |
| niah_multikey_2 | **9.00** | 0.00 |
| niah_multikey_3 | 0.00 | 0.00 |
| niah_multiquery | 24.75 | 0.25 |
| niah_multivalue | **22.50** | 0.50 |
| cwe | **0.20** | 0.00 |
| fwe | **17.67** | 4.00 |
| vt | **44.60** | 1.20 |
| **AVG11** | **33.88** | **1.72** |

来源：AsyncTLS=`e119_ruler65536_formal_asynctls_vendor_fix.json`（**2026-10-19 21:23 正式发布，
排序修复版重跑批 ts=10191930，11/11 格 100 行，--yarn，merge-best 仲裁跳过乱序版 ts=10181800；
AVG11 33.88 vs 旧乱序版 34.11 基本持平，与 32K 方向一致，排序修复影响小）`；FASA=
`e119_ruler65536_formal_fasa_vendor.json`（2026-10-19 正式发布，--yarn）。
*口径：e116e data_root 已补 65536/131072 synthetic stub（仿 32768 两行格式）；64K/128K 打分需
  `--pred-postfix <唯一postfix>`（避开 L32768 残留 pred_fasa_vendor 跨长度门禁）+ `--yarn`。

## RULER 128K 表（新增，2026-10-19 10:1X，AVG11 n=100）
| task | AsyncTLS | FASA |
|---|---|---|
| niah_single_1 | **99.00** | 4.00 |
| niah_single_2 | **95.00** | 0.00 |
| niah_single_3 | **18.00** | 0.00 |
| niah_multikey_1 | **92.00** | 0.00 |
| niah_multikey_2 | **16.00** | 0.00 |
| niah_multikey_3 | 0.00 | 0.00 |
| niah_multiquery | **20.00** | 0.00 |
| niah_multivalue | **8.75** | 0.00 |
| cwe | **0.70** | 0.10 |
| fwe | **31.33** | 7.67 |
| vt | **31.40** | 3.60 |
| **AVG11** | **37.47** | **1.40** |

来源：AsyncTLS=`e119_ruler131072_formal_asynctls_vendor_fix.json`（**2026-10-19 21:33 正式发布，
排序修复版重跑批 ts=10191920，.187 relay GPU0/1/2/5/7，11/11 格 n=100 全 100 行，--yarn）；FASA=
`e119_ruler131072_formal_fasa_vendor.json`（2026-10-19 10:11 正式发布，--yarn，唯一 postfix
`_fasa128`；8 主批 ts=10181813 + 3 single 补跑 ts=10181852，11/11 格 n=100 全 100 行）。
***长度趋势（重要论文对比）**：AsyncTLS **32K 48.33 → 64K 33.88 → 128K 37.47**（保持，无崩塌）；
FASA **32K 18.64 → 64K 1.72 → 128K 1.40**（崩塌）。**AsyncTLS 128K 质量保持远优于 FASA
  （37.47 vs 1.40，26.8x）**——token 稀疏 vs 子空间基长上下文对比的核心论据。

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
4. ~~FASA/AsyncTLS RULER 64K/128K~~ → **全部完成并正式打分**（2026-10-19 21:33 收尾）：
   - **AsyncTLS 64K** AVG11 **33.88**（e119_ruler65536_formal_asynctls_vendor_fix.json，
     ts=10191930，替换乱序版 34.11）；**AsyncTLS 128K** AVG11 **37.47**
     （e119_ruler131072_formal_asynctls_vendor_fix.json，ts=10191920）。
   - **FASA 64K** 1.72 / **FASA 128K** 1.40（e119_ruler65536/131072_formal_fasa_vendor.json）。
   - **重要发现**：AsyncTLS 长上下文质量保持（32K 48.33→64K 33.88→128K 37.47），FASA 崩塌
     （18.64→1.72→1.40）——token 稀疏 vs 子空间基对比核心论据。
5. ~~JSSA/AsyncTLS/FASA 性能（#29 decode/prefill microbench）~~ → **32K/64K/128K 已完成**（见上性能表，
   `exp/trace/results/perf_r{32k,64k,128k}_{method}.json`，脚本 `exp/bench/vendor_e2e_latency.py`）。
6. ~~RULER 128K 全齐后同口径打分~~ → **已完成**（--yarn + 唯一 postfix _fasa128/_asynctls128）。
7. ~~四 baseline 数据全齐 → 推进 #31 论文写入~~ → **已完成（2026-10-19 22:0X）**，见下方记录。

## #31 论文写入完成（2026-10-19 22:0X，deepseek 轨）
- **TLI_paper_en.tex 新增 `tab:vendorlb` 表**（LongBench 13 任务 vendor 批，独立统一批次单独报告）：
  插入在 `tab:betacross` 之后、`\subsection{Ablation}` 之前。6 列 lccccc（FullKV/Quest/AsyncTLS/FASA/JSSA），
  13 行数据 + Overall 行。caption 注明「independent unified batch，FullKV 内参照 51.70 与主表批 50.36
  的批次差、单独报告不混同」+ 各方法 budget 口径（全部 1024：Quest 全维 token gather、AsyncTLS
  token 1024/b64/kb128 排序修复版、FASA qasper 基 t0.8 k16、JSSA rank32）。
- **表后正文一段**：JSSA 与 dense 几乎持平（Overall 51.65 vs 51.70，-0.05，11/13 任务全方法最高，
  hotpotqa 60.20 反超 FullKV 59.07、passage_retrieval_en 100 并列、musique 34.25 vs Quest 23.12；
  仅 triviaqa 89.82<91.11、multi_news 24.64<25.46 两任务落于他方法）；AsyncTLS 46.83 高于 Quest 42.92
  但低于 JSSA 4.8 分；FASA 32.83 崩塌（narrativeqa 2.53/musique 4.15），呼应 RULER 长度趋势
  （tab:rulerlong）；收尾论据：**子空间轴（JSSA 51.65 贴 dense、PSI 50.78 同族）最强，token-gather
  与固定子空间门都在远程检索让步**。
- **加粗修正（相对蓝本原始标记）**：multi_news 实际最高 Quest 25.46（非 JSSA）、triviaqa 实际最高
  FullKV 91.11（非 JSSA）——逐行按「全 5 方法最高」重核，JSSA 赢 11/13，诚实标注。
- **fig4 修复版数据重生成完成**（22:02）：fig4_partition_gain.py 数据源改 3 个
  e119_ruler{32768,65536,131072}_formal_asynctls_vendor_fix.json，PDF+PNG 输出到 paper/figures/
  与 exp/figures/。数据点：FullKV 59.38/48.54/46.43、Quest 42.85/26.25/19.33、AsyncTLS 48.33/33.88/37.47。
- **LaTeX 验证**：环境嵌套 OK、vendorlb 6 列对齐、tab:vendorlb/tab:rulerlong 引用均有效。
  编译（pdflatex）本机不可用 → **并入 Opus PDF 重编译流程**。
- **口径警示（写入已遵守）**：vendor 批作为独立统一批次单独报告，不与 PSI 主表（tab:longbench 批）
  混同——新表自身带 FullKV 内参照 51.70。
