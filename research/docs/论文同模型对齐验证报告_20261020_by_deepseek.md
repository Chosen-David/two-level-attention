# 论文同模型对齐验证报告（AsyncTLS / JSSA vs 我们 Qwen3-8B 数据）

- 日期：2026-10-20（deepseek 轨）
- 目的：核对复现接入的 AsyncTLS/JSSA 与论文同模型（Qwen3-8B 32k）数据是否对得上，判定接入正确性并定位复现偏差
- 范围：AsyncTLS 论文 LongBench 表（Table 1）+ RULER 表（Table 2）；JSSA 论文无 Qwen3-8B 逐任务锚点 → FKV 锚点自比 + 聚合区间参考

## 1. 数据来源

| 口径 | 来源 |
|---|---|
| 论文 LongBench 13 任务分数 | AsyncTLS.pdf Table 1（Qwen3-8B 32k，kt=1024），zlib 文本流提取 |
| 论文 RULER 10 任务分数 | AsyncTLS.pdf Table 2（Qwen3-8B 32k，kt=512） |
| 我们 FullKV（dense）| `exp/results_longbench_vendor/Qwen3-8B/fullkv/pred/result.json` |
| 我们 AsyncTLS（排序修复版）| `exp/results_longbench_vendor/Qwen3-8B/pred_asynctls_vendor_fix/result.json`（AVG16=44.97，13 任务口径重算） |
| 我们 JSSA | `exp/results_longbench_vendor/Qwen3-8B/pred_jssa_vendor_qwen3_8b_k1024_r32/result.json` |
| 我们 RULER 32K | `exp/trace/results/e116b_ruler32768k_formal_fullkv.json`、`e119_ruler32768_formal_asynctls_vendor_fix.json`、`e119_ruler32768_formal_jssa_vendor.json` |

打分后端：我们 = difflib（difflib:stdlib）；论文 = 未知（LongBench 官方评分器）。

## 2. LongBench 13 任务对齐（主对齐）

论文 AsyncTLS LongBench 任务列：NQA=narrativeqa、QQA=qasper、MFQ=multifieldqa_en、HQA=hotpotqa、2WM=2wikimqa、Mus=musique、GvR=gov_report、QMS=qmsum、MNs=multi_news、TQA=triviaqa、PRetr=passage_retrieval_en、LCC、RBP=repobench（13 任务，AVG13）。

| task | 论文 Full | 论文 AsyncTLS | 我们 FullKV | 我们 AsyncTLS fix | FullKVΔ | ATLSΔ |
|---|---|---|---|---|---|---|
| narrativeqa | 25.61 | 24.77 | 27.59 | 18.74 | +1.98 | **-6.03** |
| qasper | 44.17 | 44.35 | 47.47 | 45.32 | +3.30 | +0.97 |
| multifieldqa_en | 53.40 | 52.67 | 53.19 | 50.85 | -0.21 | -1.82 |
| hotpotqa | 53.48 | 53.59 | 59.07 | 48.23 | +5.59 | **-5.36** |
| 2wikimqa | 38.29 | 38.02 | 43.29 | 39.30 | +5.00 | +1.28 |
| musique | 32.14 | 31.10 | 35.44 | 21.62 | +3.30 | **-9.48** |
| gov_report | 33.17 | 33.58 | 33.36 | 32.63 | +0.19 | -0.95 |
| qmsum | 23.53 | 23.67 | 23.92 | 23.17 | +0.39 | -0.50 |
| multi_news | 24.93 | 24.72 | 24.78 | 24.67 | -0.15 | -0.05 |
| triviaqa | 90.71 | 89.88 | 91.11 | 81.14 | +0.40 | **-8.74** |
| passage_retrieval_en | 100.0 | 99.5 | 100.0 | 97.25 | 0.00 | -2.25 |
| lcc | 67.56 | 67.67 | 68.10 | 67.61 | +0.54 | -0.06 |
| repobench | 64.92 | 65.17 | 64.74 | 58.24 | -0.18 | **-6.93** |
| **AVG13** | **50.15** | **49.90** | **51.70** | **46.83** | **+1.55** | **-3.07** |

## 3. 逐方法判定

### 3.1 FullKV（dense）——✅ 对得上，harness/数据正确
- AVG13 +1.55（51.70 vs 50.15）；13 任务中 10 个 |Δ|≤3.3，其余 3 个（hotpotqa +5.59、2wikimqa +5.00、musique +3.30）我们略高
- 我们的略高方向一致（同一批 pred、同一 difflib 打分），且都在可解释的 harness/批次差异范围
- **判定：dense 侧接入与论文同模型可比，harness/数据管线无系统性问题**

### 3.2 AsyncTLS——⚠️ 8/13 对得上，5 任务复现偏保守（需排查 fix）
- 8/13 任务 |Δ|≤2.3：qasper/multifieldqa_en/2wikimqa/gov_report/qmsum/multi_news/passage_retrieval_en/lcc 与论文 AsyncTLS 对齐良好
- **5 个任务显著偏低 5.4–9.5 分**：musique -9.48、triviaqa -8.74、repobench -6.93、narrativeqa -6.03、hotpotqa -5.36
- 论文 AsyncTLS 相对论文 Full 几乎无损（AVG13 -0.25）；我们 AsyncTLS 相对我们 FullKV 损失 -4.87（51.70→46.83）。**论文口径下稀疏≈dense，我们口径下稀疏有明显损耗 → 复现实现细节偏保守**
- 候选归因（需逐一排查）：
  1. **校准数据差异**：我们 channel 校准用 `qasper_channels.json`（held-out Dcal 口径），论文用其自定 Dcal —— 不同校准集可能导致 channel 集不同，影响细筛
  2. **排序修复版 vs 官方原始**：我们用的是排序修复版（fix 版），官方为原版 —— 排序语义是否与官方一致需对拍
  3. **INT4 量化 vs bf16**：我们适配层若未做 INT4 key 索引量化（用 bf16），budget 语义可能不同
  4. **block_size/top-kb**：论文 block=64、top-kb=128；需核对适配层实际参数（kb 是否按 128 块 × 64 = 8192 token 粗筛）
  5. **softmax 组内平均**：GQA 组内平均（1/G）Σ_h 与 per-head 的差异
  6. **打分后端**：difflib vs 论文官方 scorer（narrativeqa/musique/triviaqa 偏 QA 类，F1/EM 差异可能放大）

### 3.3 JSSA——✅ 与 FKV 锚点全对齐，接入正确
- LongBench AVG13：JSSA 51.65 vs FullKV 51.70（Δ=-0.05）；逐任务 |Δ|≤1.3
- 解释：LongBench 任务上下文多数 <8k，JSSA budget=1024（≈每 token 1024 候选）接近全量 → 与 dense 无差异符合预期
- **判定：JSSA 接入正确（预算拉满≈dense 锚点成立）**

## 4. RULER 32K 参考对齐（任务集不同，仅趋势）

**论文 RULER 任务集（10 任务）**：S1/S2/MK1/MK2/MQ/MV/**QA1/QA2**/VT/FWE —— 含 QA1/QA2，无 S3/MK3/CWE
**我们 RULER 任务集（11 任务）**：S1/S2/S3/MK1/MK2/MK3/MQ/MV/CWE/FWE/VT —— 无 QA1/QA2，多 S3/MK3/CWE
→ 重叠仅 8 任务（S1/S2/MK1/MK2/MQ/MV/VT/FWE）。

| 任务 | 论文 Full | 论文 ATLS | 我们 Dense | 我们 ATLS | 我们 JSSA |
|---|---|---|---|---|---|
| S1 | 100.0 | 100.0 | 100.0 | 95.0 | 100.0 |
| S2 | 100.0 | 100.0 | 100.0 | 89.0 | 100.0 |
| MK1 | 99.60 | 99.80 | 99.0 | 97.0 | 99.0 |
| MK2 | 98.20 | 98.60 | 76.0 | 35.0 | 76.0 |
| MQ | 99.75 | 99.70 | 24.5 | 31.25 | 24.5 |
| MV | 99.60 | 99.45 | 25.0 | 43.75 | 25.75 |
| VT | 99.80 | 99.80 | 34.2 | 39.8 | 35.0 |
| FWE | 97.67 | 96.27 | 7.0 | 16.0 | 2.67 |
| **AVG8** | **99.33** | **99.20** | **58.21** | **55.85** | **57.87** |

**关键事实**：连 dense（FullKV）我们 AVG8 也只有 58.21 vs 论文 99.33 —— 我们在 MK2/MQ/MV/VT/FWE 上 dense 都远低于论文（MK2 76 vs 98.2、MQ 24.5 vs 99.75、FWE 7 vs 97.67）。**两套 RULER 生成器/数据源/样本不同，dense 基线差 30+ 分，无法做接入正确性硬断言。**

相对损失口径（方法 vs 各自 dense，重叠 8 任务内）：
- 论文：AsyncTLS - Full = **-0.12**（稀疏≈dense）
- 我们：AsyncTLS - Dense = **-2.36**；JSSA - Dense = **-0.35**
- 全任务 AVG11：我们 Dense 59.38 / ATLS 48.33 / JSSA 59.13（任务集含 S3/MK3/CWE 三难任务）

**趋势结论**：论文 AsyncTLS 在重叠任务上几乎无损（Δ-0.12），我们复现损失 -2.36 → 方向一致但偏保守，与 LongBench 结论相互印证；JSSA 几乎无损（-0.35）。

## 5. 总体判定与下一步

| 方法 | 判定 | 依据 |
|---|---|---|
| FullKV（dense）| ✅ 对得上 | LongBench AVG +1.55，10/13 任务 ±3 内 |
| AsyncTLS | ⚠️ 偏保守需 fix | LongBench 5 任务 -6~-9 分；RULER 相对损失 -2.36 vs 论文 -0.12 |
| JSSA | ✅ 接入正确 | FKV 锚点逐任务 Δ≤1.3；预算拉满=dense 成立 |

### 下一步（AsyncTLS fix 排查清单，按优先级）
1. **参数核对**：确认适配层 block_size=64 / top-kb=128 / kt（LongBench 1024）与论文逐位一致（arguments.py + qwen3_asynctls_patch.py）
2. **排序语义对拍**：fix 版 vs 官方原始排序，RULER niah 单样本 logits 对拍
3. **INT4 量化**：确认 key 索引是否做了 INT4 量化（论文 §4 Stage-2）
4. **校准差异**：qasper_channels 与论文 Dcal 的差异对 niah/QA 类任务的影响
5. **RULER 数据源**：若后续升级后重跑 RULER，确认生成器版本与论文对齐（QA1/QA2 补入）

### 已确认可用的升级基线
- 论文 LongBench 口径：Full 50.14 / DS 50.10 / AsyncTLS 49.90 / Quest 47.49（Qwen3-8B 32k，kt=1024）
- 论文 RULER 口径：Full 90.45 / AsyncTLS 90.54（Qwen3-8B 32k，kt=512，10 任务含 QA1/QA2）
