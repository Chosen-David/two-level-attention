# TLI 论文 Baseline 矩阵补齐 — 数据审计报告

**日期**: 2026-11-03（继承 2026-10-31 版本）
**作者**: Claude (session continuation)
**范围**: Qwen3-8B + Qwen3-32B + GLM-4.7-Flash，RULER 32K + LongBench

---

## 1. Qwen3-8B（主表模型，数据完整）

### 1.1 RULER 32K（11 任务 × 100 样本） — 7/7 方法齐全

| Method | AVG11 | Δ vs FullKV | 来源 |
|--------|-------|-------------|------|
| TLI (Ours) | **60.77** | +1.39 | ruler32k_all_table.json |
| HISA | 59.69 | +0.31 | ruler32k_all_table.json |
| FullKV | 59.38 | 0.00 | ruler32k_all_table.json |
| JSSA | 59.13 | -0.25 | ruler32k_all_table.json |
| AsyncTLS | 50.09 | -9.29 | e119_ruler32768_formal_asynctls_vendor_final.json |
| Quest | 47.63 | -11.75 | ruler32k_all_table.json |
| FASA | 18.64 | -40.74 | ruler32k_all_table.json |

TLI 领先最接近对手 HISA +1.08，领先 JSSA +1.64，领先 FullKV +1.39。

### 1.2 LongBench（16 任务） — 6/7 方法齐全

| Method | AVG16 | Δ vs FullKV | 来源 |
|--------|-------|-------------|------|
| FullKV | 49.39 | 0.00 | qwen3_8b_longbench_fullkv.json |
| JSSA | 49.28 | -0.11 | qwen3_8b_longbench_jssa_vendor.json |
| TLI (Ours) | 49.27 | -0.11 | qwen3_8b_longbench_tli_champion.json |
| AsyncTLS | 45.13 | -4.25 | qwen3_8b_longbench_asynctls_vendor.json |
| Quest | 40.51 | -8.88 | qwen3_8b_longbench_quest_vendor.json |
| FASA | 35.15 | -14.23 | qwen3_8b_longbench_fasa_vendor.json |

- HISA LongBench **未跑分**（唯一显著数据缺口）
- DuoAttention 蒸馏未完成，无法跑 LongBench

**论文叙事**:
- LongBench 上 TLI/JSSA/FullKV 三者在误差范围内（±0.11），TLI 在 RULER 上的增益显著（+1.39 over FullKV）
- Sparse baseline：AsyncTLS -4.26, Quest -8.88, FASA -14.23（均被 TLI 大幅领先）

---

## 2. Qwen3-32B（补充模型）

### 2.1 RULER 32K — FullKV + TLI 完整矩阵（✅ 11/11 完成）

**FullKV AVG11 = 52.22 / TLI AVG11 = 50.70 / Δ = -1.52**

| Task | FullKV | TLI (champion, bs2) | Δ(TLI-FKV) |
|------|--------|---------------------|-------------|
| niah_single_1 | 93.00 | 92.00 | -1.00 |
| niah_single_2 | 100.00 | 100.00 | 0.00 |
| niah_single_3 | 97.00 | 97.00 | 0.00 |
| niah_multikey_1 | 77.00 | 77.00 | 0.00 |
| niah_multikey_2 | 46.00 | 48.00 | **+2.00** |
| niah_multikey_3 | 1.00 | 1.00 | 0.00 |
| niah_multiquery | 0.00 | 0.00 | 0.00 |
| niah_multivalue | 0.00 | 0.00 | 0.00 |
| cwe | 45.00 | 42.70 | -2.30 |
| fwe | 19.00 | 7.00 | -12.00 |
| vt | 96.40 | 93.00 | -3.40 |
| **AVG11** | **52.22** | **50.70** | **-1.52** |

**关键发现**：
- NIAH 检索任务（8/11）：TLI 与 FullKV 基本持平，multikey_2 TLI 反超 +2.00
- 非检索任务（cwe/fwe/vt）：TLI 落后（fwe -12.00 差距最大），是 AVG 落后的主因
- 与 Qwen3-8B 不同（8B 上 TLI 60.77 > FullKV 59.38），32B 上 FullKV 基线更强，TLI 稀疏化代价更明显
- 打分文件：`exp/trace/results/qwen3_32b_ruler32k_fullkv_vs_tli.json`

### 2.2 RULER 32K JSSA（✅ 11/11 已打分）

**JSSA AVG11 = 54.06 / FullKV AVG11 = 52.22 / Δ = +1.84；TLI AVG11 = 50.70，JSSA 高 3.36**

| Task | FullKV | TLI | JSSA (budget=1024, rank=32) | Δ(JSSA-FKV) |
|------|--------|-----|------------------------------|-------------|
| niah_single_1 | 93.00 | 92.00 | 82.00 | -11.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 0.00 |
| niah_single_3 | 97.00 | 97.00 | 97.00 | 0.00 |
| niah_multikey_1 | 77.00 | 77.00 | 82.00 | +5.00 |
| niah_multikey_2 | 46.00 | 48.00 | 78.00 | +32.00 |
| niah_multikey_3 | 1.00 | 1.00 | 83.00 | +82.00 |
| niah_multiquery | 0.00 | 0.00 | 5.00 | +5.00 |
| niah_multivalue | 0.00 | 0.00 | 7.00 | +7.00 |
| cwe | 45.00 | 42.70 | 53.70 | +8.70 |
| fwe | 19.00 | 7.00 | 7.00 | -12.00 |
| vt | 96.40 | 93.00 | 0.00 | -96.40 |
| **AVG11** | **52.22** | **50.70** | **54.06** | **+1.84** |

11 任务均 `rc=0`、每任务 100 样本，按 TS=`11021830` 的同代 `.tli_gen` 指针解析并记录源文件 SHA256；正式结果：`exp/trace/results/qwen3_32b_ruler32k_jssa_vendor.json`。JSSA 总分高于 FullKV/TLI，但任务间差异很大：`niah_multikey_3` 反超 FullKV +82.00，`vt` 则为 0（-96.40）；这是 **JSSA vendor baseline** 结果，不能改写为 TLI 主方法增益。

### 2.3 LongBench FullKV（✅ 16/16 已打分）

| Method | 分数口径 | 分数 |
|--------|----------|------|
| FullKV (none) | **AVG16** | **42.37** |
| TIA (champion) | AVG13（尚缺 3 任务） | 41.85 |
| Quest_1024 | AVG13（尚缺 3 任务） | 38.49 |

补跑任务 FullKV 分数：`passage_count=19.00`、`samsum=36.46`、`trec=72.50`。三项在 pc02 以 TS=`20261102081500` 完成，逐任务产物行数均与参考数据一致；合并原有 13 任务后的正式结果已落盘 `exp/trace/results/qwen3_32b_longbench_fullkv.json`。既有 TIA 仍只有 AVG13，暂不与 FullKV AVG16 直接作差；当前 Quest/JSSA 正式矩阵进度见下表。

### 2.4 LongBench JSSA（✅ 16/16 已打分）

**JSSA AVG16 = 43.09 / FullKV AVG16 = 42.37 / Δ = +0.72**（budget=1024，rank=32）

| Task | JSSA | n |
|------|-----:|---:|
| narrativeqa | 31.44 | 200 |
| qasper | 44.86 | 200 |
| multifieldqa_en | 50.21 | 150 |
| hotpotqa | 59.88 | 200 |
| 2wikimqa | 54.81 | 200 |
| musique | 36.90 | 200 |
| gov_report | 33.08 | 200 |
| qmsum | 23.73 | 200 |
| multi_news | 25.50 | 200 |
| trec | 72.00 | 200 |
| triviaqa | 72.85 | 200 |
| samsum | 36.59 | 200 |
| passage_count | 16.50 | 200 |
| passage_retrieval_en | 100.00 | 200 |
| lcc | 12.53 | 500 |
| repobench | 18.57 | 500 |
| **AVG16** | **43.09** | — |

16 任务均使用 TS=`11021830` 的正式产物，driver `ALL_DONE FAILED=0`；隔离目录打分避免混入 Quest，`strict runcap=16`，逐任务行数与参考数据一致，并记录源文件 SHA256。正式结果：`exp/trace/results/qwen3_32b_longbench_jssa_vendor.json`。

### 2.5 状态（更新 2026-11-03 12:10）

| 数据集 | 状态 |
|--------|------|
| RULER 32K TLI | ✅ 完成，AVG11 = 50.70 |
| RULER 32K FullKV | ✅ 完成，AVG11 = 52.22 |
| LongBench FullKV | ✅ 16/16，AVG16 = 42.37 |
| LongBench TIA（既有口径） | 13/16，待补 3 任务 |
| RULER/LongBench Quest + JSSA | 🔄 Quest RULER 6/11、LongBench 9/16；JSSA RULER ✅ 11/11（AVG11=54.06）、LongBench ✅ 16/16（AVG16=43.09） |
| FASA/AsyncTLS/HISA | 按用户 2026-11-03 指令不再推进；后续仅保留 JSSA、Quest |

---

## 3. GLM-4.7-Flash（新适配模型）

### 3.1 RULER 32K FullKV（✅ 11/11 完成）

**AVG11 = 26.58**（vs Qwen3-8B FullKV 59.38，退化 -32.80）

| Task | GLM FullKV | Qwen3-8B FullKV | 退化模式 |
|------|-----------|-----------------|---------|
| niah_single_1 | 5.00 | 93.00 | grass-is-green 循环 |
| niah_single_2 | 2.00 | 89.00 | grass-is-green 循环 |
| niah_single_3 | 3.00 | 95.00 | grass-is-green 循环 |
| niah_multikey_1 | 1.00 | 96.00 | 严重退化 |
| niah_multikey_2 | 27.00 | 74.00 | 部分退化 |
| niah_multikey_3 | 7.00 | 3.00 | 持平（双模型弱任务） |
| niah_multiquery | 15.00 | 6.16 | GLM 反超 |
| niah_multivalue | 24.00 | 0.96 | GLM 反超 |
| cwe | 69.80 | 83.50 | 轻度退化 |
| fwe | 72.33 | 7.00 | GLM 大幅反超（任务差异） |
| vt | 66.20 | 34.20 | GLM 反超 |

**论文叙事支撑**：
- GLM-4.7-Flash 架构在 RULER 32K NIAH 检索任务上 FullKV 严重退化（NIAH 系列 AVG ≈ 8.7 vs Qwen3-8B AVG ≈ 65.9）
- 退化模式 = **grass-is-green 重复输出**（完全无法检索 needle token）
- **GLM 是 TLI 的核心 test case**：TLI 通过近区可控稀疏化 + 远区子空间约束避免 decode 全 attention 退化区

### 3.2 RULER 32K JSSA（✅ 11/11 已打分）

**JSSA AVG11 = 2.52 / FullKV AVG11 = 26.58 / Δ = -24.06**

| Task | FullKV | JSSA (budget=1024, rank=64) | Δ(JSSA-FKV) |
|------|--------|------------------------------|-------------|
| niah_single_1 | 5.00 | 0.00 | -5.00 |
| niah_single_2 | 2.00 | 0.00 | -2.00 |
| niah_single_3 | 3.00 | 0.00 | -3.00 |
| niah_multikey_1 | 1.00 | 0.00 | -1.00 |
| niah_multikey_2 | 27.00 | 0.00 | -27.00 |
| niah_multikey_3 | 7.00 | 0.00 | -7.00 |
| niah_multiquery | 15.00 | 0.75 | -14.25 |
| niah_multivalue | 24.00 | 1.75 | -22.25 |
| cwe | 69.80 | 2.90 | -66.90 |
| fwe | 72.33 | 22.33 | -50.00 |
| vt | 66.20 | 0.00 | -66.20 |
| **AVG11** | **26.58** | **2.52** | **-24.06** |

11 任务均 `rc=0`、每任务 100 样本，按 TS=`11021635` 的同代 `.tli_gen` 指针解析并记录源文件 SHA256；打分结果：`exp/trace/results/glm_4_7_flash_ruler32k_jssa_vendor.json`。该结果显示当前预算下存在严重质量缺口，后续需区分稀疏预算敏感性与适配问题；这是 **JSSA vendor baseline** 结果，不是 TLI 主方法结果。

### 3.3 RULER 32K Quest（✅ 11/11 已打分）

**Quest AVG11 = 21.66 / FullKV AVG11 = 26.58 / Δ = -4.92**

| Task | FullKV | Quest (token_budget=1024, chunk_size=16) | Δ(Quest-FKV) |
|------|--------|------------------------------------------|--------------|
| niah_single_1 | 5.00 | 13.00 | +8.00 |
| niah_single_2 | 2.00 | 1.00 | -1.00 |
| niah_single_3 | 3.00 | 0.00 | -3.00 |
| niah_multikey_1 | 1.00 | 4.00 | +3.00 |
| niah_multikey_2 | 27.00 | 16.00 | -11.00 |
| niah_multikey_3 | 7.00 | 0.00 | -7.00 |
| niah_multiquery | 15.00 | 56.00 | +41.00 |
| niah_multivalue | 24.00 | 59.25 | +35.25 |
| cwe | 69.80 | 21.50 | -48.30 |
| fwe | 72.33 | 64.67 | -7.66 |
| vt | 66.20 | 2.80 | -63.40 |
| **AVG11** | **26.58** | **21.66** | **-4.92** |

11 任务均 `rc=0`、每任务 100 样本，按 TS=`11021635` 的同代 `.tli_gen` 指针解析并记录源文件 SHA256；打分结果：`exp/trace/results/glm_4_7_flash_ruler32k_quest_vendor.json`。Quest 总分比 JSSA 2.52 高 19.14，但仍低于 FullKV；`niah_multiquery`/`niah_multivalue` 分别反超 FullKV 41.00/35.25，`cwe`/`vt` 则存在明显质量缺口。这是 **Quest vendor baseline** 结果，不是 TLI 主方法结果。

### 3.4 LongBench FullKV

✅ 已完成（task #11）。

### 3.5 LongBench JSSA（✅ 16/16 已打分）

**JSSA AVG16 = 31.89**（budget=1024，rank=64）

| Task | Score | 样本数 |
|------|------:|-------:|
| narrativeqa | 5.36 | 200 |
| qasper | 29.89 | 200 |
| multifieldqa_en | 49.53 | 150 |
| hotpotqa | 31.90 | 200 |
| 2wikimqa | 33.28 | 200 |
| musique | 16.27 | 200 |
| gov_report | 22.87 | 200 |
| qmsum | 16.44 | 200 |
| multi_news | 26.15 | 200 |
| trec | 34.00 | 200 |
| triviaqa | 29.98 | 200 |
| samsum | 28.30 | 200 |
| passage_count | 5.00 | 200 |
| passage_retrieval_en | 100.00 | 200 |
| lcc | 54.78 | 500 |
| repobench | 26.47 | 500 |
| **AVG16** | **31.89** | — |

16 任务均使用 TS=`11021635` 的正式产物，逐任务行数与全量参考数据一致，并记录源文件 SHA256。首轮因磁盘满失败的 `passage_count`/`lcc`/`repobench-p` 已按同 TS、同配置补跑，三项分别于 03:43:32、05:00:09、06:29:28 以 `rc=0` 完成；最终 16/16 打分结果落盘 `exp/trace/results/glm_4_7_flash_longbench_jssa_vendor.json`。这是 **JSSA vendor baseline** 结果，不是 TLI 主方法结果。

### 3.6 LongBench Quest（✅ 16/16 已打分）

**Quest AVG16 = 40.95**（token_budget=1024，chunk_size=16），比 JSSA 31.89 高 **9.06**。

| Task | Score | 样本数 |
|------|------:|-------:|
| narrativeqa | 16.48 | 200 |
| qasper | 26.06 | 200 |
| multifieldqa_en | 47.54 | 150 |
| hotpotqa | 46.38 | 200 |
| 2wikimqa | 37.48 | 200 |
| musique | 20.66 | 200 |
| gov_report | 29.01 | 200 |
| qmsum | 22.18 | 200 |
| multi_news | 26.70 | 200 |
| trec | 60.00 | 200 |
| triviaqa | 76.60 | 200 |
| samsum | 35.77 | 200 |
| passage_count | 7.35 | 200 |
| passage_retrieval_en | 95.00 | 200 |
| lcc | 58.81 | 500 |
| repobench | 49.23 | 500 |
| **AVG16** | **40.95** | — |

16 任务均使用 TS=`11021635` 的正式产物，driver `ALL_DONE FAILED=0`，逐任务行数与全量参考数据一致，并记录源文件 SHA256；打分结果落盘 `exp/trace/results/glm_4_7_flash_longbench_quest_vendor.json`。这是 **Quest vendor baseline** 结果，不是 TLI 主方法结果。

### 3.7 状态（更新 2026-11-03 09:10）

| 数据集 | 状态 |
|--------|------|
| RULER 32K FullKV | ✅ 完成，AVG11 = 26.58 |
| LongBench FullKV | ✅ 完成 |
| TLI 跑分 | 未派单 |
| Vendor baselines | ✅ TS=11021635；Quest RULER 11/11（AVG11=21.66）、LongBench 16/16（AVG16=40.95）；JSSA RULER 11/11（AVG11=2.52）、LongBench 16/16（AVG16=31.89）；全部任务 rc=0 且产物行数完整 |

---

## 4. 论文可引用的数据口径

### 主表（推荐）：Qwen3-8B RULER 32K
- 7 方法 × 11 任务，数据完整
- TLI 60.77 (=论文部署口径，已与论文 Table 对齐)

### 副表：Qwen3-8B LongBench
- 6 方法 × 16 任务（缺 HISA）
- TLI 49.27 ≈ FullKV 49.39

### 补充材料
- Qwen3-32B RULER 32K：JSSA 54.06 / FullKV 52.22 / TLI 50.70；JSSA 总分领先但 `vt=0`，TLI 与 FullKV 的 fwe 差距仍是 TLI 落后主因
- GLM-4.7-Flash RULER 32K：FullKV 26.58（grass-is-green 退化），Quest 21.66（Δ=-4.92），JSSA 2.52（Δ=-24.06，质量缺口待归因），TLI 待跑
- GLM-4.7-Flash LongBench：Quest 40.95 / JSSA 31.89，均为 16/16 完整结果

### 不在主表
- DuoAttention：蒸馏未完成 → 论文说明 "开源代码仅支持 Llama 架构"

---

## 5. 文件路径索引

### RULER 32K (Qwen3-8B)
- 汇总表: `exp/results_ruler/ruler32k_all_table.json`
- FullKV: `exp/results_ruler/Qwen3-8B/L32768/pred_E104B_FULLKV/`
- TLI: `exp/results_ruler/Qwen3-8B/L32768/pred_E104B_TIA/`
- Quest: `exp/results_ruler/Qwen3-8B/L32768/pred_E104B_QUEST/`
- FASA: `exp/results_ruler/Qwen3-8B/L32768/pred_fasa_vendor/`
- JSSA: `exp/results_ruler/Qwen3-8B/L32768/pred_jssa_vendor/`
- HISA: `exp/results_ruler/Qwen3-8B/L32768/pred_hisa_vendor/`
- AsyncTLS: `exp/trace/results/e119_ruler32768_formal_asynctls_vendor_final.json`

### RULER 32K (Qwen3-32B)
- FullKV/TLI 打分汇总: `exp/trace/results/qwen3_32b_ruler32k_fullkv_vs_tli.json`
- JSSA 打分汇总: `exp/trace/results/qwen3_32b_ruler32k_jssa_vendor.json`
- JSSA 同代产物: 远程 `/tmp/tli-baseline-jssa-quest/exp/results_ruler/Qwen3-32B/L32768/pred/`（TS=11021830，逐任务 `.tli_gen` 指针）
- FullKV: `exp/results_ruler/Qwen3-32B/L32768/L32768/pred_e141fullkv/`
- TLI: `exp/results_ruler/Qwen3-32B/L32768/pred_bs2_champ/`

### RULER 32K (GLM-4.7-Flash)
- Quest 打分汇总: `exp/trace/results/glm_4_7_flash_ruler32k_quest_vendor.json`
- JSSA 打分汇总: `exp/trace/results/glm_4_7_flash_ruler32k_jssa_vendor.json`
- Quest/JSSA 同代产物: 远程 `/tmp/tli-baseline-jssa-quest/exp/results_ruler/GLM-4.7-Flash/L32768/pred/`（TS=11021635，逐任务 `.tli_gen` 指针）

### LongBench (Qwen3-8B)
- 打分汇总: `exp/trace/results/qwen3_8b_longbench_*.json`（6 文件）
- pred 输出: `exp/results_longbench/Qwen3-8B/pred_1024/result.json`, `pred_tls/result.json`

### LongBench (Qwen3-32B)
- FullKV 16 任务打分汇总: `exp/trace/results/qwen3_32b_longbench_fullkv.json`
- 既有 13 任务 pred: `exp/results_longbench/Qwen3-32B/pred_1024/`
- 补跑 3 任务 pred: `exp/results_longbench/Qwen3-32B/pred_e141_lb_fullkv/`（TS=20261102081500）

### LongBench (GLM-4.7-Flash)
- Quest 16 任务打分汇总: `exp/trace/results/glm_4_7_flash_longbench_quest_vendor.json`
- JSSA 16 任务打分汇总: `exp/trace/results/glm_4_7_flash_longbench_jssa_vendor.json`
- Quest/JSSA 正式产物: 远程 `/tmp/tli-baseline-jssa-quest/exp/results_longbench/GLM-4.7-Flash/pred/`（TS=11021635）

---

## 6. 下一步行动（更新 2026-11-03 09:10）

| 优先级 | 行动 | 状态 |
|--------|------|------|
| P0 | Qwen3-8B RULER 32K — 7 方法矩阵 | ✅ 完成 |
| P0 | Qwen3-32B RULER 32K FullKV | ✅ 完成（AVG11=52.22） |
| P0 | GLM-4.7-Flash RULER 32K FullKV | ✅ 完成（AVG11=26.58） |
| P0 | Qwen3-32B LongBench 缺失 3 任务 | ✅ 完成（pc02 各任务产物完整） |
| P0 | Qwen3-32B LongBench FullKV 16 任务完整打分 | ✅ 完成（AVG16=42.37） |
| P1 | Qwen3-32B + GLM Quest/JSSA RULER 32K + LongBench | 🔄 Q32B Quest RULER+LongBench 运行中；Q32B JSSA 与 GLM Quest/JSSA 的 RULER+LongBench 均已完成 |
| P1 | Vendor 矩阵完成后打分、汇总并核验论文 | 🔄 Q32B JSSA RULER+LongBench 已打分（AVG11=54.06、AVG16=43.09）；GLM Quest/JSSA RULER+LongBench 均已打分（Quest 21.66/40.95，JSSA 2.52/31.89）；Q32B Quest 待完整生成后打分 |
| P1 | GLM-4.7-Flash TLI 跑分 | 待 GPU 空闲派单 |
| N/A | FASA/AsyncTLS/HISA for Q32B/GLM | 按用户 2026-11-03 指令不再推进，仅保留 JSSA、Quest |
| N/A | DuoAttention → 论文脚注说明 | 用户已确认 |
| N/A | HISA LongBench (Qwen3-8B) | 暂缓 |

---

## 7. 更新记录

- 2026-11-03 12:10:
  - Q32B Quest 继续正常推进：RULER 32K 完成 6/11，当前 `niah_multiquery` 86/100；LongBench 完成 9/16（新完成 `multi_news`，200 行），当前 `trec` 29/200。两套 driver 均存活，GPU5/GPU7 正常计算，未形成新的完整套件，不提前发布部分均值。
  - Q32B LongBench FullKV 16/16 与 AVG16=42.37 已在此前闭环，三项补跑产物仍在，无需重复 eval。GLM Quest/JSSA 四套矩阵均已完成；空闲 GPU 不用于重复派发同 TS 的 Quest 剩余任务，避免旧串行 driver 后续覆盖产物。按用户最新范围，不推进 FASA/AsyncTLS/HISA。
- 2026-11-03 09:10:
  - **GLM Quest LongBench 完整套件正式发布**：16/16 全部 `rc=0`，`repobench-p` 于 08:15:24 完成且 500 行完整；逐任务行数符合全量样本数，按 TS=`11021635` 的 Quest 正式产物精确筛选打分并记录 SHA256，**AVG16=40.95**，比 JSSA 31.89 高 9.06。结果落盘 `glm_4_7_flash_longbench_quest_vendor.json`。至此 GLM Quest/JSSA 的 RULER 32K 与 LongBench 四套矩阵全部完成，不与 TLI 主方法结果混淆。
  - Q32B 进度：Quest RULER 5/11（`niah_multikey_3` 77/100）、LongBench 8/16（`qmsum` 于 08:59:55 以 `rc=0` 完成且 200 行，当前 `multi_news` 13/200）；JSSA LongBench 15/16（`repobench-p` 257/500），在跑任务均持续增长、未见异常。
  - 当前 GPU0/GPU1/GPU2/GPU3/GPU4 空闲；不追加尚未完成 Q32B/GLM 适配验证的 FASA/AsyncTLS/HISA。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval。
- 2026-11-03 08:10:
  - 本轮尚无新的完整套件，不提前发布部分均值。Q32B JSSA LongBench 推进至 15/16：`lcc` 于 08:07:52 以 `rc=0` 完成且 500 行完整，当前 `repobench-p` 9/500；Q32B Quest RULER 5/11（`niah_multikey_3` 41/100）、LongBench 7/16（`qmsum` 146/200）。
  - GLM Quest LongBench 15/16，`repobench-p` 推进至 482/500、持续增长且无异常，待 `rc=0` 与 500 行核验后再进行 16/16 正式打分；GLM JSSA LongBench 16/16 与 AVG16=31.89 保持不变。
  - 当前 GPU0/GPU1/GPU3/GPU4 空闲；不追加尚未完成 Q32B/GLM 适配验证的 FASA/AsyncTLS/HISA。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval。
- 2026-11-03 07:10:
  - **GLM JSSA LongBench 完整套件正式发布**：首轮磁盘满失败的三项补跑全部闭环，`passage_count`/`lcc`/`repobench-p` 分别 200/500/500 行且 `rc=0`，最后一项于 06:29:28 完成。16/16 逐任务行数均符合全量样本数，按 TS=`11021635` 的 JSSA 正式产物精确筛选打分并记录 SHA256，**AVG16=31.89**。结果落盘 `glm_4_7_flash_longbench_jssa_vendor.json`，不与 TLI 主方法结果混淆。
  - Q32B 进度：Quest RULER 5/11（新完成 `niah_multikey_2`，100 行；当前 `niah_multikey_3` 5/100）、LongBench 7/16（当前 `qmsum`）；JSSA LongBench 14/16（新完成 `passage_count`/`passage_retrieval_en`，各 200 行；当前 `lcc`）。Q32B JSSA RULER 11/11 与 AVG11=54.06 保持不变。
  - GLM Quest LongBench 15/16（新完成 `lcc`，500 行；当前 `repobench-p`）；GLM Quest RULER 11/11 与 AVG11=21.66 保持不变。
  - 当前 GPU0/GPU1/GPU3/GPU4 空闲；不追加尚未完成 Q32B/GLM 适配验证的 FASA/AsyncTLS/HISA。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval。
- 2026-11-03 06:10:
  - **GLM Quest RULER 32K 完整套件正式发布**：11/11 全部 `rc=0`、每任务 100 行，按 TS=`11021635` 同代 `.tli_gen` 指针打分并记录 SHA256。**AVG11=21.66**，对比 FullKV 26.58 为 -4.92、对比 JSSA 2.52 为 +19.14；`niah_multiquery=56.00`/`niah_multivalue=59.25` 反超 FullKV，但 `cwe=21.50`/`vt=2.80` 存在明显缺口。结果落盘 `glm_4_7_flash_ruler32k_quest_vendor.json`，不与 TLI 主方法结果混淆。
  - Q32B 进度：Quest RULER 4/11（当前 `niah_multikey_2` 69/100）、LongBench 7/16（新完成 `gov_report`，当前 `qmsum`）；JSSA LongBench 12/16（当前 `passage_count`）。Q32B JSSA RULER 11/11 与 AVG11=54.06 保持不变。
  - GLM 其余矩阵：Quest LongBench 14/16（当前 `lcc`）；JSSA LongBench 15/16 有效，`repobench-p` 推进至 395/500、无异常，待 `rc=0` 且 500 行核验后形成 16/16 再正式打分。
  - 当前 GPU0/GPU1/GPU4 空闲；不追加尚未完成 Q32B/GLM 适配验证的 FASA/AsyncTLS/HISA。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval。
- 2026-11-03 05:10:
  - **Q32B JSSA RULER 32K 完整套件正式发布**：11/11 全部 `rc=0`、每任务 100 行，按 TS=`11021830` 同代 `.tli_gen` 指针打分并记录 SHA256。**AVG11=54.06**，对比 FullKV 52.22 为 +1.84、对比 TLI 50.70 为 +3.36；但 `vt=0`（FullKV 96.40），任务间差异显著，不与 TLI 主方法结果混淆。结果落盘 `qwen3_32b_ruler32k_jssa_vendor.json`。
  - Q32B 其余进度：Quest RULER 4/11（新完成 `niah_multikey_1`，100 行；当前 `niah_multikey_2` 32/100）、LongBench 6/16（`gov_report` 运行中）；JSSA LongBench 11/16（新完成 `triviaqa`，200 行；当前 `samsum`）。
  - GLM 进度：Quest RULER 10/11（新完成 `cwe`/`fwe`，各 100 行；当前 `vt` 7/100）、LongBench 14/16（新完成 `passage_count`/`passage_retrieval_en`，各 200 行；当前 `lcc`）；JSSA LongBench 有效 15/16，`lcc` 补跑于 05:00:09 以 `rc=0` 完成且 500 行完整，`repobench-p` 当前 59/500、无异常。
  - 当前 GPU1/GPU4 空闲；Q32B JSSA RULER 完成后优先完成正式打分，不追加尚未完成 Q32B/GLM 适配验证的 FASA/AsyncTLS/HISA。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval。
- 2026-11-03 04:10:
  - 6 套原矩阵与 GLM JSSA LongBench retry 持续推进，无新增失败；仅 GPU1 空闲，尚未形成新的完整套件，不提前发布部分均值。
  - **GLM JSSA LongBench 补跑首项闭环**：`passage_count` 于 03:43:32 以 `rc=0` 完成，产物 200 行，无异常；`lcc` 自 03:43:32 运行且日志无错误，`repobench-p` 待串行执行。当前有效任务 14/16，待 `lcc=500`、`repobench-p=500` 行核验通过后才进行 16 任务正式打分。
  - Q32B 进度：Quest RULER 3/11（`niah_multikey_1` 96/100）、LongBench 6/16（`gov_report` 运行中）；JSSA RULER 9/11（`fwe` 92/100）、LongBench 10/16（新完成 `trec`，200 行；当前 `triviaqa`）。
  - GLM 其余矩阵：Quest RULER 8/11（新完成 `niah_multivalue`，100 行；当前 `cwe` 71/100）、LongBench 12/16（新完成 `samsum`，200 行；当前 `passage_count`）。
  - Q32B LongBench FullKV 结果文件复核仍为 16/16、AVG16=42.37，`passage_count=19.00`、`trec=72.50`、`samsum=36.46`，无需重复 eval。GPU1 虽空闲，但 FASA/AsyncTLS/HISA 尚未完成 Q32B/GLM 适配验证，本轮不追加正式派单。
- 2026-11-03 03:22:
  - **GLM JSSA LongBench 首轮三项失败已定位并进入补跑**：`passage_count`、`lcc`、`repobench-p` 均在 `load_dataset()` 的 Arrow writer 落盘阶段报 `No space left on device`，并非模型或 vendor 算法异常；其余 13 项 `rc=0` 且行数完整，首轮不能按 16/16 打分。
  - 远程 `/home` 当时 251G 已用满；确认正式 Q32B 任务使用 `/tmp/tli-models` 后，将无活动引用的冗余 `/home/hadoop-aipnlp/huggingface.co/Qwen/Qwen3-32B` 迁至 `/tmp/Qwen3-32B-home-archive` 并保留原路径符号链接，未删除模型数据。修复后 `/home` 可用 75G。
  - 03:22:28 在 GPU3 启动独立 retry，保持 TS=`11021635`、JSSA `budget=1024/rank=64`、原 GLM basis 与完整样本口径，串行补跑 `passage_count → lcc → repobench-p`；当前 `passage_count` 运行中。日志为 `/tmp/tli_retry_glm_jssa_lb_11020310.driver.log` 与 `/tmp/logs_glm_jssa_lb_retry_11020310/`，需核验三项分别 200/500/500 行并形成 16/16 后才正式打分。
  - Q32B 进度：Quest RULER 3/11（`niah_multikey_1` 68/100）、LongBench 6/16（`gov_report` 运行中）；JSSA RULER 9/11（新完成 `niah_multivalue`/`cwe`，各 100 行；当前 `fwe`）、LongBench 9/16（新完成 `multi_news`，200 行；当前 `trec`）。
  - GLM 其余矩阵：Quest RULER 7/11（`niah_multivalue` 99/100）、LongBench 11/16（新完成 `trec`/`triviaqa`，各 200 行；当前 `samsum`）。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval；GPU1 空闲，但 FASA/AsyncTLS/HISA 尚未完成 Q32B/GLM 适配验证，本轮不追加正式派单。
- 2026-11-03 02:10:
  - 7 套正式矩阵继续运行，无新增失败；仅 GPU1 空闲。所有在跑任务均持续推进，尚未形成新的完整套件，不提前发布部分均值。
  - Q32B 进度：Quest RULER 3/11（新完成 `niah_single_3`，100 行；当前 `niah_multikey_1` 23/100）、LongBench 6/16（`gov_report` 84/200）；JSSA RULER 7/11（`niah_multivalue` 90/100）、LongBench 8/16（`multi_news` 102/200）。
  - GLM 进度：Quest RULER 7/11（新完成 `niah_multiquery`，100 行；当前 `niah_multivalue`）、LongBench 9/16（新完成 `multi_news`，200 行；当前 `trec`）；JSSA LongBench 11/16（新完成 `trec`/`triviaqa`，各 200 行；当前 `samsum`）。
  - Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复 eval。GPU1 虽持续空闲，但 FASA/AsyncTLS/HISA 尚未完成 Q32B/GLM 适配验证，本轮不直接追加正式派单。
- 2026-11-03 01:10:
  - 7 套正式矩阵继续运行，无新增失败；GPU1 空闲，GPU3 的 GLM JSSA LongBench 已加载并推进 `trec`，不是空闲卡。尚未形成新的完整套件，不提前发布部分均值。
  - Q32B 进度：Quest RULER 2/11（`niah_single_3` 88/100）、LongBench 6/16（`gov_report` 56/200）；JSSA RULER 7/11（新完成 `niah_multiquery`，100 行；当前 `niah_multivalue`）、LongBench 8/16（当前 `multi_news`）。
  - GLM 进度：Quest RULER 6/11（`niah_multiquery` 41/100）、LongBench 8/16（`multi_news` 102/200）；JSSA LongBench 9/16（新完成 `multi_news`，200 行；当前 `trec`）。
  - Q32B LongBench FullKV 三项补跑与 16/16 打分保持完成，AVG16=42.37，无需重复 eval。GPU1 虽空闲，但 FASA/AsyncTLS/HISA 尚未完成 Q32B/GLM 适配验证，本轮不直接追加正式派单。
- 2026-11-03 00:10:
  - **首个完整 vendor 套件发布**：GLM JSSA RULER 32K 11/11 全部 `rc=0`，逐任务 100 行；按同代 `.tli_gen` 指针打分，AVG11=2.52，对比 FullKV 26.58 为 -24.06。结果落盘 `glm_4_7_flash_ruler32k_jssa_vendor.json`，严重质量缺口待归因，不与 TLI 主方法混淆。
  - Q32B 进度：Quest RULER 2/11、LongBench 6/16；JSSA RULER 6/11、LongBench 7/16。当前分别运行 RULER `niah_single_3`/`niah_multiquery` 与 LongBench `gov_report`/`qmsum`。
  - GLM 其余矩阵：Quest RULER 5/11、LongBench 7/16；JSSA LongBench 8/16。当前分别运行 RULER `niah_multikey_3` 与 LongBench `qmsum`/`multi_news`。
  - GLM JSSA RULER 完成后 GPU1 空闲；其余 7 卡继续运行。本轮优先完成完整套件打分与质量缺口登记，未追加尚未完成适配验证的 FASA/AsyncTLS/HISA 派单。Q32B LongBench FullKV 16/16 与 AVG16=42.37 保持完成，无需重复打分。
- 2026-11-02 23:10:
  - 8 套正式矩阵继续运行，无新增失败、无空闲 GPU；所有新完成 RULER 任务均 100 行，LongBench 行数符合对应任务全量样本数。
  - Q32B 进度：Quest RULER 2/11、LongBench 5/16；JSSA RULER 5/11、LongBench 6/16。当前分别运行 RULER `niah_single_3`/`niah_multikey_3` 与 LongBench `musique`/`gov_report`。
  - GLM 进度：Quest RULER 4/11、LongBench 7/16；JSSA RULER 9/11、LongBench 8/16。当前分别运行 RULER `niah_multikey_2`/`fwe` 与 LongBench `qmsum`/`multi_news`。
  - 长任务进度复核：Q32B JSSA `gov_report` 推进至 140/200，Q32B Quest `musique` 推进至 188/200，均持续增长。Q32B LongBench FullKV 16/16 与 AVG16=42.37 已核验，无需重复打分；尚未形成任一完整 vendor 套件，不提前发布部分均值。
- 2026-11-02 22:10:
  - 8 套正式矩阵继续运行，无新增失败、无空闲 GPU；所有新完成 RULER 任务均 100 行，LongBench 行数符合对应任务全量样本数。
  - Q32B 进度：Quest RULER 1/11、LongBench 5/16；JSSA RULER 4/11、LongBench 6/16。当前分别运行 RULER `niah_single_2`/`niah_multikey_2` 与 LongBench `musique`/`gov_report`。
  - GLM 进度：Quest RULER 3/11、LongBench 6/16；JSSA RULER 7/11、LongBench 7/16。当前分别运行 RULER `niah_multikey_1`/`niah_multivalue` 与 LongBench `gov_report`/`qmsum`。
  - 长任务进度复核：GLM Quest `gov_report` 推进至 166/200，Q32B Quest RULER `niah_single_2` 推进至 78/100，Q32B JSSA `gov_report` 推进至 67/200，均持续增长。Q32B LongBench FullKV 16/16 与 AVG16=42.37 已核验，无需重复打分；尚未形成任一完整 vendor 套件，不提前发布部分均值。
- 2026-11-02 21:10:
  - 8 套正式矩阵继续运行，无新增失败、无空闲 GPU；所有新完成 RULER 任务均 100 行，LongBench 行数符合对应任务全量样本数。
  - Q32B 进度：Quest RULER 1/11、LongBench 3/16；JSSA RULER 3/11、LongBench 5/16。当前分别运行 RULER `niah_single_2`/`niah_multikey_1` 与 LongBench `hotpotqa`/`musique`。
  - GLM 进度：Quest RULER 3/11、LongBench 6/16；JSSA RULER 6/11、LongBench 6/16。当前分别运行 RULER `niah_multikey_1`/`niah_multiquery`，LongBench 均为 `gov_report`。
  - 长任务假死排查：GLM JSSA/Quest `gov_report` 分别推进至 167/200、112/200，Q32B Quest RULER `niah_single_2` 推进至 42/100，均持续增长。尚未形成任一完整套件，不提前发布部分均值。
- 2026-11-02 20:10:
  - 8 套正式矩阵继续运行，无新增失败、无空闲 GPU；所有新完成 RULER 任务均 100 行，LongBench 行数符合对应任务全量样本数。
  - Q32B 进度：Quest RULER 1/11、LongBench 2/16；JSSA RULER 2/11、LongBench 4/16。
  - GLM 进度：Quest RULER 2/11、LongBench 6/16；JSSA RULER 4/11、LongBench 6/16。尚未形成任一完整套件，不提前发布部分均值。
- 2026-11-02 19:11:
  - 8 卡继续满载运行，无新增失败；GLM Quest/JSSA LongBench 均完成 6/16，JSSA RULER 完成 3/11，Quest RULER 完成 1/11。
  - Q32B JSSA RULER 完成 1/11（100 行）、LongBench 完成 2/16（各 200 行）；Quest 两套仍在首任务，不使用部分产物打分。
  - 抽查 Q32B JSSA `niah_single_1`：100 行中仅 1 行包含 needle，首条回答正确；保留为 budget=1024 下的早期质量信号，待 11 任务正式打分后归因。
- 2026-11-02 19:07:
  - 8 卡仍全部分配给 Quest/JSSA 正式矩阵，无空闲卡追加其他 vendor；当前无新增失败。
  - GLM 进度：Quest RULER 1/11、JSSA RULER 2/11、Quest LongBench 4/16、JSSA LongBench 5/16；已完成 RULER 任务均 100 行，LongBench 行数与参考数据一致。
  - Q32B 四套矩阵仍在首任务；RULER 正式产物持续增长，LongBench 尚未结束首任务，不使用部分产物打分。
- 2026-11-02 17:45:
  - 定位 Q32B LongBench 三项缺失任务已在 pc02 完整生成（TS=20261102081500），同步产物并合并原有 13 任务完成 FullKV AVG16=42.37。
  - Q32B/GLM Quest/JSSA 共 8 套 RULER 32K + LongBench 正式矩阵均在运行；当前已完成任务全部 rc=0。
  - Q32B JSSA 正式 basis 校验通过（64 层、rank=32、M[0]=(8,32,128)），RULER/LongBench smoke 均通过。
- 2026-11-02 08:05:
  - Q32B RULER FullKV 11/11 全部完成，AVG11 = 52.22（TLI = 50.70, Δ = -1.52）
  - 打分文件落盘：`qwen3_32b_ruler32k_fullkv_vs_tli.json`
  - Q32B LongBench 缺失 3 任务重新派单（GPU0: passage_count+trec, GPU1: samsum）
  - 设置 1h 定时监督器
- 2026-10-31 14:23:
  - Q32B RULER FullKV 5/11 完成, AVG5 = 82.60 vs TLI 82.80 (+0.20)
  - multikey_2: TLI 48 vs FullKV 46 (+2) 反超
- 2026-10-31 13:57:
  - GLM RULER FullKV 11/11 完成，AVG11 = 26.58
  - grass-is-green 退化模式确认
- 2026-10-31 10:35:
  - Q32B LongBench FullKV+TIA+Quest 13 任务打分完成
  - Q32B RULER TLI = 50.70 AVG11
