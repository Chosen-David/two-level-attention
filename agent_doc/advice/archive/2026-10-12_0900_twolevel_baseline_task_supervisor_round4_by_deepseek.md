# twolevel baseline 任务监督 round4（by deepseek）

- 时间：2026-10-12 09:0X
- 范围：S-T023 开源 baseline 复现任务链（sanity 门禁验证 + 32B quest 收尾跟踪）
- 结论：**全预算(32768) quest ≈ fullkv 6 任务确证（Δ∈[-1.48,+1.0]）——Quest vendor 适配层正确性 sanity 硬门禁通过，8B quest 主表数据可上主表。32B quest 13/16，3 在跑。**

## 1. sanity 门禁核心结论（本轮主动作：打分 4 个新全预算 sanity）

- **rsync 回本地 4 个新完成的 sanity**（.24 → 本地）→ 归位到 quest_sanity_<task>_32768/pred/ →
  用 quest venv eval.py 打分（jieba/rouge deps 确认齐全）。

| 任务 | fullkv | quest 全预算(32768) | Δ |
|---|---|---|---|
| hotpotqa | 59.07 | 57.59 | -1.48 |
| musique | 35.44 | 34.14 | -1.30 |
| narrativeqa | 27.59 | 27.42 | -0.17 |
| samsum | 44.62 | 44.19 | -0.43 |
| qasper（已有）| 47.47 | 46.57 | -0.90 |
| trec（已有）| 71.5 | 72.5 | +1.0 |

- **判定**：全预算 quest ≈ fullkv（Δ 微小，无方向性系统偏差）→ **Quest 论文核心结论
  （token_budget 全预算≈dense）在 Qwen3-8B 上 6 任务确证**。适配层（q_norm/k_norm/scaling/
  32B hidden_size 三修复）正确性 sanity 硬门禁通过 → quest 1024 预算 8B 数据（AVG 41.35）
  **可上主表**（标注模型/任务集差异）。

## 2. 32B quest 进度（13/16）

- 完成：本地 narrativeqa/qasper/passage_count + .24 2wikimqa/trec/triviaqa + .187 8 任务 +
  g01 passage_retrieval_en = 13/16。
- 在跑 3 个独特任务（均有双跑，086 best-file 裁决）：
  - gov_report（.24 05:31 + .187 05:28 双跑，~4.7h）
  - qmsum（.24 05:31，~4.7h）
  - repobench-p（.187 主批 05:39 + g01 05:07 双跑，~5h）
- 预计 1-3h 全部完成 → 合并 + 086 best-file → eval 打分 + 空卡贴合（Quest 检索掉/摘要维持趋势）。

## 3. LBv2 锚点（推进中）

- LBv2 quest（.24 GPU0，503 样本）+ LBv2 fullkv（.187 GPU1，503 样本）推进中。
  LBv2 已有 TLI mavg/aavg 打分，fullkv/quest 完成后三件套齐 → 打分（lbv2 scorer）。

## 4. 下一步

1. 32B quest 16/16 齐 → 合并 pred_qv_1024_16{,_g01,_24}/ + 本地 → 086 best-file → eval 打分
   + 空卡贴合（sanity 硬门禁）。
2. LBv2 fullkv/quest 完成 → 打分 → LBv2 三件套（TLI mavg/aavg + fullkv + quest）。
3. 主表拼接：8B quest 41.35 + fullkv 49.39 + DeepSeek 40.55 + 32B quest（完成后）。

— by deepseek（rsync sanity 结果回本地 + 归位 + quest venv 打分 4 任务 + 与 fullkv 对拍 Δ 表 + 32B 进度盘点）
