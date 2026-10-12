# twolevel GPU 监督 round4（by deepseek）

- 时间：2026-10-12 05:5X
- 范围：S-T023 竞品复现任务链（Quest vendor 8B/32B + DeepSeek fullkv 锚点）
- 结论：**发现并修复一处真 bug——.24 的 qwen3_quest_patch.py 是旧版（缺 Qwen3-32B hidden_size reshape 修复），派到 .24 的 5 个 32B quest 任务全部在 decode reshape 崩溃白跑（rc=0 假成功）；已 scp 修复版 patch + 重派 5 任务 + 本地补派 passage_count → 32B quest 16/16 任务全覆盖。** .24 DeepSeek 043751 9/11 完成。

## 1. 发现的新 bug：.24 旧版 quest patch（实质修复）

- **现象**：.24 GPU0/1/2/3/7 空 5 卡派 32B quest（2wikimqa/gov_report/qmsum/trec/triviaqa，TS 052311）后，card0（2wikimqa）log 报
  `RuntimeError: shape '[1, 1, 5120]' is invalid for input of size 8192`（quest_attention_official.py:258 reshape），且 rc=0 假成功 → **白跑**。
- **根因**（md5 核对，非臆测）：
  - 本地/.187 的 `qwen3_quest_patch.py` md5=`09bfe078...`，line 102 有 `module.hidden_size = module.num_heads * module.head_dim`（ace966b 修复：Qwen3-32B config 显式 `head_dim=128`，而 `hidden_size=5120 ≠ 64×128=8192`，Llama/Mistral/Qwen3-8B 满足 hidden=heads×head_dim 故未暴露）。
  - **.24 的 patch md5=`bd503f65...`，line 93 是 `module.hidden_size = module.config.hidden_size`（旧版，无 32B 修复）→ 官方 forward 末尾 reshape 到 5120 但 attn_output 有 8192 元素 → 崩溃。**
- **修复动作**：scp 本地修复版 patch → .24（md5 变 09bfe078 一致，line 102 注入确认）→ 重派 5 任务（TS **20261012-053150**）→ 验证 5 进程全存活、无 Traceback、card0 已跑到 7/200（~11s/it 正常 decode）✓。
- **教训**：上轮派 .24 32B quest 前只核对了 info.py md5（66b1b90 一致），**漏核 qwen3_quest_patch.py md5 → 白跑 5 任务**。跨机派单必须核对**全部**涉及文件（info.py + qwen3_quest_patch.py + pred.py）md5。

## 2. 32B quest 16/16 任务全覆盖（本轮派单）

| 机 | 卡 | 任务 | 状态 |
|---|---|---|---|
| 本地 | GPU0 | narrativeqa/qasper（TS 002743，已完成）+ **passage_count（新派，本地 16/16 收官格）** | 3 覆盖 |
| .187 | 卡2/3/4/6/7 队列 | musique/multi_news/samsum/lcc/hotpotqa（TS 035844）| 5 在跑 |
| .187 | GPU0/1 | repobench-p/passage_retrieval_en（TS 050706，独立目录 g01）| 2 在跑 |
| .187 | 卡? | multifieldqa_en（150 行完成）| 完成 |
| .24 | GPU0/1/2/3/7 | **2wikimqa/gov_report/qmsum/trec/triviaqa（TS 053150 重派，独立目录 24）** | 5 在跑 |

- 32B quest 16 任务 = multifieldqa_en/narrativeqa/qasper + musique/multi_news/samsum/lcc/hotpotqa + repobench-p/passage_retrieval_en + 2wikimqa/gov_report/qmsum/trec/triviaqa + passage_count **全部已派**。
- 完成后合并：.187 g01 + .24 独立目录 mv → 主 pred_qv_1024_16/pred → 与本地 narrativeqa/qasper/passage_count 汇总。

## 3. DeepSeek fullkv 043751（.24，Opus 接管）进度

- **9/11 chat 格完成**：2wikimqa/hotpotqa/multifieldqa_en/multi_news/musique/narrativeqa/passage_retrieval_en/qasper/qmsum
- **2 在跑**：gov_report + passage_count（队列）
- 5 非 chat 有效格（trec/triviaqa/samsum/lcc/repobench-p，TS 023927）保留 → 11 格齐后 eval 打分（16K treatment 口径）。

## 4. GPU 占用与下一步

- .24 8 卡全满（GPU0-3/7 32B quest + GPU4/5/6 DeepSeek）；.187 8 卡全满（32B quest + 8B TLI E129 混跑）；本地 GPU0/1 新派 passage_count 后仍有 1 卡余量。
- 下一步：
  1. 32B quest 16 任务齐 → 合并 → eval 打分 + 空卡贴合（对拍 Quest 论文，sanity 硬门禁）。
  2. DeepSeek 043751 11 格齐（剩 gov_report/passage_count）→ eval 打分（16K treatment 口径 MLA dense 锚点）。
  3. 主表拼接：quest 8B 41.35 + fullkv 锚点 + DeepSeek + 32B quest（模型/任务集差异标注）。

— by deepseek（四机 nvidia-smi + ps 实查；.24 patch md5 差异定位 + scp 修复 + 重派验证；本地 passage_count 派单；DeepSeek pred 目录实查）
