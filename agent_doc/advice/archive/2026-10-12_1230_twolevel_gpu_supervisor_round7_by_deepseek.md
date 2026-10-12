# twolevel 代码 bug 监督 round7（by deepseek）

- 时间：2026-10-12 12:30
- 范围：round7 例行审查（advice 归档清理 + RULER quest 32K 打分空 postfix 真 bug 修复 + 打分对拍）
- 结论：**修复空 postfix schema 真 bug（yarn_receipt.py 生产者/消费者约束不一致）；RULER quest 32K 11/11 打分 AVG 42.85（对拍 e116e 32K 锚点 fullkv 59.38/mavg 60.0/aavg 57.35 → Quest Δ≈-16.5 符合论文趋势）；advice 已归档清理（8 份已完成 advice 移入 archive/，commit a065770）。**

## 1. advice 归档清理（用户指令，先合并远端）

- 先 `git pull origin main` 合并远端 b674309（fast-forward，远端删 2 份已归档 advice）。
- 审查 advice 根目录 10 份文档，8 份已完成 → `git mv` 到 `archive/`（commit a065770 已 push）：
  - `catpaw 基线源码复现方案`：Quest vendor 已完全落地（quest_attention_official.py + qwen3_quest_patch.py + 16 任务跑完 + sanity 8 确证）；MoBA/ClusterKV 已下载，接入待办登记在 S-T023。
  - `GPT baseline_papers_integration_addendum`：选型结论已纳入 S-T023 §0-1 收敛清单。
  - `GPT ruler_identity_scope`：RULER 身份门禁（receipt schema/指针协议/identity_mode）已实现于 score_ruler_formal.py + yarn_receipt.py。
  - round4/5/6/7 任务监督报告 + round6 代码监督报告：结论均已入 S-T023.md Progress 与 task-chain.md。
- **保留 2 份仍指导当前工作**：`benchmark_paper_scope_by_gpt`（S-T023 引用为验收门禁依据，Quest 主表验收未关闭）、`phase_matched_experiment_design_by_gpt`（E0-E4 实验设计，海选/冠军仍在进行）。

## 2. 修复空 postfix 真 bug（yarn_receipt.py）

- **现象**：RULER quest 32K 打分 `score_ruler_formal.py --pred-postfix ''` → `[GATE-FAIL] generation_params.pred_postfix='' 类型非法——fail closed（060）`。
- **根因**：三处约束不一致——
  - `pred_ruler.py:115` 默认 `--pred_postfix ""`，产物合法落 `pred/` 目录（**生产者允许空 postfix**）；
  - `score_ruler_formal.py:770` 目录定位 `pred_dir = L_dir/pred{postfix}`，空 postfix → `pred/`（**消费者目录定位支持空 postfix**）；
  - 但 `yarn_receipt.py:746` `_validate_common_schema` 强制 `generation_params.pred_postfix` 非空 str（`isinstance(v, str) and bool(v)`）→ **schema 拒绝空 postfix**。
  - → 空 postfix 的合法产物**永远无法通过评分**，是生产者与消费者约束不一致的真 bug（pred_ruler 默认路径被 schema 拒收）。
- **修复**：`yarn_receipt.py` pred_postfix 校验放宽为 `str 可空`（`isinstance(v, str)`），`method`/`t` 仍非空；同步更新两处 docstring。用真实 quest 回执验证 `validate_producer_receipt` 通过。
- **验证**：重跑打分成功（见 §3）。E119 test 套件跑出 8 个既有失败（P1-P7/B5，`install_stubs` lambda 2 参数 vs 生产 `register_patch` 3 参数 = 测试桩过期），与本修复无关，未扩大改动面。

## 3. RULER quest 32K 打分完成（AVG 42.85）

- 11/11 任务 n=100 全格（min-samples 硬门禁过），产物正式落盘 `exp/trace/results/e119_ruler32k_formal_quest_vendor.json`（含 manifest/receipt/派生 pred_root）。
- 逐格：niah_single_1 41 / single_2 96 / single_3 36 / multikey_1 87 / multikey_2 44 / multikey_3 0 / multiquery 20 / multivalue 32 / cwe 44.6 / fwe 18 / vt 52.8。
- **对拍 e116e 32K 锚点**（fullkv 59.38 / mavg 60.0 / aavg 57.35）：**Quest Δ≈-16.5**。检索大掉点
  （single_1 41 vs 100、single_3 36 vs 100、multikey_2 44 vs 76、cwe 44.6 vs 83.5）、长文检索
  （multikey_3 0/4、fwe 18 vs 7、vt 52.8 vs 34.2 反而小升）——**趋势符合 Quest 论文**（token_budget=1024
  在 32K 上下文只选 1K token，检索任务大掉点）。

## 4. GPU 盘点与在飞

| 机器 | 状态 |
|---|---|
| 本地 | GPU0/1 闲（本地 32B quest narrativeqa/qasper/passage_count 已完成） |
| .187 | 8/8 打满（32B quest repobench-p/gov_report + 32B fullkv + RULER quest 64K GPU4） |
| .24 | 8/8 打满（32B quest gov_report/qmsum + 8B lbv2 quest + RULER quest 64K GPU6/7） |
| .251 | 2/2 打满（RULER quest 128K niah_single_1/2，--yarn factor 4.0） |

- 32B quest 15/16 完成，剩 gov_report 双跑（.24 GPU1 + .187 GPU3）预计 1-2h → 16/16 合并打分。
- 前轮发现的 .251 pred_ruler/sparse_attn/quest_patch 三文件漂移已 rsync 修复，128K 派单正常。

## 5. 下一步

1. 32B quest gov_report 完成 → 合并打分（AVG 与 fullkv 43.28 对拍）。
2. RULER quest 64K 完成 → rsync 回本地 + score_ruler_formal 打分对拍 e116e 64K 锚点。
3. RULER quest 128K（.251，~3h）完成 → 同流程。
4. 主表拼接：8B quest 41.35/fullkv 49.39 + LBv2 三件套 + DeepSeek 40.55 + 32B quest/fullkv + RULER 三件套。

— by deepseek（advice 清理 + 空 postfix bug 修复 + RULER quest 32K 打分 AVG 42.85 + S-T023.md/task-chain.md 更新 + a065770 已 push）
