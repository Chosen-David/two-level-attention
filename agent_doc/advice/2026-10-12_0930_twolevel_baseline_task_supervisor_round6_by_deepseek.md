# twolevel baseline 任务监督 round6（by deepseek）

- 时间：2026-10-12 09:30
- 范围：S-T023 任务链推进（32B fullkv 全覆盖 + sanity 扩展 + RULER quest 开发）
- 结论：**32B fullkv 16/16 全覆盖在跑（第一轮 qasper/multifieldqa_en 完成 → 第二轮 4 任务补派到空卡）；multi_news 全预算 sanity 打分 Δ=+0.11 → 全预算≈dense 证据扩展到 7 任务；pred_ruler.py quest_vendor 分支开发完成并 sync 远程。**

## 1. 32B fullkv 16/16 全覆盖（本轮主动作）

- 09:14 派第一轮 12 卡（.24 5 + .187 5 + 本地 2）；09:28 发现 .24 GPU3/4 空出（qasper/multifieldqa_en
  已完成，短输入任务 15min 完成）→ **立即第二轮补派 4 任务**：
  - .24 GPU3 passage_count + GPU4 passage_retrieval_en
  - .187 GPU2 lcc + GPU4 repobench-p（multi_news sanity 完成后空出的卡）
- 验证全部启动（GPU 占满）：**32B fullkv 16/16 全覆盖**。

## 2. multi_news 全预算(32768) sanity：第 7 个确证

- .187 完成 → rsync 回本地 → quest venv eval.py 降级打分（--legacy-no-runcap，f7069942 产物无 sidecar）：
  **quest 32768 = 24.89** vs fullkv = 24.78 → **Δ=+0.11**（摘要维持）。
- 全预算≈dense 证据扩展到 **7 任务**（Δ∈[-1.48,+1.0]）：
  hotpotqa -1.48 / musique -1.30 / qasper -0.90 / narrativeqa -0.17 / samsum -0.43 /
  multi_news +0.11 / trec +1.0 → **Quest 适配层 sanity 硬门禁继续通过**。

## 3. RULER quest 开发（commit 85eba1f）

- pred_ruler.py 加 quest_vendor 分支：B 轨官方 Quest 绑定（enable_quest，qwen3_quest_patch
  09bfe078 32B 修复版），--quest_token_budget/--quest_chunk_size，patched=0 fail-closed。
- 已 sync .24/.187（md5 bb3e1b9a 一致）；RULER 数据在
  sparse-bench/third_party/KVCache-Factory/data/RULER（两机有）。待远程空卡 → smoke → 派 RULER quest。

## 4. 在飞状态

| 任务 | 进度 |
|---|---|
| 32B quest gov_report 双跑 | .24 GPU1 + .187 GPU3，~4h+，预计 1-2h 完成 → 16/16 |
| 32B fullkv 16 任务 | 全部在跑（第一轮 4-5h）|
| 8B lbv2 quest（.24 GPU0）| 2h+，503 样本 |
| 8B sanity gov_report（.24 GPU6）| 2h+ |

## 5. 下一步

1. 32B quest gov_report 完成 → 合并 pred_qv_1024_16{,_g01,_24}/ + 本地 → 086 best-file → eval 打分。
2. 32B fullkv 第一轮完成 → 回收；LBv2 quest 完成 → LBv2 三件套打分。
3. sanity gov_report 完成 → 打分（第 8 个全预算 sanity）。
4. RULER quest：空卡后 smoke 验证 → 派三件套。

— by deepseek（32B fullkv 第二轮补派 + multi_news sanity 打分 Δ+0.11 + pred_ruler quest_vendor 开发/sync + S-T023.md 更新）
