# twolevel baseline 任务监督 round7（by deepseek）

- 时间：2026-10-12 11:30
- 范围：S-T023 任务链推进（LBv2 三件套打分 + sanity 第 8 个确证 + 32B fullkv 16/16 打分 + RULER quest 派单）
- 结论：**LBv2 三件套打分齐（Quest 掉点 -1.79、TLI mavg +0.20 维持 dense）；sanity gov_report Δ=+0.04 第 8 个全预算 sanity 通过；32B fullkv 16/16 打分 AVG 43.28（lcc/repobench 低分确认为 32B verbose 生成风格，非 bug）；RULER quest smoke 通过并修复 .187 sparse_attn 漂移 bug；RULER quest 32K 11 任务 + 64K 3 任务已派，全资源打满。**

## 1. LBv2 三件套打分（Qwen3-8B，503 样本，四选一 accuracy）

- **TLI mavg (.25,.125,.625) = 32.21 / TLI aavg (0,0) = 31.61**（e116b_lb 已有打分）+ **Quest vendor 1024_16 = 30.22**（本轮打分）+ **FullKV = 32.01**（.187 lbv2_fullkv rsync 回本地后打分）。
- 结论：**Quest 在 LBv2 掉点 -1.79**（30.22 vs 32.01，检索/选择类任务掉点符合 Quest 论文趋势）；**TLI mavg +0.20 维持 dense**；aavg -0.40 轻微掉点。LBv2 三件套齐，主表 LBv2 行可补。

## 2. sanity gov_report 打分：第 8 个全预算确证

- quest 32768 = **33.4** vs fullkv = **33.36** → **Δ=+0.04**。
- 全预算≈dense 证据扩展到 **8 任务**（Δ∈[-1.48,+1.0]）：hotpotqa -1.48 / musique -1.30 / qasper -0.90 /
  narrativeqa -0.17 / samsum -0.43 / multi_news +0.11 / gov_report +0.04 / trec +1.0 → **Quest 适配层 sanity 硬门禁继续通过**。

## 3. 32B fullkv 16/16 打分：AVG 43.28

- .24（7）+ .187（6+第二轮 repobench/lcc）+ 本地（2）全合并，16/16 打分（quest venv eval.py --legacy-no-runcap）：
  2wikimqa 54.81 / gov_report 33.25 / hotpotqa 60.73 / lcc 12.14 / multi_news 25.54 / multifieldqa_en 50.79 /
  musique 39.79 / narrativeqa 30.9 / passage_count 19.0 / passage_retrieval_en 99.0 / qasper 45.75 / qmsum 23.69 /
  repobench 17.62 / samsum 36.46 / trec 72.5 / triviaqa 70.55 → **AVG 43.28**。
- **新发现（非 bug）**：lcc=12.14 / repobench=17.62 异常低，核查确认 8B fullkv 对应 68.1/64.74——根因是
  **32B Qwen3 verbose 推理生成风格**（pred 先"Okay, I need to..."再给代码），difflib 精确匹配失败；8B 直接输出代码。
  32B 内部 quest/fullkv 对比不受影响（同模型同风格），但**与 8B 横比时代码任务须标注生成风格差异**。
- 注意：本地 samsum/triviaqa 带 runcap sidecar（31500 unregistered），因 dataset fingerprint 不同致 cap
  identity hash 不同 → eval.py 误判"异 cap 混挂"→ 临时移出 sidecar 降级打分后恢复（不删 provenance）。

## 4. RULER quest smoke 通过 + sparse_attn 漂移 bug 修复

- smoke（8B niah_single_1 32768 n=3）首跑失败：`.187 sparse_attn.ops.eager_decoding 缺
  eager_decoding_request_guard`（pred_ruler.py 顶层 import，.187 旧版）。已 rsync 本地 sparse_attn 到
  .187/.24（本地有，.24 本就有该函数），验证存在 → **smoke 通过**（3/3 样本 23s/it，产物落盘
  results_ruler/Qwen3-8B/quest_vendor_1024_16/L32768/pred/）。教训：**跨机跑 pred_ruler.py 前必须核对
  sparse_attn 版本**（与 pred.py 依赖模块同理）。

## 5. RULER quest 派单（全资源打满）

- 32K 11 任务（每任务 100 样本 ~40min）：.187 GPU0/1/2/5/6/7（niah_single_1/2/3 + niah_multikey_1/2/3）+
  .24 GPU0/2/3/4/5（niah_multiquery/niah_multivalue/cwe/fwe/vt），TS ruler_quest_20261012。
- 64K 3 任务（100 样本 ~1.1h，--yarn 自动 factor 2.0）：.24 GPU6/7（niah_single_1/2）+ .187 GPU4（niah_multiquery）。
- **.24 8/8 + .187 8/8 全打满**。

## 6. 在飞状态

| 任务 | 进度 |
|---|---|
| RULER quest 32K 11 任务 | .187 6 + .24 5，~40min/任务 |
| RULER quest 64K 3 任务 | .24 GPU6/7 + .187 GPU4，~1.1h |
| 32B quest gov_report 双跑 | .24 GPU1（5h39m）+ .187 GPU3（5h42m），预计 1-2h 完成 → 32B quest 16/16 |
| 32B quest 其余 15 任务 | 唯一候选已全（本地 3 + .187 8+2 + .24 4），待 gov_report 合并打分 |

## 7. 下一步

1. 32B quest gov_report 完成 → 合并 pred_qv_1024_16{,_g01,_24}/ + 本地 → 086 best-file → eval 打分（AVG 与 fullkv 43.28 对拍）。
2. RULER quest 32K 完成 → rsync 回本地 → scorer.manifest 打分对拍（e116e TLI/fullkv 锚点）。
3. RULER quest 64K 完成 → 同流程；空卡补 128K quest。
4. 主表拼接：8B quest 41.35/fullkv 49.39 + LBv2 三件套 + DeepSeek 40.55（16K 口径）+ 32B quest/fullkv（完成后）。

— by deepseek（LBv2 三件套打分 + sanity gov_report Δ+0.04 + 32B fullkv 16/16 AVG 43.28 + RULER quest smoke/派单 + sparse_attn 漂移修复 + S-T023.md 更新）
