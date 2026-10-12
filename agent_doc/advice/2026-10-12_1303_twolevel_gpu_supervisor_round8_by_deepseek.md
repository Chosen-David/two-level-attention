# twolevel 代码 bug 监督 round8（by deepseek）

- 时间：2026-10-12 13:03
- 范围：round8 例行审查（32B quest 16/16 合并打分 + RULER quest 64K 补派 + pred.py 跨机版本漂移发现）
- 结论：**32B quest 16/16 打分 AVG 38.08（对拍 32B fullkv 43.28 → Quest Δ≈-5.20，检索任务大掉点符合论文趋势）；.24 8 卡空闲 → 补派 RULER quest 64K 剩余 8 任务（补全 11 格对拍 e119 64K 锚点）；发现远程 pred.py（f7069942）缺 096 runcap sidecar 逻辑 = 跨机版本漂移，quest 8B/32B 产物全 legacy（cap 身份 null，主表拼接须标注）。**

## 1. 32B quest 16/16 合并打分（AVG 38.08）

- **16 任务全覆盖核实**：本地 3（narrativeqa/qasper/passage_count）+ .187 035844 批 9（2wikimqa/gov_report/hotpotqa/lcc/multifieldqa_en/multi_news/musique/samsum/passage_count）+ .187 g01 批 2（repobench/passage_retrieval_en）+ .24 053150 批 5（2wikimqa/gov_report/qmsum/trec/triviaqa）→ 合并三处产物 `quest_merged/pred/` 打分（quest venv eval.py，--legacy-no-runcap）。
- **逐格**：2wikimqa 42.88 / gov_report 32.54 / hotpotqa 47.38 / lcc 20.58 / multi_news 25.63 / multifieldqa_en 46.28 / musique 29.35 / narrativeqa 24.57 / passage_count 13.50 / passage_retrieval_en 87.50 / qasper 36.58 / qmsum 23.68 / repobench 31.91 / samsum 33.58 / trec 63.00 / triviaqa 50.34。
- **对拍 32B fullkv AVG 43.28（16 任务同口径）**：Quest Δ≈-5.20。检索大掉点（2wikimqa -11.93/hotpotqa -13.35/triviaqa -20.21/trec -9.5/passage_retrieval_en -11.5/narrativeqa -6.33）、摘要维持（multi_news +0.09/gov_report -0.71/samsum -2.88/qmsum -0.01）→ **趋势完全贴合 Quest 论文**（token_budget=1024 检索任务大掉点）。8B quest 41.35 vs fullkv 49.39 Δ=-8.03 对比，32B 掉点略小但同向。
- 注：lcc 20.58/repobench 31.91 = 32B verbose 生成风格（8B 对应 68.1/64.74），32B 内部对比（quest vs fullkv 同模型）不受影响。

## 2. RULER quest 64K 补派（.24 8 卡）

- **round7 遗留**：64K 只派 3 任务（single_1/2 + multiquery，已完成产物 tli_gen 指针齐）。e119 64K 锚点（fullkv/mavg/aavg）都是 **11 任务全格**（64K fullkv AVG≈48.6）→ quest 64K 需补全 11 格才公平对拍。
- **.24 8 卡全空**（32B quest gov_report + 64K 首 3 任务完成后）→ 补派剩余 8 任务（single_3/multikey_1/2/3/multivalue/cwe/fwe/vt），每卡一任务，--yarn 自动 factor 2.0，TS=20261012-124839，19/100 ~41s/it 预计 ~1h 完成。产物仍落 `L65536/pred/`（同 TS 体系，完成后 11 格打分对拍 e119 64K 锚点）。

## 3. pred.py 跨机版本漂移（发现，非 quest 路径 bug）

- **现象**：quest 8B/32B 产物全无 `.runcap.json` sidecar（096 cap provenance），打分只能 --legacy-no-runcap 降级（cap 身份 null）。
- **根因**：`.24 pred.py（md5 f7069942）grep runcap = 0`——远程 pred.py 是旧版（无 096/102/103 runcap 逻辑），本地 main 分支 pred.py 已含 runcap sidecar 全链。quest vendor 分支本身无问题。
- **影响范围**：32B fullkv 打分同样 legacy（round7 已知 samsum/triviaqa sidecar 异 cap 移出降级，全批实际也 legacy）→ quest vs fullkv 对比**同口径**，Δ=-5.20 有效。主表拼接须标注 quest 系列产物 cap 身份 null（legacy）。
- **处置**：近期无 LongBench 新派单（8B/32B quest/fullkv + LBv2 + sanity 全齐），不即时 sync（避免打断在飞 64K 进程 + 新依赖风险）；128K/64K 用 RULER yarn_receipt 体系不受影响。登记待办：下轮 LongBench 派单前必须先 rsync 本地 pred.py + 依赖（context_capability_gate/lbv2_choice/sparse_attn）到三机。

## 4. GPU 盘点

| 机器 | 状态 |
|---|---|
| 本地 | GPU0/1 闲（32B quest 已完成，RULER 打分用 CPU） |
| .24 | 8/8 打满（RULER quest 64K 补派 8 任务，12:48 起 ~1h） |
| .187 | GPU3 32B quest qmsum 在跑（035844 批尾），GPU0/2/5/6/7 其他任务占用，GPU4 空 |
| .251 | 2/2 打满（RULER quest 128K niah_single_1/2，11:29 起 ~3h → 预计 14:3X） |

## 5. 下一步

1. RULER quest 64K 补派完成（~14:0X）→ rsync 回本地 + 11 格打分对拍 e119 64K 锚点（fullkv 48.6/mavg/aavg）。
2. .187 qmsum 完成 → 32B quest 尾格落盘（双跑 053150 已有，无害）。
3. RULER quest 128K 完成（~14:3X）→ 同流程（.251 产物 rsync 回本地，注意 128K 锚点 11 格 → 需补派剩余 9 任务到空卡）。
4. 主表拼接：8B quest 41.35/fullkv 49.39 + 32B quest 38.08/fullkv 43.28 + LBv2 三件套 + DeepSeek 40.55 + RULER 三件套（标注 quest legacy cap）。

— by deepseek（32B quest 16/16 打分 Δ≈-5.20 + 64K 补派 + pred.py 版本漂移发现）
