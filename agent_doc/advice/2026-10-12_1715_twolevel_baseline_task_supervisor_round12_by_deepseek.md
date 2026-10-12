# twolevel 基线任务监督 round12（by deepseek）

- 时间：2026-10-12 17:1X
- 范围：round12 例行审查 + **.24 机器删除事件应急处置**
- 结论：**.24（33.32.33.24）用户确认已删除、SSH 不可达；唯一实质损失 = 128K quest 4 任务（~80% 时丢失），已立即重派 .187 GPU0/1/2/3（~20:1X 全齐）；其余已打分产物（32B quest / LBv2 / RULER 32K、64K）盘点全在本地，零损失；32B quest 16/16 收官确认。**

## 1. .24 机器删除事件（本轮核心）

- 用户报告 Host wys2-8xh20（=33.32.33.24）不可用、已删除；实测 `SSH No route to host` 确认网络层不可达。
- **影响评估（本地产物盘点）**：
  - ✅ 32B quest 053150 批 5 格（2wikimqa/gov_report/qmsum/trec/triviaqa）已在本地 `pred_qv_1024_16_24` + `quest_merged`；
  - ✅ LBv2 quest 8B（30.22）已在本地 `quest_lbv2_1024_16`；
  - ✅ RULER quest 32K（12 格）/ 64K（11 格）产物全在本地（已打分）；
  - ❌ **唯一损失 = 128K quest 4 任务**（multivalue/cwe/fwe/vt，.24 GPU0-3 在跑至 ~80%，partial gen 目录随机器删除不可恢复）。
- **教训**：关键产物随产随 rsync 回本地（本案例 32B/LBv2/32K/64K 均因已回收零损失）；在飞任务所在机器被删 = 任务 + 产物全丢。

## 2. 128K quest 恢复 11/11（重派 .187）

- 本地已有 7 格（single_1/2/3 + multikey_1/2/3 + multiquery，均有 .gen + yarn_receipt + .tli_gen 指针）。
- 重派 4 任务到 **.187 GPU0/1/2/3**（multivalue/cwe/fwe/vt）：
  - pred_ruler md5 **bb3e1b9a** 与本地一致核对 ✓、qwen3_quest_patch 09bfe078 ✓、模型+数据齐 ✓；
  - `--yarn factor 4.0`，TS 20261012-171155，PID 2375750，17:11 启动 1/100（~108s/it → **~20:1X 全齐**）。
- .187 当时 8 卡全空（含 E130 GPU5/6 也空）、.251 2 卡空——资源充足。

## 3. 32B quest 16/16 收官确认（qmsum 尾格）

- qmsum 035844 尾格 **200/200 行 DONE rc=0（15:52）**，产物落盘 `.187 exp/results_longbench_vendor/Qwen3-32B/pred_qv_1024_16/pred/qmsum-quest_vendor_1024_16-20261012-035844.jsonl`。
- 32B quest 16/16 产物全齐（本地 narrativeqa/qasper/passage_count + .187 035844 批 9 + g01 批 2 + .24 053150 批 5，已合并打分 **AVG 38.08**）。

## 4. 在飞任务核实（无失败无 bug）

- .187 重派 4 任务正常起步（1/100，~108s/it）；无 ValueError/崩溃。
- .187/.251 可达正常；.24 已确认删除。

## 5. 下一步

1. 128K quest 4 格全齐（~20:1X）→ rsync 回本地 → 11 格打分（--yarn-factor 4.0，staging /tmp/e119_128k_quest）对拍 e119 128K 锚点（fullkv 46.43/mavg 47.49/aavg 42.56）→ 填主表 128K 行。
2. 主表收尾（main_table_quest_vs_tli_draft.md），标注 quest legacy cap null。
3. ClusterKV 接入工作包仍排期（pylibraft 24.10 版本障碍 + gcc-11/pybind/flashinfer/kernel build/qwen3 适配）。

— by deepseek（.24 机器删除应急处置：128K 4 格重派 .187 + 全量本地产物盘点零损失；32B quest 16/16 收官）
