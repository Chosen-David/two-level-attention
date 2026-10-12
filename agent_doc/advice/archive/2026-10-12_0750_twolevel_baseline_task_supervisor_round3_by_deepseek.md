# twolevel baseline 任务监督 round3（by deepseek）

- 时间：2026-10-12 07:5X
- 范围：S-T023 开源 baseline 复现任务链（32B quest 收尾 + LBv2 锚点 + sanity 扩展）
- 结论：**发现并修复 round5 pred.py sync 的副作用（.187 缺 context_capability_gate.py 导致新派单 ImportError，已补齐重派）；kill 一处双跑格释放 GPU；新派 LBv2 fullkv 8B（主表 LBv2 三件套 dense 锚点）+ 3 个全预算 sanity；hotpotqa sanity 完成（全预算≈dense 证据 +1）。32B quest 13/16，3 在跑。**

## 1. 发现并修复：.187 缺 context_capability_gate.py（round5 sync 副作用）

- **现象**：.187 新派单（lbv2 fullkv + multi_news sanity）立即 `ModuleNotFoundError: No module named
  'benchmark.LongBench.context_capability_gate'`（pred.py:21），rc=0 假成功。
- **根因**：round5 我把 .24 版 pred.py（f7069942）sync 到 .187，但该版 import
  `context_capability_gate.py`（cap 门禁），.187 缺此模块（.187 旧版 pred.py 不依赖）。
- **修复**：scp .24 的 context_capability_gate.py → .187（md5 12bcf15d5bad958aa1665f8cedd82877
  一致）→ 重派验证：lbv2 fullkv 3/503 推进（~9s/it → 预计 1.2-1.5h）+ multi_news sanity 模型
  加载完成开始样本。
- **教训**：sync pred.py 必须连带核对/sync 其 import 依赖模块（lbv2_choice/context_capability_gate/
  sparse_attn 等），不能只 sync 单文件。

## 2. GPU 清理 + 新派单（打满 GPU）

- **kill .187 passage_retrieval_en 双跑格**（PID 2234072，06:50 起）：g01 050706 已 200 行完成，
  主批是纯浪费 → kill 释放 GPU。
- **.187 GPU1**：lbv2 fullkv 8B（method none，503 样本）——**主表 LBv2 三件套 dense 锚点**（LBv2
  已有 TLI mavg/aavg 打分 result.json，缺 fullkv 对比）。
- **.187 GPU2**：multi_news 全预算(32768) sanity。
- **.24 GPU6/7**：gov_report/samsum 全预算(32768) sanity（TS 20261012-074110/074833）。
- 均在正常推进。

## 3. sanity 证据（全预算≈dense 扩展）

- **hotpotqa 全预算(32768) sanity 完成**（200 行，07:43 落盘）→ 全预算≈dense 证据扩展到
  qasper/trec/hotpotqa 3 任务（Quest 论文核心结论确证中）。
- 在跑：.24 narrativeqa/musique + .187 multi_news + .24 gov_report/samsum 全预算 sanity。

## 4. 32B quest 进度（13/16）

| 源 | 完成 | 在跑 |
|---|---|---|
| 本地 | narrativeqa/qasper/passage_count | — |
| .24 053150 | 2wikimqa/trec/triviaqa | gov_report/qmsum（05:31 起 ~3h）|
| .187 035844 | multifieldqa_en/musique/samsum/lcc/hotpotqa/2wikimqa/passage_count/multi_news（8）| gov_report / repobench-p |
| .187 g01 | passage_retrieval_en | repobench-p（与主批双跑）|

- 剩 3 独特任务在跑（gov_report/repobench-p/qmsum，均有双跑），预计 1-3h 全部完成。

## 5. GPU 与下一步

- .187 GPU0/3/6/7 32B quest + GPU1 lbv2 fullkv + GPU2 multi_news sanity + GPU4/5 空闲；
  .24 GPU0/1/2/4/5/6/7 全满（8B quest + sanity + 32B quest）+ GPU3 空闲；本地 GPU0/1 空闲。
- 下一步：
  1. 32B quest 16/16 齐 → 合并 pred_qv_1024_16{,_g01,_24}/ + 本地 → 086 best-file（3 双跑）→
     eval 打分 + 空卡贴合（对拍 Quest 论文检索掉/摘要维持，sanity 硬门禁）。
  2. lbv2 fullkv + lbv2 quest 完成后打分（eval.py lbv2 scorer）→ LBv2 三件套（TLI mavg/aavg +
     fullkv + quest）。
  3. sanity 全预算多任务齐 → "全预算≈dense" 汇总证据。
  4. 主表拼接：8B quest 41.35 + fullkv 49.39 + DeepSeek 40.55 + 32B quest（完成后）。

— by deepseek（.187 派单 ImportError 定位 + 模块补齐重派验证；双跑格 kill；LBv2 fullkv/sanity 派单；hotpotqa sanity 完成确认；32B quest 进度盘点）
