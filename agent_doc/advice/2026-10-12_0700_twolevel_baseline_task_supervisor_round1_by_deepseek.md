# twolevel baseline 任务监督 round1（by deepseek）

- 时间：2026-10-12 06:5X
- 范围：S-T023 开源 baseline 复现任务链（Quest vendor 32B + DeepSeek fullkv 锚点）
- 结论：**32B quest 16/16 任务全覆盖核实通过；发现 3 个任务双跑（gov_report/passage_count/repobench-p，086 best-file 决策处理无害）；DeepSeek 043751 剩 2 格在跑。无卡死/失败进程。**

## 1. 32B quest 16/16 全覆盖核实（本轮主动作）

| 覆盖源 | 任务 | 状态 |
|---|---|---|
| 本地（TS 002743）| narrativeqa / qasper | 完成 |
| 本地（TS 053753）| passage_count | 在跑 |
| .187 rerun 批（TS 035844，5 卡串行）| multifieldqa_en / musique / samsum / lcc | 完成 4 |
| .187 rerun 批（同批在跑）| hotpotqa / gov_report / multi_news / passage_count / repobench-p | 在跑 5 |
| .187 g01 批（TS 050706）| passage_retrieval_en / repobench-p | 在跑 |
| .24 重派（TS 053150）| 2wikimqa / gov_report / qmsum / trec / triviaqa | 在跑 5 |

- 16 任务 = narrativeqa/qasper/multifieldqa_en/musique/samsum/lcc/hotpotqa/multi_news/
  gov_report/passage_count/repobench-p/passage_retrieval_en/2wikimqa/qmsum/trec/triviaqa
  **全部有覆盖，无遗漏。**

## 2. 认知修正：.187 rerun 批实际覆盖 9 任务（非我此前假设的 5 任务）

- 源码核验 `/tmp/qv_lb_v1_32b_rerun_20261012-035844/card*.log`：
  - card2: multifieldqa_en(done) → hotpotqa(running)
  - card3: musique(done) → gov_report(running)
  - card4: multi_news(running)
  - card6: samsum(done) → passage_count(running)
  - card7: lcc(done) → repobench-p(running)
- **影响**：round4 我把 gov_report 派到 .24（053150）、本地补派 passage_count（053753）、
  g01 派 repobench-p（050706）→ 与 .187 rerun 批**3 任务双跑**。
- **判定无害**：双跑输出目录不同（.24→pred_qv_1024_16_24/、本地/主批→pred_qv_1024_16/、
  g01→pred_qv_1024_16_g01/）+ TS 不同（文件名含 TS 不覆盖）→ 合并时 086 协议
  SKIP 前缀归一 + AMBIGUOUS 检测 + best-file（行数门 200）决策处理。仅浪费算力，不污染数据。
- 教训：**派单前必须看批脚本的每卡任务序列（card*.log 或脚本），不能只按"批名+几卡"假设覆盖**。

## 3. DeepSeek fullkv 043751（.24，Opus 接管）进度

- **9/11 chat 格完成**（2wikimqa/hotpotqa/multifieldqa_en/multi_news/musique/narrativeqa/
  passage_retrieval_en/qasper/qmsum，pred 均 200 行）。
- **2 在跑**：gov_report（PID 3935492，~57min）+ passage_count（PID 3971607，~12min）。
- 5 非 chat 有效格（trec/triviaqa/samsum/lcc/repobench-p，TS 023927）保留 → 11 格齐后 eval 打分
  （16K treatment 口径 MLA dense 锚点）。

## 4. GPU 健康与下一步

- .187 8 卡 util 62-100% 全在算；.24 GPU0-3/7 32B quest 全在算 + GPU4/5/6 DeepSeek；
  本地 GPU0 passage_count 94%。无卡死/失败进程。
- 下一步：
  1. 32B quest 各批完成 → 合并 pred_qv_1024_16{,_g01,_24}/ → 086 best-file → eval 打分
     + 空卡贴合（对拍 Quest 论文检索掉/摘要维持趋势，sanity 硬门禁）。
  2. DeepSeek 043751 剩 2 格齐 → eval 打分（16K treatment 口径）。
  3. 主表拼接：quest 8B 41.35 + fullkv 锚点 + DeepSeek + 32B quest。

— by deepseek（四机 nvidia-smi + ps 实查；.187 rerun 批 card*.log 逐卡源码核验；32B quest 覆盖矩阵整理；DeepSeek pred 行数实查）
