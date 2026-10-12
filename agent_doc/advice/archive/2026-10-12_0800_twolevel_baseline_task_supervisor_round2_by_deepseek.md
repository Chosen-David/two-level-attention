# twolevel baseline 任务监督 round2（by deepseek）

- 时间：2026-10-12 08:0X
- 范围：S-T023 开源 baseline 复现任务链
- 结论：**DeepSeek-V2-Lite-Chat fullkv 16/16 打分完成（AVG 40.55，16K treatment 口径 MLA dense 锚点）——里程碑：DeepSeek 整臂收官可上主表；32B quest 9/16 完成，7 在跑推进正常。无卡死/失败。**

## 1. DeepSeek fullkv 16/16 收官（本轮主动作）

- **11 chat 格（TS 043751）全部完成**（均 200 行），5 非 chat 有效格（TS 023927：trec/triviaqa/samsum/lcc/repobench）保留。
- **隔离作废格**：023927 批的 gov_report（chat 格双 BOS 作废）mv → pred_quarantine_dsv2_094/，pred/ 剩 16 格唯一候选无 AMBIGUOUS。
- **打分**：本地 quest venv（jieba 有、rouge 补装成功）eval.py 打分完成，result.json 落盘。
- **结果（AVG 40.55）**：
  - 2wikimqa 20.07 / gov_report 33.67 / hotpotqa 23.91 / lcc 61.6 / multi_news 25.79 /
    multifieldqa_en 47.87 / musique 11.48 / narrativeqa 22.76 / passage_count 5.72 /
    passage_retrieval_en 73.5 / qasper 31.27 / qmsum 23.98 / repobench 62.62 / samsum 44.52 /
    trec 75.5 / triviaqa 84.52
- **口径声明**：16K 受限上下文 treatment（[EXPERIMENT-CAP-DIVERGENCE]），MLA dense 锚点；
  与 Qwen3 fullkv（31500 口径 AVG 49.39）**不横比**，跨模型线独立陈述（S-T023 协议）。
- 注：eval 依赖（jieba/rouge）本地/远程均缺 → quest venv 补装 rouge 解决。

## 2. 32B quest 进度（9/16 完成，7 在跑）

| 源 | 完成 | 在跑 |
|---|---|---|
| 本地（002743/053753）| narrativeqa / qasper | passage_count |
| .187 035844 rerun | multifieldqa_en / musique / samsum / lcc / **hotpotqa（新完成）** | multi_news / gov_report / passage_count / repobench-p |
| .187 g01（050706）| **passage_retrieval_en（新完成）** | repobench-p |
| .24 053150 重派 | — | gov_report / qmsum / trec / triviaqa / 2wikimqa |

- 待办：剩余 ~7 格完成后合并 pred_qv_1024_16{,_g01,_24}/ → 086 best-file（3 双跑）→ eval 打分 + 空卡贴合。

## 3. GPU 健康

- .24：GPU0/4/5/6 空闲（DeepSeek 收官释放），GPU1/2/3/7 跑 32B quest。
- .187：8 卡 util 61-100% 全在算（32B quest + 8B TLI E129 混跑）；GPU1 空闲。
- 本地：GPU0 跑 passage_count，GPU1 空闲。无卡死/失败进程。

## 4. 下一步

1. 32B quest 剩余格完成 → 合并 + 086 best-file → eval 打分 + 空卡贴合（Quest 论文趋势 sanity 硬门禁）。
2. 主表拼接：quest 8B 41.35（已完成）+ fullkv 锚点 49.39 + DeepSeek 40.55 + 32B quest（完成后）。
3. ClusterKV A 轨（cavg/ccluster 公平实现）依赖评估；FASA/SparQ 接入排期。

— by deepseek（.24 DeepSeek pred 目录实查 + 作废格隔离 + 本地 quest venv 补 rouge 打分 + 逐格分数记录）
