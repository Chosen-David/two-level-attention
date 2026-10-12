# twolevel GPU 监督 round3（by deepseek）

- 时间：2026-10-12 05:1X
- 范围：S-T023 竞品复现任务链（Quest vendor 8B/32B + DeepSeek fullkv 锚点）
- 结论：**无新 bug；.187 GPU0/1 空闲已派 32B quest 2 任务（repobench-p/passage_retrieval_en）；Qwen3-32B 权重 rsync → .24 后台进行（下一轮分流 32B quest 到 .24 空 5 卡）**。

## 1. GPU 占用表（四机，2026-10-12 05:1X）

| 机 | GPU 状态 | 进程 |
|---|---|---|
| 本地 | GPU0/1 空闲（143GB，0 占用） | 无（本地仅 DeepSeek-V2-Lite-Chat 模型，无 8B/32B） |
| .187 | GPU0/1 空闲；GPU2-7 占用（19-106GB） | 32B quest 5 队列（hotpotqa/musique/multi_news/samsum/lcc，t=035844）+ **本轮新派 GPU0/1（repobench-p/passage_retrieval_en，t=050706）** + 8B 任务（E116b，epoch TS） |
| .24 | GPU0/1/2/3/7 空闲；GPU4/5/6 占用（35-83GB） | Opus DeepSeek 043751 重跑 3 任务（multi_news/gov_report/qmsum） |
| .251 | GPU0/1 空闲（143GB） | 无（DeepSeek 005629 批 0 样本已弃，无残留） |

## 2. 在飞任务与脚本

- **DeepSeek fullkv（.24，Opus 接管）**：TS 043751 chat 11 格 **7 完成**（2wikimqa/hotpotqa/multifieldqa_en/musique/narrativeqa/passage_retrieval_en/qasper）+ 3 在跑（multi_news/gov_report/qmsum）+ 1 队列（passage_count）；5 非 chat 有效格保留。
- **32B quest（.187）**：multifieldqa_en 完成 150 行（info.py 修复验证 ✓）；5 卡队列 + GPU0/1 新派共 7 任务在跑。
- **8B 任务（.187）**：E116b 相关，epoch TS，正常推进。

## 3. Bug 审查结论

- 无新 bug。32B quest 重派后 pred 写文件正常（multifieldqa_en 150 行完成，无 ValueError 白跑）；DeepSeek 043751 无 DynamicCache 崩溃；无残留作废进程（上轮已清理 qmsum 023927 + 2 卡脚本树）。

## 4. 空闲 GPU 派单记录（本轮动作）

- **.187 GPU0/1** → 派 32B quest 队列最靠后 2 任务（repobench-p/passage_retrieval_en），独立目录 `pred_qv_1024_16_g01/` 防与原队列双写，完成后 mv 合并。脚本 `/tmp/dispatch_quest_32b_gpu01_187.sh`（TS 050706），已确认启动（PID 2218218/2218219）。
- **Qwen3-32B → .24 rsync**（后台 bxgs0k70t）：.24 空 5 卡（GPU0/1/2/3/7）下一轮分流 32B quest 剩余任务，大幅加速主线。
- **本地/.251 未派**：本地仅 DeepSeek（被 Opus 覆盖）；.251 无任务需求。不强行占用。

## 5. 下一步建议

1. rsync 完成后：.24 GPU0/1/2/3 派 32B quest 剩余（qmsum/gov_report/trec/triviaqa/2wikimqa/passage_count 中与 .187 队列错开分配），实现 32B quest 14 任务多机满卡并行。
2. DeepSeek 11 chat 格齐后：eval 打分（16K treatment 口径）+ 主表拼接（quest 8B 41.35 + fullkv 锚点）。
3. ClusterKV A 轨（accuracy 路径）依赖评估；FASA/SparQ 接入排期。

— by deepseek（四机 nvidia-smi + ps 实查；32B quest 派单验证启动；rsync 后台进行）
