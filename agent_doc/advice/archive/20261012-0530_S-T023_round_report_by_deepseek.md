# S-T023 对比基线 round 汇报（by deepseek）

- 时间：2026-10-12 05:3X
- 触发：前会话多个后台任务完成通知核验（.251 DeepSeek 派单 / DeepSeek 派单重启 / rsync / 下载等）
- 结论：**后台任务全部为历史动作已收尾，无冲突；发现并清理一处残留作废进程（02:06 派单的 qmsum 023927 占 GPU0）→ GPU0 释放；.24 Opus 043751 重跑 6 完成 + 4 在跑 + 1 队列，一切正常。**

---

## 1. 后台任务通知核验（全部历史动作，无当前冲突）

- `baycpw2x7` Qwen3 fullkv 4 卡派单（TS 020308）= 已收官的 16/16 AVG 49.39 批次 ✓
- `bt2smv2mz`/`b0g78ikuq`/`bb2ycyjyu` DeepSeek 派单（TS 020611 + 重启）= 031e6b8 版作废批次，已被 Opus 043751 覆盖
- `bal41ioyv` .251 DeepSeek 派单 = .251（10.238.139.251）现无进程、GPU 0、fullkv pred 空（005629 批 0 样本），干净
- `buktimgko`/`b5fvsm7ew` rsync 模型、`bx817i1un` transformers 下载 = 完成，无影响
- `b39ee2wya`/`bujg5gdl3` .187 32B 派单重启+验证 = 正常推进（multifieldqa_en 已完成 150 行）

## 2. 发现并清理残留作废进程（实质动作）

- **残留**：`dispatch_dsv2_fullkv_2card_24.sh`（02:06 的 2 卡派单，脚本树 3831653/3831654/3831659）仍在跑，
  当前任务 `PID 3936780 task=qmsum t=20261012-023927`（**占 GPU0**，CUDA_VISIBLE_DEVICES=0）。
- **危害**：qmsum 是 chat 格（094 双 BOS 作废），产出作废数据 + 浪费 GPU0 + 与 Opus GPU4 队列的
  043751 qmsum 撞车（mv 隔离时双写）。
- **处理**：kill 3936780 + 脚本树 3831653/3831654/3831659 → 验证无残留、GPU0 释放。
- **无副作用确认**：Opus 进程（3924752-3924760 脚本 + 3924814/3935492/3938142/3940781 四 pred）未受影响，正常存活。

## 3. .24 DeepSeek 043751 重跑进度（05:3X）

| 状态 | 任务 |
|---|---|
| 完成 6 | narrativeqa / multifieldqa_en / hotpotqa / musique / qasper / passage_retrieval_en |
| 在跑 4 | multi_news / gov_report / 2wikimqa / qmsum |
| 队列 1 | passage_count（GPU5，multi_news 完成后） |

- 5 非 chat 有效格（trec/triviaqa/samsum/lcc/repobench-p，TS 023927）在 pred/ 保留，可留。
- GPU0 现在真正空闲，留作重跑补位。

## 4. 待办

1. 等 043751 的 11 chat 格齐（剩 passage_count）→ 与 5 有效格合并 → eval 打分（16K treatment 口径）
2. .187 32B quest 14 任务继续（~20h）→ 合并本地 narrativeqa/qasper → 打分 + 空卡贴合
3. 主表拼接：quest 8B 41.35 + fullkv 锚点 + DeepSeek

— by deepseek（核验全部后台任务输出 + 三机进程/pred 目录实查 + 残留进程源码确认后 kill）
