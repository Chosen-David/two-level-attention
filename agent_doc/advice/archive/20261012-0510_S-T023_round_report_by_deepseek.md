# S-T023 对比基线 round 汇报（by deepseek）

- 时间：2026-10-12 05:1X
- 触发：调查 DeepSeek pred 目录文件"消失"事件（04:37 出现 pred_quarantine_dsv2_094，文件从 13 掉到 8）
- 结论：**非事故**——是 Opus 093/094 修复派单脚本的有意隔离（094 双 BOS 作废格），Opus 已代跑重派 TS 043751（GPU1-6 六链），各格正常推进；.187 32B quest 重派第一格 multifieldqa_en 完成（150 行），5 卡推进正常，无白跑。

---

## 1. quarantine 事件澄清（不是数据损坏）

- **现象**：DeepSeek fullkv pred/ 出现 `pred_quarantine_dsv2_094/`（04:37），6 个 023927 TS 文件被移入（hotpotqa/multi_news/musique/narrativeqa/qasper/passage_retrieval_en），pred/ 剩 8 文件。
- **根因**（源码核验，非臆测）：Opus 派单脚本 `/tmp/dsv2_fullkv_chat_rerun.sh` 的 `run_arm` 函数在每个 task 启动前：
  `mv "$OUTDIR"/${P}-none-*.jsonl → pred_quarantine_dsv2_094/`（排除当前 TS），
  即**隔离"即将被重跑覆盖的同前缀旧 TS 作废文件"**。被隔离的 6 个 = 第一波已启动的 6 任务（GPU1-6 各卡首个）；第二波（multifieldqa_en/2wikimqa/qmsum/passage_count/gov_report）启动时逐个隔离。
- **隔离内容核查**：6 文件内容完整（有真实 pred 输出），非空非损坏——是有意隔离的作废格。
- **qasper 无重跑进程却已隔离**：GPU2 队列 qasper 已完成（0/150→98/150 是下一个 multifieldqa_en），隔离发生在它启动时。逻辑一致。

## 2. DeepSeek fullkv 污染面（Opus 实查，efabec6 修复合入）

- **094 bug**：chat template 路径 encode 缺 `add_special_tokens=False` → 双 BOS `[BOS,BOS,User:...]` 偏离官方 prompt（红证：首两 token [100000,100000]）。
- **093 bug**：三方能力值不一致，裁决 16K 受限上下文 treatment（model2maxlen=16384，eager OOM 实证；协议值 31500 留 flash 路线锚），gate 打 [EXPERIMENT-CAP-DIVERGENCE]。
- **11 个 chat 格作废**（narrativeqa/qasper/multifieldqa_en/hotpotqa/2wikimqa/musique/gov_report/qmsum/multi_news/passage_count/passage_retrieval_en）→ 已由 Opus 重派重跑。
- **5 个非 chat 格有效可留**（trec/triviaqa/samsum/lcc/repobench-p）：不走模板恰一个 BOS，16K cap 与裁决后 treatment 同口径，不重跑。
- 我上轮并行抢派的 samsum/lcc/repobench-p 是有效格（已收尾）；passage_count/passage_retrieval_en 是作废格（已被重跑覆盖）。

## 3. Opus 接管重跑状态（TS 043751，.24 GPU1-6）

| GPU | 队列 | 当前进度（05:1X） |
|---|---|---|
| 1 | narrativeqa | 126/200，4-5s/it |
| 2 | qasper→multifieldqa_en | multifieldqa_en 98/150（qasper 已完成） |
| 3 | hotpotqa→2wikimqa | 135/200 |
| 4 | musique→qmsum | 109/200 |
| 5 | multi_news→passage_count | 35/200（16s/it 慢，长文档） |
| 6 | passage_retrieval_en→gov_report | 184/200（2.8-3s/it 快） |

- generate 路径无 DynamicCache 崩，[EXPERIMENT-CAP-DIVERGENCE] + [CONTEXT-CAP-GATE] 生效，16K cap 口径运行。
- **GPU0 空闲**：我的并行脚本已全部收尾（repobench-p 04:30 完成）。**未擅自派新任务到 GPU0**——避免与 Opus GPU1-6 队列撞车双写产生 AMBIGUOUS 候选（086 协议）；留作重跑失败时的补位。
- GPU7 旧 gov_report（TS 023927，031e6b8 作废格）跑完自然收官，GPU6 队列的 gov_report 会隔离覆盖，不污染新数据。
- 我**无需再重跑**——Opus 指示确认 11 格齐后直接 eval 打分即可。

## 4. .187 32B quest 重派（info.py 修复后）进度

- **multifieldqa_en 已完成**（150 行 ✓）——证明 info.py 修复有效、pred 写文件正常、无 ValueError 白跑。
- 5 进程在跑：musique / multi_news / samsum / lcc / hotpotqa，每卡串行队列正常（.187 共 9 GPU 进程含 8B 任务）。

## 5. 待办

1. **DeepSeek 11 chat 格重跑完成**（预计数小时）→ 与 5 非 chat 有效格合并 → eval 打分（16K treatment 口径，MLA dense 锚点）
2. **32B quest 14 任务**（.187，预计 ~20h）→ 与本地 narrativeqa/qasper 合并 → 打分 + 空卡贴合
3. 主表拼接：quest 8B 41.35 + fullkv 锚点 + DeepSeek
4. 记忆更新：DeepSeek 污染面/接管入 vendor-baselines + task-chain

— by deepseek（源码核验：Opus 派单脚本 run_arm 隔离逻辑 + 各进程 cmdline/task/TS；进度实查 GPU1-6 + .187 5 卡）
