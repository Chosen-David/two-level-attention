# TL-ASYNCTLS-PAPER-QWEN32B-LBV1-20261110

## 目标

按用户 2026-11-10 指令，先修正 AsyncTLS 的论文精度语义，再在 Qwen3-32B 上运行 LongBench v1。模型、代码、校准、输入、配置及输出身份均须固定；新结果在独立验收前保持 `pending`。

论文依据：`agent_doc/advice/other_paper_pdf/AsyncTLS.pdf` §4.1–4.2。实现范围是同步执行的选择算法复现，不包含异步 CPU offload、resident cache 或增量块传输，也不开展性能测试。

## 既有结果检索与复用判定

- Llama-3.1-8B 旧近似臂 `llama_vendor_lb_v1_20261109` 的 16/16、3750 请求及 AVG16=31.81 已独立验收，保留为旧实现历史结果，不覆盖、不重写。
- `exp/results_longbench_vendor/Qwen3-32B/pred_asynctls/` 为跨 KV group 共享选集、同步当前块集及 question 逐 token 稀疏的旧口径产物，只可参考任务协议，不能复用为本次修正后的成绩。
- `exp/vendor_baselines/asynctls_vendor/calib/qwen3-32b_qasper_channels.json` 是每层一份全局 channel 的旧格式；新语义要求每层、每 KV group 各一份 channel，必须重新校准。
- 因此本次代码路径相关臂全部重跑，新校准和新输出使用新身份，不与旧产物混池。

## 实现与验收范围

1. Stage-1 按每个 KV group 独立打分和选择 top blocks，不再跨 KV head 聚合。
2. channel 校准仅在该 KV group 的 G 个 query head 内聚合，格式为 `[layer][kv_group][channel]`；旧全局格式显式拒绝。
3. Stage-2 按每个 KV group 独立计算组内 softmax 均值和 top token，Stage-3 使用各组自己的全维 K/V 选集。
4. 完整 prompt 一次性 dense prefill，并以 prompt 最后位置初始化各层 `M_{t-1}`；decode 的 token 细筛使用上一步块集，同时保存当前 `M_t` 供下一步使用。
5. 当前 decode token 必须进入候选并至少保留一次；最终有效选集不超过 `kt`，无效 padding 不参与 softmax 或预算计数。
6. LongBench 的 AsyncTLS 路径不得再把 question 后缀逐 token 送入稀疏 forward；其他 baseline 的 prompt 协议不变。
7. Qwen 与 Llama 共享 helper 必须同步回归；覆盖 INT4 16 级、per-group 独立性、上一步块集、当前 token 去重/预算、full-budget dense 等价和稀疏形状，普通及 `python -O` 均执行。

## 实验门槛与交付

- 源码回归通过后才扫描 GPU；只用实测空闲卡，不杀其他任务，精确 PID 互斥。
- 固定新代码后生成 Qwen3-32B per-group channel，记录校准数据、模型、代码和文件 SHA256。
- 真实 32B smoke 通过后启动 LongBench v1 英文 16 任务；沿用 `kt=1024、B=64、kb=128` 和 greedy 协议。
- 点火后检查进程、显存与日志；保存输入身份、逐请求输出及 run-cap 证据。正式结果须由独立检查重算人口与官方分数后才可判决。

## Progress

- 2026-11-10：任务建立；完成论文/源码差异定位与旧结果复用判定。源码修正、校准和正式测试尚未完成。
- 2026-11-15：只读复核当前 `1cfed8481e87a62b3a705cd981049e8e699a1f81`：Qwen/Llama 共享实现已包含 per-KV-group Stage-1/2/3、完整 prompt dense prefill、`M_{t-1}` 和当前 token 强制入选；`benchmark/LongBench/pred.py:514-526` 将 question 与 material 拼接后一次 prefill，`_manual_generate` 保持 greedy、真实 EOS 和 max-new。每次完整 prompt prefill 都重新初始化 `_asynctls_prev_blocks`，多请求不复用上一请求块集。普通及 `python -O` 执行 `tests/test_llama_vendor_adapters.py -k asynctls` 均为 6 passed、4 deselected。旧 `qwen3-32b_qasper_channels.json`（SHA256=`3e2e227dabee70e60088757e703e1e29639a453deb7764a92314b10f60b8fc78`）被 `_load_channels` 明确拒绝，未复用。
- 2026-11-15：重新实测本机双卡各 1 MiB、无 compute app 后，在物理 GPU0 点火新 per-group 校准，后台任务 `byq5u24ms`；协议为 qasper 前 8 条、每条最多 4096 token、dc=32，目标为 64 层 × 8 KV group。点火后复核真实 Python 进程、GPU0 约 61322 MiB 和 17/17 checkpoint 加载日志，三重验证通过。校准代码 SHA256=`2b5a34909122930d5479c24dcf17d66dedac1a40cec3881a60653d9b0f4db067`，模型配置 SHA256=`97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb`，qasper 数据 SHA256=`29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854`。产物与日志位于 `agent_doc/results/qwen32b_asynctls_per_group_20261115/calibration/`；校准完成、结构复核和真实 32B smoke 前不得启动正式 LongBench。
- 2026-11-15：校准已完成并独立复核为 64 层 × 8 KV group × 32 channel，共 16384 个排序唯一且位于 `[0,128)` 的索引，当前加载器 round-trip PASS；channel SHA256=`0fc1c57319cebf6c1f6674f9c7239b33bd3d81d82211e5d73be1d5eaf6cb3cbb`，校准回执=`calibration/receipt.json`。新建固定身份正式 runner `benchmark/LongBench/run_longbench_asynctls_qwen32b_per_group.sh`，16 任务采用 `source_order[rank::world_size]` 互斥分片、complete-only SKIP，并核验 runcap 人口、结果行数/SHA 和 producer；当前 runner SHA256=`fcb1b67c3cb42567a4180b029006d0087d6cee7105edddf4e9e03b6ecdcfb1ff`。
- 2026-11-15：首次双路真实 32B smoke（rank0/1、world=100、每路 qasper 2 请求）均在生成前被 079 门禁拒绝；真实原因是新命令漏传非 NFA baseline 必需的 `--nfa_enable_layer_skip false`，共享 identity 因默认 true 展开缺失的 NFA mask。未削弱门禁，只在 runner 与 smoke 命令补齐显式开关；`bash -n` 通过，`tests/test_longbench_vendor_perf.py` 为 5 passed。attempt1 原始日志保留为 `smoke/logs/qasper_r{0,1}.log`，未产生 prediction。
- 2026-11-15：attempt2 双路真实 32B smoke 已完成，后台任务 `b008tc6il`/`b7stmigzh`；点火后核验真实 Python 进程、模型加载日志和 GPU0 约 69062 MiB，两路日志均为 2/2、`ave_budget: None`。独立只读复算四条 prediction、两份 complete runcap、结果字节 SHA、producer、分片 ID（0/100 与 1/101）、逐样本输入 token/SHA、max-new=128 及 eager requested/actual 全 PASS，truncated=0；输入长度分别为 `[4085,7644]` 与 `[3422,5978]`。rank0/1 result SHA256=`d293e41b81ad2ebb545e6cbf058687e7ad1c66001a50f9ded36681039b64029d`/`551281c6413a70e5df1eb8f2056448199dc0fe9dd40a723f47d818b35d0bfb1e`，证据=`smoke/receipt.json`。该回执仅证明真实模型功能 smoke，不是 LongBench 质量或性能结论，保持 `pending-independent-verification`。
- 2026-11-15：正式 LongBench v1 英文 16 任务已在本机双卡点火烧录，后台任务 `bg6io0zno`（GPU0/rank0）与 `bc078wl5c`（GPU1/rank1），world=2、timestamp=`20261115T01`，输出=`~/ak_verify_results/qwen32b_asynctls_lb_v1_per_group_20261115`。启动前 HEAD、runtime diff、channel SHA、runner 语法及双卡空闲门禁全过；点火后两个真实 `benchmark.LongBench.pred` 进程、双卡各约 62816 MiB 和 narrativeqa 加载日志三重验证通过。当前仅表示正式链在跑，3750 请求、canonical 合并、官方评分和独立验收均未完成。
- 2026-11-15：在 `CUDA_VISIBLE_DEVICES=""` 的独立 CPU 进程中运行只读 `audit_progress.py`，动态只消费原子发布的 complete runcap；语法检查及实际执行均通过。快照为 10/32 complete 分片、950/3750 条 finalized prediction，narrativeqa/qasper/multifieldqa_en/hotpotqa/2wikimqa 共 5/16 任务双 rank 完整闭集，两后台任务仍 running。950 条均逐样本重跑 Qwen3-32B tokenizer、31500 cap 截断、chat wrapper、question split 和 signed-int64 输入哈希，并核对完整源身份、round-robin ID/人口、结果字节 SHA、producer、固定 treatment、逐任务 max-new、truncation、答案元数据及 eager requested/actual，全部 PASS；canonical treatment SHA256=`932711dc4efdbc6a4cdf800d318fcc503d65d89c25d9dbd03af01fcfe4760910`，与文件名 `h932711dc4e` 一致。审计器 SHA256=`6e2b1675b15b16d6fea988ade3e18140a4b8ef06f8751545a48a288802e0c74d`，回执=`agent_doc/results/qwen32b_asynctls_per_group_20261115/progress_audit.json`，SHA256=`4592cd1b7268a265db924cf04a025cba1ab2484290cffbb927b3a067dee82520`。该回执是生产侧独立进程局部复算，不是不同 owner 独立验收；未向正式目录补写 treatment sidecar，未合并、未评分、未生成 macro/micro，当前 JSONL 不含逐请求 stop-reason/backend-call receipt，保持 `pending-independent-verification`。
- 2026-11-15：第二轮以同一 CPU-only 只读审计器复算新增 complete sidecar，滚动快照推进为 12/32 complete 分片、1150/3750 条 finalized prediction；新增 musique rank0/rank1 各 100 条，narrativeqa/qasper/multifieldqa_en/hotpotqa/2wikimqa/musique 共 6/16 任务双 rank 完整闭集。12 条分片的逐样本输入 token/SHA、31500 cap 截断、完整源身份、round-robin ID/人口、结果字节 SHA、producer、固定 treatment、逐任务 max-new、truncation、答案元数据及 eager requested/actual 全部 PASS；审计时间为 `2026-11-15T04:37:54.900765+00:00`，审计器 SHA256=`6e2b1675b15b16d6fea988ade3e18140a4b8ef06f8751545a48a288802e0c74d`，滚动回执 `progress_audit.json` SHA256=`467587d663d33df9797c2758eaf0700e4a461d1628ebd03f5a22a89f57eb6c83`。复核时 `bg6io0zno`/`bc078wl5c` 均仍 running，无新失败；随后真实日志确认两路均已进入 gov_report（各 100/200），musique 两路均打印 `ave_budget: None`，GPU0/GPU1 分别为 65174/72516 MiB、利用率 44%/42%，各有一个 compute app，未见 Traceback、CUDA error、Killed 或 GATE-FAIL。本轮未合并、未评分、未生成 macro/micro，也未修改正式输出。该结果仍是生产侧独立进程局部复算，不是不同 owner 独立验收，保持 `pending-independent-verification`。
- 2026-11-15：第三轮以同一 CPU-only 只读审计器复算全部已发布 complete sidecar；截至 `2026-11-15T06:37:50.800898+00:00`，快照保持 12/32 complete 分片、1150/3750 条 finalized prediction，narrativeqa/qasper/multifieldqa_en/hotpotqa/2wikimqa/musique 共 6/16 任务双 rank 闭集，本轮没有新增 complete 分片、未虚增完成量。12 条分片的逐样本输入 token/SHA、31500 cap 截断、完整源身份、round-robin ID/人口、结果字节 SHA、producer、固定 treatment、逐任务 max-new、truncation、答案元数据及 eager requested/actual 再次全部 PASS；审计器 SHA256=`6e2b1675b15b16d6fea988ade3e18140a4b8ef06f8751545a48a288802e0c74d`，最新滚动回执 `progress_audit.json` SHA256=`4dfdab8e0d588d2c7d78a6b39f4c09879df6b68e049429c9ba2ff961416cb17b`。随后真实日志确认 `bg6io0zno`/`bc078wl5c` 均仍 running，gov_report rank0/rank1 分别推进至 41/100、39/100；GPU0/GPU1 分别为 75008/72212 MiB、利用率 43%/41%，各有一个 compute app，全量日志未见 Traceback、CUDA error、Killed 或 GATE-FAIL。本轮未合并、未评分、未生成 macro/micro，也未修改正式输出；该结果仍是生产侧独立进程局部复算，不是不同 owner 独立验收，保持 `pending-independent-verification`。
