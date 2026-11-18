# TL-NFA-K2-SENSITIVITY-20261117

## 目标与范围

按用户指令，RULER 保持暂停，先完成 Qwen3-8B × LongBench v1 HotpotQA/GovReport 的 NFA gate 阈值 × L2 token 预算二维敏感性：

- gate epsilon=`0.01/0.05/0.1/0.2/0.5`，K2=`256/512/1024/2048`，两任务每格 200 源请求，共 40 个质量格、8000 请求。
- 每个 epsilon×K2×任务另完成一个独立 eager L1+L2 算子格，共 40 格；主 attention 一并记录，但不扩展成完整 TPOT 矩阵。
- 完成后只提交并正常推送本轮 Gov/Hot 代码、测试、runner、必要证据、汇总与任务文档到 `origin/main`；不夹带 RULER、Quest、其他 AI 或其他会话修改。
- 不同 owner 独立验收前，所有结果保持 `pending-independent-verification`。

## 固定协议

质量格固定 tail32、`tia_level2_cmp_ratio=4`、K1=128、page64、near32页/far96页、`near_tokens=off`、Avg/MinMax、gate-on、observation rows=256、eager、输入 cap=31500、HotpotQA/GovReport max-new=32/512，以及官方 EOS/stop-string/min-new/sample_score 协议；每格显式传入 epsilon 与 K2。

算子格固定 32K、B1、sample0、LongBench deterministic context repetition、3 warmup、5 profile repeats、32 decode steps、eager、natural gate、独立 nsys/CUPTI 进程。L1=L1 score+Top-K+candidate，L2=fine score+filtered Top-K+selection union；profiler 数据不与 clean TPOT 相减推导 host 开销。

## 结果复用检索与身份决定

在当前 `agent_doc/results/` 对既有 Gov/Hot 完整结果做了有界检索，没有发现可匹配本轮完整二维网格的闭集结果。旧 gate 消融虽有部分 K2=1024 数据，但本轮给质量和算子入口增加真实 epsilon×K2 配置链会改变 harness 身份，不能直接与新组合拼曲线；因此 20 个 epsilon×K2 组合全部在同一新 producer 下闭集，旧结果只作历史参考。

## 实现与主机验证

- `benchmark_longbench_quality.py`：`--nfa-level2-topk` 真实进入 model patch、`NFAIndexer`、arm 名、runtime manifest 与 receipt；非默认档使用 `tail_gate_on_k2256/k2512/k22048`，默认 K2=1024 保持 `tail_gate_on`。FullKV 显式携带 K2 时 fail-closed。
- `measure_nfa_selector_e2e.py`：K2 进入真实 patch/indexer，patch 后立即核对 `indexer.args.tia_level2_topk`，结果 JSON/stdout 记录真实运行时值。
- `run_nfa_perf_matrix.py`：新增 `--level2-topk`，向 NFA 测量命令透传，并纳入 `expected_fields()` 的断点复用身份，防止跨预算错误复用。
- 普通及 `python -O` 聚焦回归均为 **59 passed**；`git diff --check`、三份 shell `bash -n`、审计器 `py_compile` 与入口依赖检查通过。
- 干净生产提交：`ebcae298297c18f68ec224de611dc8b922ab9e95`。质量与算子 source diff 均为空，因此不依赖跨机 Git blob 缩写长度。
- 质量 harness SHA256=`d05703fb50bc106e560f88588c020418eca82dbcc51989a8450a031d73f03fde`；算子 harness SHA256=`f64a35b43e258f33617c5156439a66ed4a8672a03ec80cb22d86af902e7c9cb1`。

## Runner 与审计

脚本位于 `agent_doc/results/nfa_k2_sensitivity_20261117/_dispatch/`：

- `run_quality_grid_cell.sh EPS K2 TASK GPU`：完整 manifest 且真实 epsilon/K2/源码/200行门禁通过才跳过；否则只续跑未闭集单元。每格 `flock` 唯一 owner。
- `run_operator_grid_cell.sh EPS K2 TASK GPU`：复用成熟 matrix driver 的 GPU hygiene、nsys finalize、SQLite/report SHA 与完整身份复用；只选 `nfa-operator-profile`。
- `queue_grid_cells.sh GPU quality|operator EPS:K2:TASK ...`：每格点火前实测空卡，20秒内完成进程、显存/compute apps、日志三重验证；前任失败即停止，不带病接力。
- `audit_results.py`：按 arm 精确检查 epsilon/K2、空 source diff、官方 evaluator 与逐样本分数、runcap、输入身份、36层 decisions、gate、receipts、算子 kernel 分类、nsys 证据和文件哈希；要求40+40格全部闭集。

原始输出根目录：`~/ak_verify_results/nfa_k2_sensitivity_ebcae298_20261117/`。pc33 仅在实测空闲时用于本网格算子补缺，其他机器同样不抢占或终止其他用户/会话进程；新派单前重新实测 GPU。

## Progress

### 2026-11-17 生产前门禁

K2 配置链、matrix 复用守卫、runner 与审计器已完成，生产提交和空 diff 身份已固定。尚未启动任何 K2 GPU 质量或算子格，当前没有质量分数、算子耗时或趋势结论。下一步只在实测空闲的远端 GPU 上派发互斥单元，所有新结果保持 pending-independent-verification。

### 2026-11-17 四机部署与互斥等待队列

生产提交 `ebcae298297c18f68ec224de611dc8b922ab9e95` 已通过单提交 Git bundle 部署到 pc02/pc40/pc31/pc32 的独立 detached worktree `/tmp/nfa-k2-sensitivity-20261117`，四机 tracked diff 均为空。四份 `_dispatch` 工具已同步并逐机核对 SHA256、shell 语法、审计器编译、模型、两任务数据和 nsys；确定性分配与 queue PID 固定于 `_dispatch/dispatch_plan.json`。

- pc02 GPU0/1/2/3：GovReport K2=256/512/1024/2048 质量格。
- pc40 GPU0/1、pc31 GPU0/1：HotpotQA K2=256/512/1024/2048 质量格。
- pc32 GPU0/1：GovReport/HotpotQA 各四档 eager L1+L2 算子格。

18:30 复核 10/10 queue 进程存活；所有目标卡仍有其他 compute app，因此没有 K2 单元点火，也没有闭集结果。队列每格重新实测空卡，前任失败即停止，不抢占或终止其他进程。pc14 两次 SSH 均连接超时，未部署或派单；pc33 继续按本任务边界排除。RULER 保持暂停。

### 2026-11-18 完整 epsilon×K2 网格与六卡补缺

用户澄清目标为 5 档 epsilon × 4 档 K2 × 2 任务的精度+算子网格，而非仅 epsilon=0.1 下扫 K2。当前固定 producer 下已有 8 个 epsilon=0.1 质量 manifest（GovReport 四档、HotpotQA 四档）和 1 个 GovReport epsilon=0.1/K2=256 算子 JSON；本轮只派其余 32 个质量格和 39 个算子格，不把 partial 输出、日志或失败 nsys 计作闭集。

- 新增 `run_quality_grid_cell.sh`、`run_operator_grid_cell.sh`、`queue_grid_cells.sh`，20 组真实质量/算子配置链主机测试通过；三份 shell 均通过 `bash -n`。
- `audit_results.py` 已扩展为 40+40 格：逐格核对真实 epsilon/K2、arm、manifest/receipt、summary、官方评分、输入身份和 nsys 证据；`py_compile` 通过。当前不能提前运行最终审计，须等全格闭集。
- pc02 四条质量队列各 8 格：GPU0/1 分别 GovReport K2=256/512 与 1024/2048 的四个非默认 epsilon，GPU2/3 对 HotpotQA 做同样分配。queue PID=2222815/2223166/2223514/2223841。
- pc33 使用同一 `ebcae298` 的新干净 detached worktree `/tmp/nfa-k2-grid-producer-ebcae298`，GPU0 运行 GovReport 其余 19 算子格，GPU1 运行 HotpotQA 全 20 算子格；修复后 queue PID=2940949/2942420。原工作树有任务文档 tracked 修改，未用于生产测量，也未放宽 source-diff 门禁。
- 六卡派单前均为 1 MiB、无 compute app；首格点火 20 秒验证均通过。pc02 四卡约 21 GiB，pc33 两卡约 25 GiB，每卡仅一个测量 compute app，日志记录 `GRID_TRIPLE_VERIFIED`。
- 当前派单与脚本 SHA 固定于 `_dispatch/dispatch_plan.json`。每小时 cron=`76fe7d8e`（`7 * * * *`）收割、修复、重新实测空卡并只补缺失格；pc14 继续排除，RULER 保持暂停。所有结果继续 `pending-independent-verification`，尚无新增网格分数或性能结论。
- 首轮算子后置门禁发现 matrix driver 默认只写 driver receipt，不自动写 `summary.json`，两队列因此按 fail-closed 停止；原始 operator JSON 与 nsys 证据完整。runner 已增加同参数 `--summarize-only`，两个首格均以 `valid-existing` 复用原始测量、补齐 summary 并通过完整门禁，未重复 GPU 测量；随后队列重启并继续后续格。

### 2026-11-18 07:24 每小时监督第 1 轮

- 全池盘点为 **9/40 质量格、12/40 算子格**：质量是原 8 格加 pc02 新增 HotpotQA epsilon=0.01/K2=256；算子是 pc32 原 1 格加 pc33 新增 11 格。其余 31 质量格和 28 算子格仍未闭集。
- pc33 新增 11 格逐格调用正式 `audit_operator_cell()`，真实 producer/epsilon/K2、空 source diff、32K/B1/eager/sample0/natural 配置、输入 SHA、GPU hygiene、kernel 分类、nsys finalize、report/SQLite SHA、summary 和正算子耗时全部 PASS。正在测量但尚无 summary 的 K2=512 格已明确排除在闭集计数外。
- pc02 四条质量队列与 pc33 两条算子队列均存活，后续格继续点火并通过三重验证；HotpotQA 已完成首个非默认质量格，GovReport 首格仍在生成。
- pc40/pc31/pc32 双卡均有其他 compute apps，本机与 pc02 卡也由本任务队列占用，因此本轮没有可安全新增派单的卡，未重复测量、未抢占进程。RULER 保持暂停，结果仍待完整闭集与独立验收。

### 2026-11-18 08:37 每小时监督第 2 轮

- 全池严格盘点增至 **16/40 质量格、13/40 算子格**：质量为 pc02 十二格、pc40 两格、pc31 两格，较上轮新增七格；算子仍为 pc33 完整复核的十三格。其余 24 个质量格和 27 个算子格未闭集。
- 16 个质量格已逐格通过正式 `audit_quality_cell()`：真实 producer/epsilon/K2、空 source diff、200 predictions/receipts、complete runcap、输入身份、36 层 decisions、gate counts、官方 evaluator 与逐样本分数、runcap/预测/评分文件 SHA 全部 PASS。pc40/pc31/pc32 的审计器已同步到同一 SHA256=`2b640830...6301` 并通过 `py_compile`，不修改生产 harness。
- pc32 旧 GovReport epsilon=0.1/K2=256 算子 JSON 已定位无效根因：当前 `gpu_hygiene="polluted-rerun"`，`nsys_profile_finalized=false`，且无 operator/kernel summary 和 nsys report/SQLite receipt；虽然 producer、epsilon/K2、输入 SHA 和其余配置字段匹配，仍不满足 `valid_existing()`，不得复用或计入闭集，后续须在真实空卡上只重测这一格。
- pc02 四条质量队列均存活并继续推进；pc33 两条算子队列已退出，本机双卡以及 pc40/pc31/pc32 均实测有其他 compute apps，因此本轮没有安全的新空卡可派，未重启本机队列、未重复测量、未抢占或终止进程。RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 09:24 每小时监督第 3 轮

- 本轮已完整复核 **22/40 质量格、23/40 算子格**，分别剩 18 和 17 格。质量为 pc02 十八格、pc40/pc31 各两格；本轮新增六格均通过正式官方评分、输入身份、receipt/runcap、36 层 gate 与 SHA 审计。算子为 pc33 十九格、pc31 四格，均通过真实 epsilon/K2、summary、GPU hygiene、kernel 分类、nsys report/SQLite 与 SHA 审计。
- pc31 双卡与 pc33 双卡本轮实测均为 1 MiB、无 compute app，已将当时剩余 27 个算子格确定性分片到四张空卡：pc33 GPU0/GPU1 各七格 GovReport，pc31 GPU0 七格、GPU1 六格 HotpotQA。queue PID 分别为 pc33=`3218694/3219833`、pc31=`2536072/2536514`；四卡首格及后续已点火格均在 20 秒完成进程、约 25 GiB 显存/单一 compute app、日志三重验证，未重复派发已闭集格。
- pc32 旧 GovReport epsilon=0.1/K2=256 已在 pc33 GPU0 按同一 producer 和协议重新测量，新生成完整 operator JSON、summary 与 nsys 证据并通过正式 `audit_operator_cell()`，旧 polluted-rerun 记录保留为已修复历史，不再计作缺失。
- pc02 四条质量队列与新四条算子队列均存活；pc40/pc32 仍有其他 compute apps，pc14 再次 SSH ConnectTimeout，未部署或派单。未抢占或终止其他进程，RULER 保持暂停；在飞但尚未形成 summary 的格不计闭集，所有结果继续 `pending-independent-verification`。

### 2026-11-18 10:45 每小时监督第 4 轮

- 已完整复核 **28/40 质量格、40/40 算子格**。质量为 pc02 二十四格、pc40/pc31 各两格；算子为 pc33 二十七格、pc31 十三格。每格均通过真实 producer/epsilon/K2、空 source diff、完整配置与文件 SHA 审计；质量另通过 200 predictions/receipts、complete runcap、36 层 decisions/gate、官方 evaluator 和逐样本 `sample_score`，算子另通过干净 GPU hygiene、kernel 分类、正 L1/L2/main-attention 耗时、nsys finalize 和 report/SQLite SHA。
- 跨机联合检查通过：40 个算子 `(task,epsilon,K2)` 组合无缺失/重复，两个任务各 20 格的输入 SHA、material/suffix SHA 与 32645 prompt tokens 完全一致；28 个质量组合无重复，HotpotQA 十八格和 GovReport 十格的 ordered/per-sample 输入身份分别一致。审计结果保存在各机输出根目录 `_audits/quality_audit_round4_*.json` 与 `_audits/operator_audit_round4_*.json`。
- pc33/pc31 四条第三轮算子 supervisor 均以 `GRID_QUEUE_ALL_DONE` 正常结束，双卡回到 1 MiB、无 compute app。pc02 新完成 HotpotQA epsilon=0.05/K2=2048，官方分数 59.16，gate keep/near/far/both=6601/17/581/1；该格已补做正式审计，不把在飞日志或单独 manifest 当作最终联合审计。
- 为利用算子闭集后释放的空卡，已做**精确无重复质量重平衡**：只终止 pc02 原三条队列父 shell（2222815/2223166/2223841），保留 GovReport epsilon=0.5/K2=256 与 epsilon=0.5/K2=1024 两个在飞 runner（2545945/2476419）；HotpotQA epsilon=0.05/K2=2048 在冻结前已闭集，不重复派发。其余十个尚未点火格分到五张实测空卡，每卡两格：pc02 GPU2 queue=2581459，pc33 GPU0/1 queue=3347988/3347989，pc31 GPU0/1 queue=2565919/2565920。pc33/pc31 的质量 runner 已补齐并核对 SHA256=`5c178d98...9568b1a6`、`bash -n` 通过；pc33 显式使用同一干净 producer worktree。
- 五条新队列首格均在 20 秒完成进程、约 16–21 GiB 显存/单一 compute app、日志三重验证。当前剩余 12 个质量格全部有唯一 owner，其中 10 个属于新队列、2 个为 pc02 保留在飞格；原 pc02 父队列不会再接力点火，因此无跨机重复测量。详细单元归属固定于 `_dispatch/dispatch_plan.json` 的 `quality_rebalance`。
- 质量尚未 40/40，本轮不运行最终联合审计、不提交、不推送；继续每小时收割新闭集并复核。RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 11:21 每小时监督第 5 轮

- 完整审计增至 **30/40 质量格、40/40 算子格**。新增闭集为 pc02 保留在飞的 GovReport epsilon=0.5/K2=256 与 K2=1024，官方分数分别 15.97/15.98，两格 gate keep/near/far/both 均为 91/769/54/6286；均通过 200 predictions/receipts、complete runcap、36 层 decisions、官方 evaluator、逐样本评分与 SHA 审计。30 格跨机组合唯一，HotpotQA 十八格、GovReport 十二格的逐任务 ordered/per-sample 输入身份一致。
- 其余 10 个质量格均已有唯一 owner，没有把缺失清单误当作可任意重复派单。为使用 pc02 新释放的 GPU0/1，只冻结 pc02 queue=2581459 和 pc33 queue=3347988 的后续点火，保留当前 GovReport epsilon=0.01/K2=512（runner=2581466）与 epsilon=0.2/K2=512（runner=3348047）继续测量。
- 两个原队列中尚未点火的后续格已转移到 pc02 空卡：GPU0 queue=2656049 测量 epsilon=0.05/K2=512，GPU1 queue=2656050 测量 epsilon=0.5/K2=512；20 秒进程、约 16–21 GiB 显存/单一 compute app、日志三重验证通过。pc33 GPU1 queue=3347989 与 pc31 GPU0/1 queue=2565919/2565920 保持原有两格队列不变。
- 当前 10 个缺失格中 7 个正在测量、3 个在未冻结队列中等待，归属固定于 `_dispatch/dispatch_plan.json` 的 `quality_rebalance_round5`；未抢占或终止其他用户进程，无跨机重复测量。质量尚未 40/40，本轮不做最终联合审计、不提交、不推送；RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 12:31 每小时监督第 6 轮

- 已完整复核 **34/40 唯一质量格、40/40 算子格**，质量剩 6 格。物理质量 manifest 共 35 份，其中 HotpotQA epsilon=0.2/K2=2048 在 pc02 与 pc31 各一份，只计 pc31 为 canonical。当前 canonical 分布为 pc02 二十九格、pc40 两格、pc31 三格；HotpotQA 十九格、GovReport 十五格。
- 本轮新增四个唯一质量格均通过正式 `audit_quality_cell()`：GovReport epsilon=0.01/0.05/0.5 的 K2=512 官方分数分别 **33.90/34.01/16.01**，gate keep/near/far/both 分别为 **7108/0/92/0、6158/1/1041/0、91/769/54/6286**；HotpotQA epsilon=0.2/K2=2048 官方分数 **57.63**，gate=**2137/1894/1201/1968**。200 predictions/receipts、complete runcap、36 层 decisions、官方 evaluator、逐样本评分、输入身份与 SHA 全部 PASS。
- **调度竞态披露与修复**：第四轮首次冻结 pc02 原队列时，HotpotQA epsilon=0.2/K2=2048 在父队列接力与跨机重派之间发生时间竞争，pc02 与 pc31 各产生一份完整结果；此前“无跨机重复测量”的表述不再成立。两份均通过完整审计且官方分数、gate、输入身份一致，prediction/receipts/manifest SHA256 也分别完全相同；原始产物全部保留，pc02 份标记 `duplicate/excluded`，最终只收割 pc31 canonical，不重复计数。
- 为利用 pc02 空卡，只转移两个仍在队列等待、尚未点火的 GovReport 后续格。先按精确 PID 冻结 pc33 queue=3347989 与 pc31 queue=2565919，保留当前 runner=3348023（epsilon=0.01/K2=2048）和 2565933（epsilon=0.2/K2=2048），随后全机清点全部质量 runner/queue，确认无隐藏接力、两个转移格未点火。
- 转移后 pc02 GPU0 queue=2688483、runner=2688498 测量 GovReport epsilon=0.05/K2=2048，GPU1 queue=2688484、runner=2688500 测量 epsilon=0.5/K2=2048；两格均在 25 秒完成进程、16248 MiB 显存、单一 compute app、日志三重验证。其余在飞格为 pc33 GovReport epsilon=0.2/K2=512、pc31 HotpotQA epsilon=0.5/K2=2048，归属固定于 `_dispatch/dispatch_plan.json` 的 `quality_rebalance_round6`。
- 当前剩余 6 格均有唯一 owner，pc02 GPU2/3 虽空闲但没有未 owned 格，不再重复派发；未抢占或终止其他用户/会话进程。质量尚未 40/40，本轮不做最终联合审计、不提交、不推送；RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 13:13 每小时监督第 7 轮

- 完整审计增至 **35/40 唯一质量格、40/40 算子格**。pc31 新完成 HotpotQA epsilon=0.5/K2=2048，官方分数 **41.26**，gate keep/near/far/both=**20/167/12/7001**；200 predictions/receipts、complete runcap、36 层 decisions、官方 evaluator、逐样本评分、输入身份与 SHA 全部 PASS，审计保存在 `_audits/quality_audit_round7_pc31.json`。
- **HotpotQA 已完成 20/20 全网格**；GovReport 当前 15/20。物理质量 manifest 共 36 份，仍只有第六轮披露的一个跨机重复组合被排除计数；canonical 为 pc02 二十九格、pc40 两格、pc31 四格。
- 剩余 5 格全部为 GovReport 且均有活动 runner：pc33 GPU0 epsilon=0.2/K2=512、GPU1 epsilon=0.01/K2=2048，pc31 GPU0 epsilon=0.2/K2=2048，pc02 GPU0 epsilon=0.05/K2=2048、GPU1 epsilon=0.5/K2=2048。pc02 GPU2/3 与 pc31 GPU1 实测空闲，但没有未 owned 缺失格，因此本轮不新增派单、不重复测量；pc40 双卡仍有其他 compute apps，不抢占。
- 算子保持 pc33 二十七格、pc31 十三格完整闭集。质量尚未 40/40，本轮不做最终联合审计、不提交、不推送；RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 14:15 每小时监督第 8 轮

- 完整审计增至 **38/40 唯一质量格、40/40 算子格**，质量只剩 2 格。物理质量 manifest 共 39 份，仍只有一个已披露的跨机重复组合被排除计数；canonical 为 pc02 三十一格、pc40 两格、pc31 五格，HotpotQA 二十格、GovReport 十八格。
- 本轮新增三个 GovReport 格均通过正式 `audit_quality_cell()`：epsilon=0.05/K2=2048 官方分数 **33.48**、gate=**6158/1/1041/0**；epsilon=0.2/K2=2048 官方分数 **28.98**、gate=**1945/1259/2041/1955**；epsilon=0.5/K2=2048 官方分数 **15.98**、gate=**91/769/54/6286**。三格均为 200 predictions/receipts、complete runcap，36 层 decisions、官方 evaluator、逐样本评分、输入身份和 SHA 全部 PASS。
- pc02 两个转移 queue 与 pc31 在飞 runner 均已结束，pc02 四卡、pc31 双卡实测释放。只剩 pc33 GPU0 的 GovReport epsilon=0.2/K2=512（runner=3348047）和 GPU1 的 epsilon=0.01/K2=2048（runner=3348023）仍在测量；两格都有唯一活动 owner，其他空卡不重复派发，本轮无新 GPU 任务。
- 质量尚未 40/40，本轮不做最终联合审计、不提交、不推送；下轮收割最后两格后统一跨机收割与最终审计。RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 15:13 每小时监督第 9 轮

- 严格盘点保持 **38/40 唯一质量格、40/40 算子格**，本轮没有新增完整 manifest，不把 partial 输出计作闭集。pc02 三十二份物理质量 manifest、pc31 五份、pc40 两份及 pc33/pc31 的 40 份算子 summary 均与上轮一致。
- pc33 最后两格运行正常并持续推进：GovReport epsilon=0.2/K2=512 已完成 **167/200** 请求，runner=3348047；epsilon=0.01/K2=2048 已完成 **106/200** 请求，runner=3348023。日志分别持续追加逐请求输入/输出长度，双卡各有本任务单一 compute app，没有失败或停滞证据。
- 两格都有唯一活动 owner，其他空卡不重复启动；本轮无新 GPU 派单、无新审计分数。质量尚未 40/40，不做最终联合审计、不提交、不推送；RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 16:19 每小时监督第 10 轮

- 完整审计增至 **39/40 唯一质量格、40/40 算子格**，质量只剩 1 格。物理质量 manifest 共 40 份，仍只有第六轮披露的一个跨机重复组合被排除计数；canonical 为 pc02 三十一格、pc40 两格、pc31 五格、pc33 一格，HotpotQA 二十格、GovReport 十九格。
- pc33 新完成 GovReport epsilon=0.2/K2=512，官方分数 **29.42**，gate keep/near/far/both=**1945/1259/2041/1955**，ordered input SHA256=`cdf8eb4e8e259acc4c6ec2c307457f7ff920ea47748d9043b96650d9519e013d`；200 predictions/receipts、complete runcap、36 层 decisions、官方 evaluator、逐样本评分、输入身份与 SHA 全部 PASS，审计保存在 `_audits/quality_audit_round10_pc33.json`。
- 最后一格 GovReport epsilon=0.01/K2=2048 仍在 pc33 GPU1 由原 runner=3348023 测量，已完成 **146/200** 请求；compute app=1525576，显存约 19976 MiB，日志持续追加，无失败或停滞证据。pc33 GPU0 已释放，其他空卡没有未 owned 缺失格，因此不新增派单、不重复测量。
- 质量尚未 40/40，本轮不做最终联合审计、不提交、不推送；下轮收割最后一格后统一跨机收割与最终审计。RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 17:22 每小时监督第 11 轮

- 严格盘点保持 **39/40 唯一质量格、40/40 算子格**，本轮没有新增完整 manifest，不把 partial 输出计作闭集。pc02 三十二份物理质量 manifest（排除一份重复后 canonical 三十一格）、pc40 两格、pc31 五格及 pc33 一格均与上轮一致；pc32 旧 polluted-rerun 没有 summary，继续不计闭集。
- 最后一格 GovReport epsilon=0.01/K2=2048 已由原 runner=3348023 推进至 **174/200**，compute app=1525576 仍活跃。pc02/pc40/pc31/pc32 各卡及 pc33 GPU0 均实测释放，但缺失格已有唯一活动 owner，因此不新增派单、不重复测量。
- 已完成**统一预收割**：39 个 canonical 质量格与 40 个算子格汇入 `~/ak_verify_results/nfa_k2_sensitivity_ebcae298_20261117/`，HotpotQA 二十格、GovReport 十九格；pc02 的重复 HotpotQA epsilon=0.2/K2=2048 未收入 canonical，只保留 pc31 来源。40 份 operator receipt 的 nsys report/SQLite 绝对路径全部可达，缺失数为 0。
- 修复最终审计器的跨格输入身份集合比较：`per_sample_input_sha256` 是 list，现改用 canonical JSON 字符串比较，`py_compile` 通过；审计器 SHA256=`c59fdac375e9dda742db22411a10f66668d3220a17802a2788f3e8d897b0b920`。该修改不涉及生产 harness 或测量身份。
- 质量尚未 40/40，本轮不运行最终联合审计、不提交、不推送；最后一格闭集后只需将其纳入同一根目录并执行 40+40 最终审计。RULER 保持暂停，所有结果继续 `pending-independent-verification`。

### 2026-11-18 18:27 全部闭集与最终联合审计

- 最后一格 GovReport epsilon=0.01/K2=2048 已完成 **200/200** 请求并正常退出，pc33 双卡释放。该格官方分数 **33.49**，gate keep/near/far/both=**7108/0/92/0**，ordered input SHA256=`cdf8eb4e8e259acc4c6ec2c307457f7ff920ea47748d9043b96650d9519e013d`。
- **40/40 质量格、8000/8000 请求与 40/40 算子格全部闭集**。最终 `audit_results.py` 联合审计返回 `pass-production-side`：逐格真实 producer/epsilon/K2、空 source diff、200 predictions/receipts、complete runcap、36 层 decisions/gate、官方 evaluator 与逐样本 `sample_score`、输入身份和 SHA 全部通过；算子的 clean hygiene、kernel 分类、正 L1/L2/main-attention 耗时、nsys finalize、report/SQLite 及 SHA 也全部通过。
- 20 个 epsilon×K2 组合均同时包含 HotpotQA/GovReport 官方精度、四类 gate 计数和两任务 L1/L2/main-attention 算子耗时。两任务质量 ordered/per-sample 输入身份分别一致，算子输入/material/suffix SHA 与 32645 prompt tokens 一致；没有把 profiler 数据相减推导 host 开销，也没有将算子数据扩展为 clean TPOT。
- 最终汇总已归档为 `agent_doc/results/nfa_k2_sensitivity_20261117/summary.json`，SHA256=`4a2276a79d05efdfc595973af31585845f20f4809984d61f63a8ddc3438b2433`；`evidence_manifest.json` SHA256=`439e7fb9274c24d33b4ce4c77ad7c779206746d4b11eac4edb856440f031edd6`。另对清单中的 **1210/1210** 个文件重新计算大小与 SHA256，缺失或不匹配数为 0。
- 物理质量 manifest 共 41 份，仍只排除第六轮披露的一个跨机重复组合；统一根目录只含 40 个 canonical 组合。RULER 始终暂停，所有结果继续 `pending-independent-verification`。下一步只合并最新 `origin/main`、运行聚焦回归并正常提交推送本轮代码、测试、runner、汇总、必要证据和任务文档。
