# [TL-RULER-REGION-SELECTOR-20261110] RULER 四臂区域选择精度对照

Task-ID: TL-RULER-REGION-SELECTOR-20261110
Date: 2026-11-10

## Plan

### 目标与验收

冻结用户指定对照：Qwen3-8B（GQA），RULER 11 任务，每任务前 100 条，32768/65536/131072 三档，比较 FullKV、Single-pool Avg、Single-pool MinMax、NFA（Near Avg / Far MinMax，L2 联合竞争）四条质量曲线。重点补齐 Single-pool MinMax，但不按预期差距调参、挑样本或删除负结果。

用户明确本轮语义：“只是粗筛变成全部 avg 或者 minmax 子空间下筛 K1 个 token，再子空间下细筛 K2 个 token。不需要 skip 了”。因此稀疏臂均保留 K1→K2 两级选择及 sink/SWA 正交保护区，不启用 layer/near/far/dual/static/dynamic skip。当前 Qwen3 NFA 已强制要求 `GraphReadyPagedRegionCache`，故正式入口必须使用 `prefill_frozen_request()`/`decode_frozen_request()`；其中 frozen 仅指 cache/请求生命周期，`GateConfig(mode="full")` 仍令全部层 `skip_near=false, skip_far=false`，不等于启用区域跳过。

完成条件：4 臂×3 长度×11 任务×100 条=132 个 task cells、13200 条预测齐全；模型、逐输入、生成、EOS、YaRN、tokenizer、scorer、选择器与源码身份闭合；独立只读实现重新计算官方逐样本分数和汇总，并核验 prediction/receipt/manifest 哈希。独立验收前保持 pending，最终只能报告 `usable-with-scope`，不宣称绝对无 bug。

### 冻结配置

- 生产源码固定为本地 clean HEAD `84d6110c929a09cda0f3e3fd659c9cb8a6c62f1e`；主工作区其他会话 dirty 文件不带入生产。以只含 `benchmark/`、`sparse_attn/` 的同字节源码归档部署，run config 逐文件 SHA256 由 `pred_ruler.py` 的 extended receipt 在每次生成前复核。
- 数据：`/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/{length}/{task}.jsonl`，每任务 `skip=0, max_num=100`，不筛选样本。
- 生成：dense prefill、greedy；FullKV 使用普通 eager decode，稀疏三臂使用 GraphReady capture/replay decode。NIAH=128、cwe=120、fwe=50、vt=30 个新 token，`min_new_tokens=1`；EOS 为模型 generation config 与 tokenizer EOS 的去重并集；seed=42。
- YaRN：32K 关闭，64K factor=2.0，128K factor=4.0。
- 公共稀疏几何：page=64、K1=128、K2=1024、`nfa_subspace=tail`、`tia_level2_cmp_ratio=4`（tail32）、`nfa_l2_subspace=same`、sink=4、SWA=128、`nfa_boundary_bypass=true`、`nfa_l2_quant=false`；`nfa_enable_layer_skip=false`、`nfa_dyn_dgate=false`。三稀疏臂显式使用 `nfa_prefill_frozen_regions=true, nfa_frozen_region_mode=full`，禁止 static gate artifact。
- FullKV：`method=none`，不做 token 选择。
- Single-pool Avg：`method=nfa, far=avg, near_pages=0, near_tokens=off, near_len=0`，K1 全分给唯一池，L2 在该池候选中选 K2。`RegionGeometry` 不接受 near=0，因此 cache 仅把边界钳制为 `near=page=64`；selector 的 `near_pages=0` 保证没有 near 候选页，与 selector 自身的 `POOL_NEAR_LEN=max(page, near_len)` 一致。
- Single-pool MinMax：同上，仅 `far=minmax`。
- NFA：`far=minmax, near=avg, near_pages=32, near_tokens=off, near_len=4096`；L1 双池，L2 near/far 候选联合 top-K2，不使用数值 near/far token 配额。当前 GraphReady L2 kernel 不读取 `nfa_near_tokens`，故两个单池同样使用 `off`，不写入旧数值配额身份。

### 历史结果检索与复用判决

统一 `scripts/result_store.py` 在当前项目不存在；已用有界 Glob/Grep 和 JSON 字段解析检索 `agent_doc/results/` 与 `exp/trace/results/`，不冒充调用过统一 result store。

- `e146ca_r{32,64,128}k_formal` 是旧 Single-pool Avg，AVG 分别约 58.3655/45.2727/39.2527，11×100 人口完整；其 producer 可定位到 `0b03d4f8250b42365ee834c0021ba3917a7b9325`，但当前入口已改变任务级生成预算、EOS 和后续运行路径，不能仅凭 K1/K2 相同复用。
- `e147c/e147h` 使用 `Aa0.25_b0.0625_g0.625`、P_n=8/P_f=120、k_n=320/k_f=572 等数值区域配额，明确不是本轮 NFA 联合 L2。
- FullKV 候选 `e116e_ruler32_formal_fullkv`、`e119_ruler64k_formal_fullkv`、`e119_ruler128k_formal_fullkv` 的 AVG 为 59.38/48.54/46.43；32K manifest 明示 `identity_mode=legacy-partial`，32K/64K scorer/formal 哈希也不同于当前协议，旧任务生成预算不能闭合成当前任务级配置。
- `ruler_dynamic_gate_scan_20261110` 只含 sample0、首个 gate、decision-only/no-reclaim，不是全样本质量证据；`nfa_hotpath_opt` 中同名数值含 TPOT，不作质量分。

复用决定：四臂 12 个 arm×length 格子全部 `rerun`，旧结果仅保留作事后交叉核对。原因是当前严格配对要求相同实际生成预算、EOS、scorer 与源码身份，旧候选至少有一项实质差异或身份缺口；不是根据分数好坏选择重测。

### 执行与独立验收

- 只用实测空闲卡，不杀占用进程；生产前再次检查，每卡一个串行驱动，任务确定性互斥分片，128K→64K→32K。先做四臂 32K、`max_num=1` 冒烟，核验实际 treatment、逐层全 keep、GraphReady/eager 执行身份、单池几何、联合 L2、预算/EOS/receipt 后才启动正式缺失格。
- 每个 task cell 独立输出目录和 prediction pointer，不用跨臂 best-file 池；失败保留日志，不以进程存在或 DONE 字符串代替完成检查。
- 原始预测、producer receipt、运行日志和配置保留在稳定结果区；中央结果目录保存精简汇总、脚本、必要原始证据或有界哈希清单，不把大型模型/预测副本重复入库。
- 独立只读审计不导入 producer 汇总逻辑：重新读取 33 个输入文件及每格 100 行预测，核对样本 ID、输入/答案身份、生成 token、预算、模型/config/source/input SHA、11 任务人口，调用官方 scorer 逐样本重算并重建四臂三长度曲线。检查 implementation、reference_boundary、data_integrity、numerical_sanity、measurement_validity（准确率协议；无性能主张）与 reproducibility。
- 单 Agent 串行执行；独立验收是不同实现/独立只读进程，不冒充第二位人类 reviewer。

## Progress

### 进度与证据

- 2026-11-10/11：已定位当前真实入口 `benchmark/RULER/pred_ruler.py`、L1 统计与配额 `sparse_attn/indexer/nfa_layer_batch_selector.py`、显式几何约束 `sparse_attn/indexer/nfa_indexer.py`。临时 clean worktree 已在制作源码归档后删除；生产精简源码树为 `/tmp/ruler-region-selector-src-84d6110c`。
- 2026-11-11 空卡实测：pc40 GPU0、pc31 GPU1、pc32 GPU1 各 1 MiB；其他卡有 compute 进程，不抢占。pc40 `/usr/bin/python3`、pc31/pc32 `/home/fluentllmenv/bin/python3` 均为现有 `torch=2.8.0+cu128, transformers=4.56.2` 环境，未安装或更换依赖。
- run config 已绑定 1195 个执行源文件、12 个模型文件、33 个输入文件，SHA256=`918e253d5656d0f1f1644ea58feaf94ef08d9d0d5ec2c5e1a7a08e441407a8bf`。新增任务级入口 `benchmark/RULER/pred_ruler_region_selector.py` 仅绕过原 champion-only 的单池几何校验，NFA 仍走原生校验；其 SHA256=`3f0cc7709efe4d2fbf9e52410c47f29ef67ede6eeaaf5f04c39be9096e7abef1`，已纳入上述源码闭包。
- 三份旧 eager 冒烟均在 `qwen3_attn_patch.py:41` 被 `NFA requires GraphReadyPagedRegionCache` 正确拒绝；日志保留在 pc40/pc31/pc32 的 `/tmp/ruler_region_selector_smoke_logs_pc*/000_32768_*_niah_single_1.log`，没有删除负结果或靠开启 skip 绕过。
- GraphReady 单样本冒烟：Single-pool Avg（pc31）、Single-pool MinMax（pc32）和 NFA（pc40）均已完整通过 receipt/treatment、graph 执行、128-token 预算、36/36 层 keep、全部 skip flag=false 审计；pc40 FullKV 配对项也已通过 eager/receipt/预算审计。pc40 驱动终态 `state=complete, jobs_complete=2/2`，GPU0 已释放至 1 MiB。
- 完成审计器自身两处误判已修正且未重跑或修改预测：`info.py:421-423` 会有意省略默认 `far_skip_backend/cache=legacy/retain`，审计器现按缺省语义归一；`request_phase` 的真实枚举值是大写 `FINISHED`。当前 driver SHA256=`0c5819bc5d06be940b4692cf31b71abc49a09e0f216106769652aa024583d8c5`。
- 2026-11-11 11:08/11:09 正式 132 cells 已在三张实测空卡点火，每卡 44 cells，按 128K→64K→32K 排序且 `(length, arm, task)` 全局唯一。分片 SHA256：pc40=`5247c2f4f298368c08f4ad349336a0e615e5fbd76e0aced3508e6783b0437b5d`、pc31=`c478b9dfed9728ed90e378e4ae643bd9f728eecac32e832f27dd6d1f831b4355`、pc32=`73e25ad89765cd16a1f3fe93501f93e102f9c05f33ea8fcb68f63c834457fd74`。
- 正式输出为各机 `/home/wangyuanshuo02/ak_verify_results/ruler_region_selector_20261111`，`t=rsel20261111`、`pred_postfix=_rsel`，与冒烟目录隔离；status/logs 位于同级 `..._status_pc{40,31,32}.json` 与 `..._driver_logs_pc{40,31,32}`。driver PID 分别为 4171617/1983442/3530445，首轮 producer PID 为 4171619/1983445/3530447。
- 点火三重验证通过：三 driver/producer 均存在，pc40 GPU0、pc31/pc32 GPU1 各约 49.5 GiB，日志已进入 128K、100 条生成；三个首轮命令均含 `yarn_factor=4.0`，run config SHA 与冻结身份一致。尚未完成任何正式 task cell，也未产生或验收本轮质量分。
- 2026-11-11 首轮部分生成只读核查：pc40 NFA 曾写 2/100，pc31 Single-pool Avg、pc32 Single-pool MinMax 曾写 4/100；三臂 sample0 均为 `execution.requested/actual=graph/graph`、128 个生成 token、`request_phase=FINISHED`、36/36 层 keep、`near/far/both=0`，36 层 `skip_near/skip_far` 全 false，且每层 8 个 KV head 的 `visible_lens` 均为 1024。以上仅是未提交 generation 的早期 sanity evidence，不代替 100 行完整人口、receipt 和独立评分验收。
- 2026-11-11 首轮三链均在第 7 个 128K 样本创建 cache 时 OOM，driver 均 fail-closed 停止、`jobs_complete=0/44`、目标卡恢复 1 MiB；三个未提交 generation 各保留 6 行，旧 status/log/stdout 已以 `pre_oomfix` 另存，未删除负结果或把 partial 计为完成。根因是独立 B1 cache 的 `GraphSessionPool` 在请求结束后仍保留 runner/CUDA Graph/static output，导致约 19 GiB/请求的 KV arena 逐请求累积。任务级 wrapper 现于 `complete_frozen_request` 边界清空旧 session；选择、decode 数值和提交协议不变。修复 wrapper SHA256=`14bd8e8d9a6c01a1c96353fd460a382cd0393efea72918612767d4c25ac45cfc`，新 run config SHA256=`e38d5b981efa3f4b5019db4647de99f3743cee84d490955cf6bcfe4a3ba07482`（旧配置已保存为 `/tmp/ruler_region_selector_run_config_84d6110c.pre_oomfix.json`）。三机源码/config 哈希复核一致，128K×8 样本修复冒烟已在独立输出目录点火；须全部提交并与旧 partial 前 6 行逐 token 对齐后才能重启正式链。
- 2026-11-11 显存修复冒烟三臂均 `state=complete, jobs_complete=1/1`、8/8 行提交，第 7/8 行越过原 OOM 点且目标卡最终释放至 1 MiB；三臂新 generation 与旧 partial 前 6 行的 `_id/_answers_sha/length/generated_token_ids/pred` 全部逐行一致，证明该生命周期修复未改变已观察生成。正式 132-cell 链于 11:43 重启，新 driver PID：pc40=4186031、pc31=1997582、pc32=3544915；producer PID=4186033/1997584/3544917，三卡目标显存约 49.5 GiB，YaRN=4.0 与新 run config 身份复核通过。

### 阻塞与下一步

1. 先完成三臂 128K×8 显存修复冒烟，核验第 7/8 行不再 OOM、完整 receipt/audit，并与首轮 partial 前 6 行逐 token 对齐；随后保留失败证据并用同一 132-cell 分片重启正式链。
2. 监督重启后的首个 128K cell，核验 100 行人口、YaRN=4.0、graph/eager 与逐层 no-skip 审计；失败则停链修复，不跳过失败格。
3. 132 cells 全部提交后，将 #17 标记完成并启动 #18，由独立只读实现重算 11×100 官方逐样本分数并交付四条曲线；验收前全部保持 pending。
4. 汇总前把三机 prediction/receipt/log/status 与分片、run config、driver/wrapper 身份做完整对账，不将部分 cell 当作全量结论。
