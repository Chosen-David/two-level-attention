# [TL-RULER-REGION-SELECTOR-20261110] RULER 四臂区域选择精度对照

Task-ID: TL-RULER-REGION-SELECTOR-20261110
Date: 2026-11-10

> **2026-11-11 用户最新范围覆盖**：只要 Single-pool MinMax，所有实测空卡并行铺满；Single-pool Avg、FullKV、NFA 均不再运行。当前交付矩阵因此改为 1 臂×3 长度×11 任务×100 条=33 cells、3300 条预测。此前其他臂的完整/部分输出全部保留为历史证据，不删除、不并入本轮 MinMax 结论；下文四臂方案仅作变更前记录。

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

- 2026-11-11 16:30 正式链只读核查：三机均 `state=running, jobs_complete=2/44`，合计已完整提交 6/132 cells、600/13200 条预测，无新增 failed cell。pc40 当前 128K Single-pool MinMax/`niah_single_2`=16/100，GPU0=50590 MiB；pc31 当前 128K FullKV/`niah_single_2`=17/100，GPU1=42534 MiB；pc32 当前 128K NFA/`niah_single_3`=15/100，driver/producer=3544915/3583375 存活且日志持续增长。pc32 GPU1 总显存=97091 MiB，nvidia-smi 两个 compute PID=1514122/1516302（35116/61960 MiB）在当前容器 PID namespace 均不可见；本任务 producer 的 NSpid 仅 3583375，故尚不能把任一条目直接归为本任务，不作泄漏复发结论，也未杀任何进程。

- 2026-11-11 用户将范围收窄为只跑 MinMax 并要求所有空卡并行：已按精确 PID 停止 pc40 driver/producer=4186031/46879、pc31=1997582/2055007；pc32 旧 driver 已退出，三机相关进程清点均无旧 producer。旧 status 均以 `.pre_minmax_only_20261111` 备份，完整和 partial 预测未删除。停车前三条混合链各完成 3 cells，合计 9 cells，其中正式 MinMax 完整格为 128K `niah_single_1`（pc32）和 `niah_single_2`（pc40）。
- 2026-11-11 空卡实测后采用 9 卡互斥分片：pc33 GPU0/1、pc02 GPU0/1/2/3、pc40 GPU1、pc31 GPU0、pc32 GPU0；pc14 SSH 连接超时，pc40 GPU0、pc31 GPU1、pc32 GPU1 有非本任务占用，均未抢占。五台机器的 1195 源文件、12 模型文件、33 输入文件全部通过 SHA256 复核，run config 仍为 `e38d5b981efa3f4b5019db4647de99f3743cee84d490955cf6bcfe4a3ba07482`。
- MinMax-only 矩阵共 33 cells，`(length,arm,task)` 全局唯一；已完成两格由新 driver 重新验收后 `action=skip`，其余 31 格新跑。分片 manifest=`/tmp/ruler_minmax_only_shards.json`，SHA256=`5186e69615876d9fc4d62c9676eda3bf9488d084f93a830dfad9a527833cb5ea`。九卡均已点火并完成进程/显存/日志三重核验，当前命令全部 `arm=single_minmax, nfa_far_method=minmax`：pc33 driver=1573093/1573094，pc02=3368213/3368214/3368215/3368216，pc40=59172，pc31=2066361，pc32=3622271；pc40/pc32 已分别跳过既有完整格，九卡各有一个 128K MinMax producer 在运行。
- 2026-11-12 09:36 八条链已完成，正式完整人口为 30/33 cells、3000/3300 条；pc31 GPU0 的 128K `fwe` 在 `index=69` fail-closed 停止，前 69 行留在未提交 generation，不计完成，后续 64K/32K `niah_multivalue` 未启动。该样本实际 131078 tokens，加 fwe 预算 50 后为 131128，超过 `context_length + max_new_tokens=131122` 的已验证运行时位置容量 6 tokens。旧 status/log/partial 全部保留，未跳过样本或把 partial 当完整。
- 用户选择“左截断到128K”。独立源码树 `/tmp/ruler-region-selector-src-84d6110c-leftcap` 仅修改 `benchmark/RULER/pred_ruler.py`：只对 `actual_tokens > args.context_length` 的输入保留末尾名义长度 token；预测行 `_input_truncation` 与 receipt `input_truncation` 记录 `left-to-nominal-context-v1`、样本 ID、原始输入 SHA256、保留 token IDs SHA256、原始/保留/左删长度。其他输入、预算、选择、decode 和原子提交路径不变。旧 guard 保证旧 30 格每行均满足 `actual_tokens <= nominal_context_length`，故该分支对旧结果为结构性 no-op，不重跑、不改写旧预测。
- 左截断入口 `py_compile` PASS，SHA256=`43bd5b928702599fb0d5cde869a4f36864f56c5eacc74c44ef337601a5f28f85`；新 run config SHA256=`adfe1e29093883429953e9c7e1e3e4e0ce46d0d95504ac1aa5ae9ec8cb3ca995`，1195 源文件、12 模型文件、33 输入文件逐文件核验 PASS。真实 100 条 fwe tokenizer 扫描仅 `fwe:69` 触发：131078→131072、左删 6，原始输入 SHA256=`d5b2a747bcfb0bdb478188df15a5195754971d044b9098d81d454984d2edaa09`，保留 token IDs SHA256=`028b296c609da7e8882b7725bdb9152d767d99be3ac89e16651f336dbf8ba283`；64K/32K `niah_multivalue` 最大输入分别 65453/32625，无截断。
- 2026-11-12 pc31 GPU0 实测 1 MiB 后补跑链已点火，driver/producer=2407980/2407982，首个 128K fwe producer 显存约 49494 MiB，日志已进入 100 条生成，run config 身份复核为 `adfe1e29…c995`。新 status/log 为 `ruler_minmax_leftcap_20261112_status_pc31_gpu0.json` 与同级 `..._driver_logs_pc31_gpu0/`，旧失败现场未覆盖。wrapper 与 driver SHA256 仍分别 `14bd8e8d…45cfc`、`0c5819bc…4d8c5`。

- 2026-11-12 pc31 补跑终态 `state=complete, jobs_complete=3/3`，128K `fwe`、64K/32K `niah_multivalue` 均完整提交 100 行，GPU0 释放至 1 MiB。五机结果汇集后共 **33/33 cells、3300/3300 条**，每格 100 个唯一样本；30 格绑定原 config，3 格绑定 leftcap config，prediction/receipt SHA、行数与 `status=complete` 初核全部闭合。旧失败 status/log/partial 保留，未并入完成人口。
- 独立审计器三次自身误判均已修正且未修改预测/receipt：①32K YaRN-off 合法省略 `requested_factor`，改查 `rope_scaling=null, snapshot_preserved=true`；②`100*hits/n` 与官方 `hits/n*100` 可有约 `7.1e-15` 结合顺序差，官方对照改为零相对容差、`1e-12` 绝对容差；③64K `max_inv_freq_error=1.1641532182693481e-10` 是生产验证器明确允许的浮点误差，并非必须为零，改按生产门禁 `inv_freq<=1e-7、cos/sin<=5e-4` 检查。最终审计器 SHA256=`6f475f0ac4baf267b9339d959ec4570d6f6acb34a1aed2068df491636195efb9`。

### 最终结果（独立验收后）

结果目录：`agent_doc/results/ruler_region_selector_minmax_20261111/`。独立只读实现从 3300 条原始预测重算官方 `string_match_all`，六项检查 `implementation/reference_boundary/data_integrity/numerical_sanity/measurement_validity/reproducibility` 全部 PASS，状态为 **`usable-with-scope`**。另以独立汇总交叉检查复核 3300 个唯一 ID、33×100 人口、逐格/宏平均算术及 149 份 manifest 条目哈希，全部 PASS。

| 任务 | 32K | 64K | 128K |
|---|---:|---:|---:|
| niah_single_1 | 100.0000 | 97.0000 | 98.0000 |
| niah_single_2 | 57.0000 | 66.0000 | 38.0000 |
| niah_single_3 | 37.0000 | 74.0000 | 49.0000 |
| niah_multikey_1 | 55.0000 | 53.0000 | 35.0000 |
| niah_multikey_2 | 19.0000 | 12.0000 | 9.0000 |
| niah_multikey_3 | 1.0000 | 0.0000 | 0.0000 |
| niah_multiquery | 61.5000 | 55.0000 | 18.5000 |
| niah_multivalue | 46.5000 | 37.5000 | 5.5000 |
| cwe | 57.8000 | 0.6000 | 2.2000 |
| fwe | 6.6667 | 3.0000 | 12.3333 |
| vt | 0.0000 | 0.0000 | 0.0000 |
| **AVG11** | **40.1333** | **36.1909** | **24.3212** |

最终证据哈希：`summary.json=fd345bbb27d0c9ad5d7d4dbf2c2742f3d2be783d758b764d521fcc682b651348`、`validation_record.json=5d1ee945c76ad59aef487abab0053874942bc56656f032d522c344dfb4689109`、`artifact_manifest.json=763e28218f3cbc3d82e030ba9321df6c5f602724ae5d5037bb36d44f435dd308`、`per_sample_scores.jsonl=65985ac89702929c4b9dd13ceea02d88f5aa6a08ff2e196aa9d2805e5cf282a7`。

### 范围与限制

1. 本任务按用户最新范围只交付 Single-pool MinMax；FullKV、Single-pool Avg、NFA 的历史完整或 partial 输出均不并入本轮结论，也不据此作跨臂优劣判断。
2. 独立验收是不同实现的只读进程，不冒充第二位人类 reviewer；未重新执行 3300 条 GPU 生成，可复现性限于冻结身份、真实 tokenizer 变换、官方指标和原始证据交叉核对。
3. 结果仅支持本次 Qwen3-8B、Single-pool MinMax、RULER 三长度准确率口径，不支持性能主张，也不外推到其他模型、任务或稀疏臂；`usable-with-scope` 不等于绝对无 bug。
