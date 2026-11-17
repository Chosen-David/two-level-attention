# TL-FULL-PRECISION-MATRIX-20261116

## 目标与范围

盘点并补齐 GLM-4.7-Flash、Llama-3.1-8B、Qwen3-8B、Qwen3-32B 的 FullKV、NFA 与全部现行 baseline 在 LongBench v1 英文16任务、LongBench v2、RULER 11任务×32K/64K/128K 的精度证据。完整人口分别为3750、503、每模型方法3300请求；MultiKey2单任务不能冒充完整RULER。

方法范围为 FullKV、NFA、Quest、JSSA、AsyncTLS、MInference、SparQ、MoBA、ClusterKV、HISA、FASA。官方实现、vendor适配、论文语义复现和历史TLI是不同implementation identity，不混池。所有新恢复或生成结果在不同owner独立验收前保持 `pending-independent-verification`。

## 已核实结论

并非所有模型、方法和数据集已经有数据。机器可读矩阵见 `agent_doc/results/full_precision_matrix_20261116/summary.json`，逐格区分 `reuse/rescore-only/merge-audit/rerun/recover/implementation-blocked/resource-blocked`。

### 本轮恢复与生产侧重评分

- Llama FullKV与历史K1=128 TLI/NFA的RULER三长度原始现场已恢复，六个单长度root均通过11/11任务、每任务100条正式评分。FullKV AVG11=85.53/77.67/71.63，NFA=86.08/76.16/61.05；128K FullKV `niah_single_1=25` 和NFA multikey退化原样保留。
- GLM 32K另一份完整历史TLI（ABD+layer-skip）通过11/11×100正式评分，AVG11=19.94；它不是当前gate-off NFA，也不与GLM full/dual的35.01/34.82混池。
- Qwen3-32B 32K冠军历史TLI通过11/11×100正式评分，AVG11=50.70，与旧报告一致。
- Qwen3-8B扩展baseline的15个历史RULER运行归档完成16500条prediction的SHA、ID/answers和官方分复算；其中3个为替代/重复generation，不增加唯一覆盖。
- Qwen3-8B FASA LongBench v1的`pred_fasa_1024`恢复目录完成16/16、3750/3750、strict runcap、SHA和独立staging评分，AVG16=33.31，且与`exp/trace/results/fasa_lb_16.json`逐任务一致；输入身份只有`legacy_strict`。另一份AVG16=35.15的汇总source file已定位到`/home/wangyuanshuo02/sglang/exp/trace/results/qwen3_8b_longbench_fasa_vendor.json`（SHA256=`cf8659bb4b19bd000357475e32a324ce3152f4d8686ce44d7ef260fa1baeb098`），但原始generation尚未定位，仍作为独立report-only恢复候选保留，不覆盖也不混入33.31证据。
- Qwen3-32B AsyncTLS LongBench v1正式summary已经确认3750/3750、16/16、v2输入身份和独立重评分通过；`TASK.md`旧1150/3750只是过时进度，不重跑。
- Qwen3-32B历史TLI LongBench v1既有13任务原始generation已从`host_backup_187/host_backup_wys2`恢复，3150行、四副本SHA一致；canonical `difflib` AVG13=41.854615并逐任务吻合旧`result.json`。补缺3项为另一generation且有strict runcap/v2输入身份；legacy13无sidecar且原始记录无`_id`，所以16个任务名虽齐仍保持`merge-audit`，跨代算术AVG16=41.899375只作参考，不能冒充统一协议正式臂。
- MoBA旧`e89_moba_full.json`实际只有13/16任务，原始generation已定位并完成3150行及13项SHA256副本核验，缺`samsum/trec/passage_count`且无run-cap sidecar。canonical `difflib`重评分LCC/RepoBench=67.68/66.67、AVG13=49.04；显式`Levenshtein 0.27.5`复算为68.98/68.13、AVG13=49.19，逐任务吻合旧报告，证明差异来自已披露的代码相似度后端而非prediction变化。两种评分口径分列，不能算完整LongBench v1。

### 立即可执行缺口

- Qwen3-8B RULER MultiKey2的C2 near/far=minmax/minmax与C3 avg/avg，三长度各100条，共600请求；既有avg/minmax只作配对参考。
- SparQ 64K存在跨producer script漂移，128K MultiKey1只有96/100且无committed pointer。为保持同一代统一协议，64K/128K均按当前producer全11任务重跑，不把4行嫁接到旧generation。
- HISA和ClusterKV 64K/128K没有完整闭集，按各11任务×100补齐；32K历史完整结果复用。
- 现行NFA LongBench质量入口已复用共享`run_quality()`扩展Qwen3-8B `lbv2`唯一任务门禁，不能把旧LongBench v2 full/mavg/aavg结果改标为当前NFA；主机契约测试和分片脚本已完成，仍需真实GPU smoke及503请求生成。
- 非Qwen3-8B官方Quest/JSSA RULER入口当前有明确模型限制；其他vendor方法也必须逐个验证真实adapter，不能由Qwen3脚本存在推断四模型都可运行。

## 派单实现

- `agent_doc/results/full_precision_matrix_20261116/_dispatch/run_regional_chain.sh`：落盘的C2/C3三长度×双分片链，避免远程内联shell变量提前展开。
- `agent_doc/results/full_precision_matrix_20261116/_dispatch/run_ruler_vendor_gap.sh`：SparQ/HISA/ClusterKV 64K/128K独立输出、固定HEAD、干净worktree、GPU占用检查、全量日志、11任务生成及formal scorer硬门禁。
- `agent_doc/results/full_precision_matrix_20261116/_dispatch/run_lbv2_nfa.sh`：Qwen3-8B NFA LBv2的隔离一样本smoke与503请求确定性分片；固定HEAD、干净worktree、实测GPU、完整日志、逐分片runcap/source manifest评分，评分目录按分片隔离以避免跨分片ID误合并。
- C2/C3固定producer为`9f3e8057acd528a904aa8883b830d541b389e39f`，输出与历史目录隔离。vendor资产位于被忽略的`exp/vendor_baselines`，clean worktree不会自动携带，必须在点火前同步并记录哈希；ClusterKV还需要`ckv_env.sh`。

## Progress

### 2026-11-16 资产盘点与第一次点火尝试

- 四台远程机曾实测10卡空闲，但随后pc02/pc40/pc31出现其他计算PID；实时状态优先，未触碰这些进程。
- pc32目标commit对象存在，已创建`/tmp/nfa-dac-ruler-regional-9f3e8057` detached clean worktree，HEAD和status核验通过，没有修改pc32主仓的未提交改动。
- 首次使用远程内联循环启动C2/C3时，外层shell提前展开`CELL/SHARD`，两条链均在usage参数检查退出；未生成prediction、未创建正式manifest、未有点火后的本任务worker。失败chain日志保留。
- 随后pc32也出现其他计算PID，因此当前没有本任务正式生成在跑。改为落盘链脚本，待重新实测空闲后执行；不得把资源阻塞写成实验完成。

### 2026-11-16 历史来源闭合与恢复阻塞定位

- FASA两个LongBench v1来源已分列：`pred_fasa_1024`的3750请求、16任务、strict runcap、SHA与独立staging重评分均通过，AVG16=33.31并与`fasa_lb_16.json`逐任务一致；AVG16=35.15的另一份报告所引source file/generation仍未定位，不覆盖或混入前者。
- MoBA原始13任务、3150行及13项SHA256副本一致；缺`samsum/trec/passage_count`且无run-cap sidecar。canonical `difflib` AVG13=49.04，显式`Levenshtein 0.27.5` AVG13=49.19并逐任务吻合旧报告，差异仅为代码相似度评分后端，两种口径均保留。
- Qwen3-32B NFA既有13任务随后也在两份本地备份中定位并恢复，3150行、四副本SHA一致，canonical复评逐任务吻合旧汇总，AVG13=41.854615；补缺三项strict runcap/v2输入身份仍属另一generation。合并审计落盘为`ak_verify_results/full_matrix_recovered_20261116/audits/qwen32b_nfa_lbv1_merge_audit.json`（SHA256=`a81474457f03b1bd3c8d45377c31c97610ca9c207dacc0a569e0b5e3f02b14ce`）；legacy13无sidecar/_id，全16任务继续`merge-audit`，跨代算术AVG16=41.899375仅作参考。
- 历史报告将GLM/Qwen3-32B Quest/JSSA原始现场精确定位到pc14 `/tmp/tli-baseline-jssa-quest`；pc14当前与历史地址均SSH超时，其余四机精确检索无匹配，四套RULER generation继续`recover`，不把“当前不可达”写成“从未运行”。
- 最新实测pc33/pc02/pc40/pc31/pc32全部GPU均有其他compute app，本任务正式生成请求数仍为0；未触碰任何其他进程，也未把资源阻塞登记为实验完成。

### 2026-11-16 NFA LBv2入口扩展与FASA来源定位

- FASA AVG16=35.15的汇总source file已定位到`/home/wangyuanshuo02/sglang/exp/trace/results/qwen3_8b_longbench_fasa_vendor.json`，SHA256=`cf8659bb4b19bd000357475e32a324ce3152f4d8686ce44d7ef260fa1baeb098`；当前只恢复汇总，原始generation、runcap和source manifest仍未定位，继续与33.31完整generation分列。
- `benchmark/efficiency/benchmark_longbench_quality.py`新增与官方入口一致的`lbv2`任务门禁：允许唯一任务`lbv2`，保留英文16任务，拒绝重复、未知任务及v1/v2混跑。生成继续复用共享`run_quality()`，没有新造预测、选项解析或评分协议。
- 新增NFA任务门禁测试，并复跑共享LBv2官方评分、完整SKIP/分片ID和round-robin合并契约；普通与`python -O`均为29/29通过。首轮唯一失败是评分子进程环境缺`jieba`，补入既有`local_pylibs`后同组测试全过，未改代码规避。
- 新增`run_lbv2_nfa.sh`隔离smoke/formal分片脚本，`bash -n`、ShellCheck和diff格式检查通过；每分片独立评分目录避免并行shard被同一source manifest误判。尚未真实GPU smoke，矩阵仅改为`implementation-ready-pending-gpu-smoke`，没有生成新精度结果。
- 本轮再次实测pc33/pc02/pc40/pc31/pc32全部卡仍有其他compute app，未点火；正式请求数保持0。

### 2026-11-16 MoBA缺失任务候选的跨身份排除

- 在`results_longbench_moba`和`harvest_pc40_20261022`新定位到`samsum/trec/passage_count`的`moba_vendor`原始JSONL，但它们不属于e89既有13任务generation：同一HotpotQA首条中，e89为`Mi v. California`且无`_id`，vendor文件为`Miller v. California`且有`_id`；文件名身份也分别为`tli_64_128_1024_c4_A-moba`与`moba_vendor_c256_k4`。
- e89同身份的`samsum/trec/passage_count-tli_64_128_1024_c4_A-moba.jsonl`在本机全部备份、harvest和恢复目录中精确检索均为0。故MoBA继续保持13任务部分覆盖，不把另一generation的三项嫁接成AVG16。配对record、三项候选路径/行数/SHA及精确检索范围已落盘为`ak_verify_results/full_matrix_recovered_20261116/audits/moba_missing_task_identity_audit.json`，SHA256=`dfef58404df6a16fe5ced739a1a6ad2108f8ecafbea1723426d18e4046156dc0`。

### 2026-11-17 LBv2真实smoke、正式分片与C3点火

- 资源复测一度发现8张空卡，但实际点火前pc02四卡、pc31双卡及pc33双卡陆续被其他compute app占用；只在仍实测空闲的pc32 GPU0/1派单，未触碰其他进程。pc14两次直接`sshpass`密码登录均为连接超时，Quest/JSSA继续保持recover。
- pc33 LBv2首次smoke在GPU门禁发现新占用后退出，未创建日志/prediction。pc32旧producer `/tmp/nfa-dac-ruler-regional-9f3e8057`不含后新增的`run_regional_chain.sh`，首次C2/C3启动立即报文件不存在且无prediction；将同一脚本投送到`/tmp/run_regional_chain_9f3e8057.sh`后重新点火，producer仍固定为`9f3e8057`，没有修改旧工作树。
- C3（near/far=avg/avg）已在pc32 GPU1启动三长度双分片链，当前32K shard0/2共50请求；进程PID=2143738，显存约25–30GB，日志已出现多条逐请求`score=100.000000`。C2尚未点火，不能与既有avg/minmax混池。
- pc32缺少最新LBv2 producer对象且其origin镜像未同步，使用本地`9f3e8057..HEAD` git bundle传输5个提交并创建干净detached worktree，HEAD=`b3ee770f8865549ef7271bc83f73968a328950d1`。pc02/pc31/pc40也已准备同HEAD工作树，GPU占用不改变。
- Qwen3-8B NFA LBv2真实单样本smoke已在pc32 GPU0完成生成：原始长文tokenization出现284105>131072警告，共享`build_input()`随后按31500 cap截断，实际输入31512、输出23。首轮生成成功但评分子进程缺`jieba`；从本机`local_pylibs`复制同版`jieba/rouge`后仅重跑官方评分，1/1、official-strict、strict runcap和v2输入身份全PASS，不重复生成。smoke单分不是503人口精度结论。
- smoke通过后已启动正式`world=8`的shard0（63请求），进程PID=2161646、显存与日志三重验证通过；其余7片只保留明确待派状态，未写成已生成。正式输出与smoke隔离，每片评分目录也隔离。
- SparQ关键patch在本机/pc02/pc40的SHA256一致（`qwen3_sparq_patch.py=97040f44...`、`sparq_attention_official.py=b0f8929b...`）；HISA已从pc02恢复（`qwen3_hisa_patch.py=96186a77...`），ClusterKV vendor和`third_party/ClusterKV`已从pc40恢复（`ckv_env.sh=c120d745...`、patch=`bb5f58ca...`）。`run_ruler_vendor_gap.sh`新增可选`smoke|formal`模式，smoke只跑`niah_single_1`一条并使用独立输出/锁/评分门禁；已通过`bash -n`、ShellCheck和diff检查，尚未真实GPU smoke。

### 2026-11-17 LBv2八片并行、C3推进与ClusterKV环境恢复

- LBv2正式`world=8`现已全部互斥点火：pc32 GPU0=`0/8`，pc02 GPU0/1/2/3=`1/2/3/7`，pc31 GPU0=`4/8`，pc40 GPU0/1=`5/6/8`。除shard7为62条外其余各63条，总计503。每片均有独立Python、compute app/显存和启动日志；启动后行数持续增长，没有把pid存在单独当作健康证据。
- LBv2全部8片已完成63/63（shard7为62/62）、逐片官方strict评分及本机收割，分片score依次34.92/30.16/31.75/34.92/38.10/19.05/31.75/37.10；所有分片strict_runcap=1、v2_identity=1。现有`merge_longbench_quality.py`已校验8个rank、源顺序、官方ID、strict choice、输入身份、producer和runcap，canonical prediction恰为503个唯一官方ID。隔离目录官方评分为32.21，生产侧重算`pred_choice`命中162/503=32.20675944333996%；cap=31500，44项证据SHA复核PASS。结果与逐片哈希归档为`agent_doc/results/full_precision_matrix_20261116/qwen3_8b_nfa_lbv2.json`（SHA256=`62a9828d50c7a5c7a3f1f3531f766eee9203512110deef4bca50f25a873f4c6a`），继续pending-independent-verification，禁止重复生成已闭集503请求。
- C3在pc32 GPU1已完成32K和64K各两片，共200条；128K shard0已于10:52:45启动，新Python PID=2213090，GPU1显存约47GB，模型加载日志正常，链内剩余128K shard1。C2尚未点火。pc32 GPU0在shard0完成后被另一会话gate-sensitivity worker接棒，pc31/pc40/pc33其他计算进程也均保持不动。
- C3的32K/64K完整证据已收割并通过`audit_ruler_results.py`生产侧正式审计：每长度100个source index恰好闭合，逐样本官方分、输入身份、runcap、prediction/receipts/scores/source manifest SHA及36层全部keep均PASS；official_score分别80.0/78.0。审计=`ak_verify_results/nfa_dac_ruler_regional_9f3e8057_20261116/_audits/c3_32k_64k_production_audit.json`，SHA256=`0ef40778577e5e1b07e390be01190e819b1b89d205d57a4b5aca090bd19b4905`；仅为MultiKey2单任务且继续pending-independent-verification，128K尚未闭集。审计器仅新增允许实际producer `9f3e8057`，未放宽任何数据门禁。
- 四台远端已用bundle部署独立clean producer `/tmp/full-precision-matrix-cc480756`，HEAD=`cc48075695a827840c25d35df917df3a1079838d`，并同步被忽略的SparQ/HISA/ClusterKV源码资产，关键文件SHA一致。pc02的4.5GB ClusterKV环境已恢复，`libcublas.so.12`就位，`torch 2.8.0+cu128 + pylibraft.cluster + rmm`真实导入PASS；这只解除环境阻塞，三方法仍未完成真实GPU smoke。
- 最新GPU复核没有发现可安全新增任务的空卡；vendor smoke和C2保持待派，不能以环境就绪代替生成结果。LBv2八片齐后复用现有`merge_longbench_quality.py`按source index恢复503顺序，重建canonical runcap/source manifest，再在隔离目录用官方strict scorer评分；所有结果继续`pending-independent-verification`。
- 11:30再次复测发现pc33 GPU0已空闲，本机clean producer `/tmp/nfa-dac-ruler-regional-9f3e8057`的HEAD=`9f3e8057acd528a904aa8883b830d541b389e39f`、工作树、模型和RULER数据预检均通过，C2输出目录为空。已从当前main的链脚本启动C2，实际生成仍使用旧producer内runner；32K shard0的runner PID=606271、Python PID=606301，GPU0约25GB显存且有compute app，模型加载日志正常，三重验证通过。链内后续250请求保持待跑，未写成完成；pc33 GPU1及其他机器上的计算进程均未触碰。
- 随后pc33 GPU1一度释放，本机已创建独立clean producer `/tmp/full-precision-matrix-cc480756`并同步SparQ/HISA/ClusterKV资产；HEAD=`cc48075695a827840c25d35df917df3a1079838d`，SparQ/HISA关键patch SHA256分别与锁定值`97040f44...`/`96186a77...`一致。SparQ 64K smoke正式点火时GPU门禁发现新compute PID=1896650并按设计退出，未创建正式smoke日志、prediction或done标记，不能记为smoke通过；未触碰该占用进程，也未持续重试抢卡。

## 下一步

1. LBv2 503已完成canonical merge、官方strict评分和44项SHA复核，不再重复生成；保留pending-independent-verification并交不同owner验收。
2. 保持pc32 C3 128K链运行，资源允许时启动C2；两臂完成后分别合并三长度各100条并正式评分，不与avg/minmax混池。
3. 将SparQ/HISA/ClusterKV锁定资产同步到独立新producer worktree，逐方法执行64K/128K单样本smoke，通过后执行各11任务×100正式补格。
4. 继续恢复GLM/Qwen3-32B Quest/JSSA原始generation及FASA 35.15原始prediction；Qwen3-32B NFA若需统一协议16任务正式臂则重跑legacy13，不直接拼接13+3。
5. 逐方法核验其余模型与数据集支持；完成官方评分、输入身份、runcap、SHA256和生产侧审计后更新矩阵及`CURRENT_NARRATIVE.md`，独立验收前不作最终质量结论。
