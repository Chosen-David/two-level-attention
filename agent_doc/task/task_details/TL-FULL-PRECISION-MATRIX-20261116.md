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
- Qwen3-8B FASA LongBench v1的`pred_fasa_1024`恢复目录完成16/16、3750/3750、strict runcap、SHA和独立staging评分，AVG16=33.31，且与`exp/trace/results/fasa_lb_16.json`逐任务一致；输入身份只有`legacy_strict`。另一份AVG16=35.15报告引用的`qwen3_8b_longbench_fasa_vendor.json`及原始generation尚未定位，作为独立report-only恢复候选保留，不覆盖也不混入33.31证据。
- Qwen3-32B AsyncTLS LongBench v1正式summary已经确认3750/3750、16/16、v2输入身份和独立重评分通过；`TASK.md`旧1150/3750只是过时进度，不重跑。
- MoBA旧`e89_moba_full.json`实际只有13/16任务，原始generation已定位并完成3150行及13项SHA256副本核验，缺`samsum/trec/passage_count`且无run-cap sidecar。canonical `difflib`重评分LCC/RepoBench=67.68/66.67、AVG13=49.04；显式`Levenshtein 0.27.5`复算为68.98/68.13、AVG13=49.19，逐任务吻合旧报告，证明差异来自已披露的代码相似度后端而非prediction变化。两种评分口径分列，不能算完整LongBench v1。

### 立即可执行缺口

- Qwen3-8B RULER MultiKey2的C2 near/far=minmax/minmax与C3 avg/avg，三长度各100条，共600请求；既有avg/minmax只作配对参考。
- SparQ 64K存在跨producer script漂移，128K MultiKey1只有96/100且无committed pointer。为保持同一代统一协议，64K/128K均按当前producer全11任务重跑，不把4行嫁接到旧generation。
- HISA和ClusterKV 64K/128K没有完整闭集，按各11任务×100补齐；32K历史完整结果复用。
- 现行NFA LongBench质量入口只接受Qwen3-8B英文16任务，不能把旧LongBench v2 full/mavg/aavg结果改标为当前NFA；该组合先记实现扩展缺口。
- 非Qwen3-8B官方Quest/JSSA RULER入口当前有明确模型限制；其他vendor方法也必须逐个验证真实adapter，不能由Qwen3脚本存在推断四模型都可运行。

## 派单实现

- `agent_doc/results/full_precision_matrix_20261116/_dispatch/run_regional_chain.sh`：落盘的C2/C3三长度×双分片链，避免远程内联shell变量提前展开。
- `agent_doc/results/full_precision_matrix_20261116/_dispatch/run_ruler_vendor_gap.sh`：SparQ/HISA/ClusterKV 64K/128K独立输出、固定HEAD、干净worktree、GPU占用检查、全量日志、11任务生成及formal scorer硬门禁。
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
- Qwen3-32B NFA的`passage_count/samsum/trec`三项原始generation、strict runcap、v2输入身份和正式评分已定位；既有13任务仍只有汇总result.json，因此全16任务继续`merge-audit`，不改标完整复用。
- 历史报告将GLM/Qwen3-32B Quest/JSSA原始现场精确定位到pc14 `/tmp/tli-baseline-jssa-quest`；pc14当前与历史地址均SSH超时，其余四机精确检索无匹配，四套RULER generation继续`recover`，不把“当前不可达”写成“从未运行”。
- 最新实测pc33/pc02/pc40/pc31/pc32全部GPU均有其他compute app，本任务正式生成请求数仍为0；未触碰任何其他进程，也未把资源阻塞登记为实验完成。

## 下一步

1. 用落盘链脚本在真实空闲卡点火C2/C3，30秒内检查进程、compute app/显存和日志。
2. 同步SparQ/HISA/ClusterKV锁定vendor资产并做一样本隔离smoke，通过后执行64K/128K全量补格。
3. 继续恢复GLM/Qwen3-32B Quest/JSSA原始generation和Qwen3-32B NFA LongBench既有13任务原始generation；FASA两个来源及MoBA双评分后端已分列闭合。
4. 扩展现行NFA LongBench v2真实入口，逐方法核验其余模型支持；不可运行组合保留明确实现阻塞。
5. 完成官方评分、输入身份、runcap、SHA256与生产侧审计后更新矩阵和`CURRENT_NARRATIVE.md`索引；独立验收前不作最终质量结论。
