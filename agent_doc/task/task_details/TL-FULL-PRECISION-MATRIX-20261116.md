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
- Qwen3-8B FASA LongBench v1恢复目录完成16/16、3750/3750、strict runcap、SHA和独立staging评分；但高层报告与恢复generation分数来源不一致，且输入身份只有`legacy_strict`，所以仍需来源 reconciliation，不能覆盖旧报告。
- Qwen3-32B AsyncTLS LongBench v1正式summary已经确认3750/3750、16/16、v2输入身份和独立重评分通过；`TASK.md`旧1150/3750只是过时进度，不重跑。
- MoBA旧`e89_moba_full.json`实际只有13/16任务，不能算完整LongBench v1。

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

## 下一步

1. 用落盘链脚本在真实空闲卡点火C2/C3，30秒内检查进程、compute app/显存和日志。
2. 同步SparQ/HISA/ClusterKV锁定vendor资产并做一样本隔离smoke，通过后执行64K/128K全量补格。
3. 继续恢复GLM/Qwen3-32B Quest/JSSA原始generation，审计Qwen3-32B NFA LongBench 13+3合并与FASA来源差异。
4. 扩展现行NFA LongBench v2真实入口，逐方法核验其余模型支持；不可运行组合保留明确实现阻塞。
5. 完成官方评分、输入身份、runcap、SHA256与生产侧审计后更新矩阵和`CURRENT_NARRATIVE.md`索引；独立验收前不作最终质量结论。
