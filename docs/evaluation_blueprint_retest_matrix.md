# Near-Far Attention 评测蓝图差异与历史结果处置清单

日期：2026-10-30。本文是协议修复与重测清单，不是新质量或性能结果。审计依据为目标
`evaluation_blueprint_audit_20261030/BLUEPRINT.md`、独立审计
`protocol_independent_audit.md`、实际源码和既有 manifest。历史预测、日志和判决保持不可变；
“可复用”只表示在列明范围内可作为证据，不表示满足新主协议。

## 1. 修复分支与当前证据边界

| 分支 | 提交 | 已落地的代码/CPU 证据 | 尚未证明 |
|---|---|---|---|
| `fix/near-far-prefix-protocol` | `5d86381`, `4fa5bbc` | canonical Quest 输入、prefix freeze、suffix replay、逐步 attended-K、16 任务宏平均及结果链回归 | 真实 Qwen/GLM 正式 16 任务质量 |
| `fix/near-far-evaluation-blueprint` | `2e388e5` | Qwen original=32768、runtime capacity 分离、实际 RoPE 检查；identity 10/10、binding/locking/consumer 31/31 | corrected YaRN 的正式 64K/128K 生成与质量 |
| `feat/frozen-eager-physical` | `069c796` 至 `53493fe` | RULER opt-in 生命周期、真实 CLI 校验、四态物理 cache、eager 最终支持集 attention、GPU harness original=32768；相关 27 项回归及 15 个子测试通过 | 8K 数值差异归因、32K/64K/128K 正确性/性能验收 |

公共/Qwen 回归 69 项及 Transformers 5.12.1 下 GLM 回归 16 项已在 Quest 修复分支通过。
这些是代码与小模型证据，不能替代真实模型 GPU 验收。corrected YaRN 分支与 frozen-eager
分支尚需集成后复验；不能拿其中一个分支的测试给另一个分支的端到端运行背书。

## 2. 蓝图差异与处置状态

| 项目 | 审计发现 | 当前修复/协议 | 状态与剩余门禁 |
|---|---|---|---|
| Qwen YaRN | 把 runtime 40960 错当 `original_max_position_embeddings` | 32K 不额外启用；64K factor=2、128K factor=4；original=32768；runtime max 覆盖输入+输出；比较实际 inv_freq/scaling/旋转结果 | 代码和 CPU 已修；所有受影响的 64K/128K 正式结果重生成 |
| GLM RoPE | 不得套用 Qwen YaRN | 保留发布快照 rope 配置，禁止 Qwen 参数注入 | Qwen 修复不要求 GLM 因此重跑；新 Quest 质量仍需独立生成 |
| Quest 输入 | 分段分词、丢 BOS/跨界 BPE，full-prompt 冻结冒充 prefix 冻结 | 完整渲染文本只分词一次；offset mapping；跨界 BPE 整体归 suffix；`prefix+suffix==full` | 已修并有回归；所有新主协议预测重生成 |
| 冻结时机 | full prompt profile 后 rewind/replay | 材料前缀 dense prefill/profile/freeze，再逐 token replay suffix，最后正常答案生成；冻结后不重新标定 | 已修；validator 核验 freeze 位置、相位和绝对位置 |
| LongBench V1 | 五任务 AVG5 与 16 任务主表混杂 | 固定 16 个 English/code 任务，按 manifest 任务集合等权宏平均；不再为新结果生成 AVG5 | 分析/导出已修；正式 16 任务结果未运行 |
| 输入预算 | 把 31500 称为严格最终 32768 token | 明确为 chat template 前 raw prompt 截断；记录最终输入长度和生成余量 | 文档与 manifest 已修；正式运行仍须逐行身份验证 |
| no-chat/停止规则 | 任务模板和 SAMSum 停止行为不统一 | TREC/TriviaQA/SAMSum/LCC/RepoBench-p 不套 chat；SAMSum EOS+newline、首 token EOS suppression | 已修并有测试；真实生成需复核 |
| LongBench V2 | 非官方 fallback 可把冠词 `a` 当选项 A | `official-strict-v1` 只接受官方 direct-answer 两种格式；full503 capped 与 Short/full-context 分开；不混入 CoT | 解析器已存在；历史预测是否只需重评分须按第 3 节逐 run 判定 |
| RULER | frozen 方法无生产入口，或把已知 token replay 当生成 | B=1 eager opt-in；完整 prompt prefill 后正常 greedy decode；32K/64K/128K、11 任务×100；last-50 replay 只能是独立压力测试 | 入口已接入；正式矩阵和 corrected YaRN 集成复验未运行 |
| 完整 near/far 方法 | 仅 far skip/两级筛选代替完整方法 | 独立 near/far gate + sink/far/near/SWA 生命周期；near-only 是不可见 staging，不宣称 KV 驻留节省；far/both 物理删除 | 四态实现存在；真实 Qwen 数值和跨请求/ring-wrap GPU 验收未闭合 |
| 真实预算 | K1/K2、mask block 或 visible KV 被当作 attended-K | frozen TLI 逐步、逐层、逐 KV-head 记录最终唯一有效 attended keys；K1=粗筛、K2=细筛预算 | 本方法结果链已修；Quest/SparQ/FFD 等基线的原生修正项、量化、强制区和去重计数仍需补齐 |
| 效率公平性 | 用理论 mask/KV 字节或 replay token 宣称加速 | 完整 prompt prefill + 正常 decode；分别报告 prefill/decode/e2e，包含 gate/index/cache 转换；Graph 预热/捕获与稳态分开 | frozen harness 只有部分总量字段；KV/index/metadata/workspace/peak 分解及跨方法公平比较仍未完成 |
| 模型覆盖 | Qwen 结果外推 GLM/32B | 主模型 Qwen3-8B GQA、GLM-4.7-Flash MLA；Qwen3-32B 只作规模扩展 | GLM 仅有集成/小模型证据；真实质量矩阵未运行 |

## 3. 历史结果四类处置

### 3.1 可复用，但仅限原适用范围

| 历史证据 | 可复用范围 | 不能用于 |
|---|---|---|
| Qwen 32K、无额外 YaRN 的 full-prompt RULER/效率结果 | 仅在源码、输入 IDs、停止规则、模型/权重、K1/K2、runtime 和独立验证全部匹配时复用 | old one-shot LongBench 不得变成新 Quest 主表；缺身份字段时转入“无法判断” |
| GLM 发布 rope 下的四态/consumer/真实权重 smoke | 原生 MLA/RoPE、cache 生命周期和集成正确性证据 | 不证明 GLM LongBench/RULER 质量，也不证明加速 |
| K1=64 的 E168/E169/E172–E174 等结果 | K1 消融臂 | 当前 K1=128 生产主表；不得按性能比例折算成新主表 |
| 旧 one-shot full-prompt 五任务 frozen 结果（含 `frozen_near_far_full5_independent005_20261029`） | 明确标注的历史 alternate protocol | 新 Quest prefix-freeze、16 任务宏平均或 AVG5 主结论 |
| 旧 one-shot 16 任务 frozen 结果（含 `frozen_near_far_v1_16_independent005_20261030`） | 原 one-shot 协议的历史质量与工程诊断 | 新 Quest 主协议；即使任务集合相同也不能绕过重新生成 |
| 已验证的 negative/inconclusive 记录 | 说明某配置无效、无收益或尚未归因 | 改写成正向等价/加速证据 |

### 3.2 只需重评分或补充说明，且必须先满足不变量

| 历史证据 | 处置条件 |
|---|---|
| LongBench V2 direct-answer 不可变预测 | 若输入 IDs、生成参数、模型和预测文本均未变，仅解析器从宽松 fallback 改为 `official-strict-v1`，可重评分；保留旧分数并新增 parser 版本 |
| capped full503 结果 | 可保留原预测，但表头/报告必须明确 capped full503，不得标成官方 Short 或 full-context；缺样本数/任务身份时先补 manifest，不可靠说明补成另一协议 |
| 仅统计口径变化的 V1 预测 | 只有输入、相位、停止和预测均符合同一协议时，才能重算 16 任务宏平均；旧五任务缺任务不能靠重评分补齐 |
| 仅报告字段缺失的效率原始事件 | 只有原始 trace 已含对应事件和内存计数时可重新汇总；不存在的数据不得从理论公式补写 |

CoT 预测不能直接当作 direct-answer 主结果重评分；需要独立 CoT 表，或按 direct-answer
协议重新生成。

### 3.3 必须重生成

| 范围 | 原因 |
|---|---|
| 所有 Qwen 64K/128K、original=40960 的 Full Attention 与稀疏/基线结果，包括 `frozen_qasper_mk2_64k_20261030` | RoPE 实际数值已变；config 自洽或少量 token 相同不证明分布等价，所有对照统一重测 |
| 所有新 Quest prefix-frozen LongBench 主结果 | 旧 one-shot full-prompt 与材料前缀冻结是不同计算协议；不能复用旧生成或只重评分 |
| 采用旧分段分词、BOS 删除或跨边界 BPE 的结果 | canonical token stream 和切分边界改变 |
| 改变停止规则、chat template、thinking 开关、生成余量或样本集合的结果 | 生成身份改变 |
| 当前 K1=128 生产质量/效率主表中只有 K1=64 原始数据的格子 | 预算和运行时路径不同；需要实测，不做解析折算 |
| frozen eager 数值/性能验收 | 既有 8K 数值差异未归因，且旧 physical 臂更慢；修复后须重做四态、中间选择、逐层/logits/生成对拍和性能测量 |
| 声称物理 KV/访存/计算收益的新主结果 | 理论 mask、visible-KV proxy 或逻辑长度 scratch 不能证明物理节省；需要真实 attended-K、内存分解和计时 |

### 3.4 无法判断，禁止默认复用

- 缺源码 commit、模型/权重 SHA、数据 SHA、输入 IDs、样本顺序、factor/original/runtime max、停止规则或生成参数的历史行。
- 只有 K1=K2 相同，却没有最终强制保留、去重、mask 后唯一有效 key 数的基线；不能宣称 matched budget。
- Quest 当前块保留、SparQ V-mean correction、FFD 阈值/量化等原生语义未记录时，不能通过改成统一 top-k 制造“公平”。
- 无锁卡/共居证据、Graph 私有池跨 run 污染或 warmup/capture 混入稳态的旧性能行，正式性能复用状态未知。
- 只有文件存在、旧 pass 或独立文件身份检查，没有对应生成/质量/预算验证的结果。

## 4. 可运行检查与正式生成入口

Quest 16 任务完整生成（`--samples 0`；正式运行前必须锁卡并记录 GPU hygiene）：

```sh
TLI_SCORER_BACKEND=difflib python -m benchmark.LongBench.prefill_frozen_skip \
  --attention tli --thresholds 0.05 --samples 0 \
  --model /path/to/Qwen3-8B --data /path/to/LongBench/data \
  --device cuda:0 --verify-native-full \
  --out agent_doc/results/frozen_region_skip_RUN_ID
python -m benchmark.LongBench.validate_frozen_skip \
  agent_doc/results/frozen_region_skip_RUN_ID --data /path/to/LongBench/data
python -m benchmark.LongBench.analyze_frozen_skip \
  agent_doc/results/frozen_region_skip_RUN_ID
python -m benchmark.LongBench.export_frozen_skip \
  agent_doc/results/frozen_region_skip_RUN_ID
```

RULER frozen eager 单任务入口如下；正式矩阵对 11 个任务和三个长度逐一执行，64K/128K
加 `--yarn`，32K 不加。必须在集成 corrected YaRN 后使用，且每个新方法/配置采用新输出目录：

```sh
python -u -m benchmark.RULER.pred_ruler \
  --model Qwen3-8B --model_path /path/to/Qwen3-8B \
  --task niah_single_1 --context_length 65536 --method tli --yarn \
  --data-root /path/to/KVCache-Factory/data/RULER \
  --output-dir /path/to/ruler/RUN_ID --max-num 100 \
  --tli_prefill_frozen_regions --tli_frozen_region_mode dual \
  --tli_frozen_region_epsilon 0.05 --tli_frozen_region_statistic mean \
  --tli_frozen_region_rows 256 \
  --tli_far_method minmax --tli_near_method avg \
  --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 \
  --tli_subspace tail --tia_level2_cmp_ratio 4 \
  --tli_sink_tokens 4 --tli_swa_size 128 \
  --tia_level1_topk 128 --tia_level2_topk 1024 \
  --tli_enable_layer_skip false \
  --tli_far_skip_backend legacy --tli_far_skip_cache retain
```

`--tli_dyn_dgate` 是 `store_true` 开关，保持缺省即为 false，不能传字面量 `false`；
入口会 fail-closed 检查该值及完整几何。

真实 Qwen paired harness 独立于质量主表，用于 correctness/performance/profiler 原始记录；
其结果不得自动宣称等价或加速：

```sh
python exp/bench/check_qwen_frozen_eager.py \
  --model /path/to/Qwen3-8B --data-root /path/to/RULER \
  --task niah_single_1 --context-length 65536 --row 0 --max-new 8 \
  --states keep,far,near,both --mode all --reps 3 --profile \
  --out /path/to/frozen-eager/RUN_ID
```

## 5. 尚未执行的实验与解锁条件

1. corrected original=32768 下 Qwen 64K/128K 的 Full、完整方法及全部对照正式重生成。
2. 新 Quest canonical prefix-freeze 的 Qwen3-8B 与 GLM-4.7-Flash 16 任务质量主表。
3. RULER 11 任务×100、32K/64K/128K 的完整 frozen 方法矩阵；last-50 replay 若做，单独报告。
4. frozen eager 8K 数值归因及 8K/32K/64K/128K 四态、边界、ring-wrap、跨请求、异常恢复、中间选择、逐层/logits/生成对拍。
5. prefill/decode/e2e、真实 attended-K、KV/index/metadata/workspace/peak、Graph setup/steady-state 的公平效率表。
6. Qwen3-32B 仅作为规模扩展；不能替代以上两种主模型架构覆盖。

上述 GPU 工作只有在目标卡实测空闲、无其他计算进程后启动；性能采数使用
`CUDA_VISIBLE_DEVICES` 锁卡并旁路监控。未获得合法空卡、模型或数据时，状态保持未执行，
不得以 CPU 测试、理论计算或旧结果替代。
