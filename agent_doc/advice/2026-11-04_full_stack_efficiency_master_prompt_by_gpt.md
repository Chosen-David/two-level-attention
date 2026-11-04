# Advice：全库推理效率审计与优化终极执行 prompt（vLLM / FlashInfer 对照）

- 作者类别：GPT / AI 源码与原始回执审查。
- 日期：2026-11-04。
- 审查 main：`37a8e5d386fd80dbbef3d775e39990c27f790138`。
- 原始数据归档：`bfe9cc8f4199ee943e79f6c085e3da0f328cc064`，`benchmark/efficiency/results/nfa_flashinfer_20261104/`。
- 状态：待执行方评估、实施和验收。本文发布只增加 advice；没有修改生产代码、重跑 GPU、证明候选更快或改写历史结果。
- 与已有建议的关系：接续 [eager / Graph / GLM 验收建议](2026-11-04_eager_graph_glm_acceptance_by_gpt.md)，以当前源码重新核对状态，合并 64K、调度/cache/Graph、attention 与评测链路建议；保留旧证据，不另建任务体系。
- 适用范围：two-level-attention 的实际推理调用链与评测脚本。本次性能数值限定 Qwen3-8B、BF16、H20-3e、FlashInfer 0.6.14 回执；全库其余条目是注明范围的源码审查，不代表每个文件或模型已实测。后续必须重新核对当前源码与运行配置。

## 1. 已定位的机制，不预设性能结论

“上下文更长时相对 dense 的加速比应更有优势”是待验证的设计目标，不等于我方绝对 ms 必须随长度下降。当前数据还有一个具体实现断崖，不能用泛泛的长上下文开销解释。

### 1.1 分派看的是分配容量，不是有效 prompt 长度

[build_frozen_cache](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/engine/frozen_runner.py#L85-L114) 使用：

`capacity = (ceil((prompt_tokens + max_new_tokens) / page_size) + 8) * page_size`

本轮 page=64、max_new_tokens=512：

| 名义输入 | 实际 prompt token | capacity |
| --- | ---: | ---: |
| 32K | 32621 | 33664 |
| 64K | 65198 | 66240 |
| 128K | 130986 | 132032 |

[selector 分派](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/indexer/nfa_layer_batch_selector.py#L1200-L1276)：

- capacity < 65536：fine 阶段输出 int32 packed key，Top-K 用 2048-slot 分块 partial + merge。
- capacity >= 65536：fine 阶段输出 BF16 logits，Top-K 组装 64-bit key，走单 CTA 整行 `tl.topk`。
- [生产调用](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/indexer/nfa_layer_batch_selector.py#L1562-L1566) 的容量来自 `cache.max_blocks * bs`。所以本轮 64K 从开始就走宽 key 慢分支，不必等有效序列达到 65536。
- 同一分配公式下，prompt 64448 → 64449 对应 capacity 65472 → 65536，是精确边界。
- 本轮 K1=128，near=32 / far=96，page=64，sink=4，SWA=128。候选宽度 R=(96+32+2)×64+4+128=8452，补齐 R_PAD=16384，K2=1024。它并非在 64K 把所有 token 都交给 L2 排序。
- B1、8 个 KV head 时，单层整行路径只发 8 个 CTA；5-tile partial 路径发 40 个 partial CTA 加 8 个 merge CTA。寄存器 spill、occupancy 或 shared-memory 瓶颈尚没有编译器 / Nsight 证据，不能写成已证实原因。

### 1.2 原始 kernel 证据与 E2E 量级一致，但不是同一次计时

固定为 natural / Graph / a1 / repeat 0、36 层调用：

| Top-K 原始 kernel | 32K | 64K |
| --- | ---: | ---: |
| partial + merge | 0.500547 + 0.310974 = 0.811521 ms | 未使用 |
| 单体 filtered Top-K | 未使用 | 15.340393 ms |

原始文件：[32K](https://github.com/Chosen-David/two-level-attention/blob/bfe9cc8f4199ee943e79f6c085e3da0f328cc064/benchmark/efficiency/results/nfa_flashinfer_20261104/operators/natural/niah_multikey_1_s0_32768_graph_a1.json)、[64K](https://github.com/Chosen-David/two-level-attention/blob/bfe9cc8f4199ee943e79f6c085e3da0f328cc064/benchmark/efficiency/results/nfa_flashinfer_20261104/operators/natural/niah_multikey_1_s0_65536_graph_a1.json)。

[完整模型回执目录](https://github.com/Chosen-David/two-level-attention/tree/bfe9cc8f4199ee943e79f6c085e3da0f328cc064/benchmark/efficiency/results/nfa_flashinfer_20261104/e2e) 中 B1 Graph：NFA 32K 9.816 ms → 64K 24.665 ms；FlashInfer 9.827 → 11.304 ms。Top-K 原始差约 14.529 ms，与 NFA TPOT 增加约 14.849 ms 量级接近，支持将宽 key 整行 Top-K 列为首要热点；不能把两个独立测量直接相减，宣称已经完成严格因果归因。

这里 TPOT 是排除前 3 个 decode step 后的稳态中位数；512 个生成 token 对应 511 个 decode step，不含 prefill。重复同一个 RULER niah_multikey_1 sample 0 的 batch 不是独立多请求质量测试。

### 1.3 算子图漏统计，须先修正再比较

32K Graph raw kernels 中存在但 `stage_memberships` 没有计入的项：

- `_nfa_filtered_topk_partial_kernel`
- `_nfa_filtered_topk_merge_kernel`
- `_topk_candidate_prepare_kernel`
- `_candidate_expand_batched_kernel`

64K / 96K / 128K Graph 也漏后两项。32K summary 的 L2 约 0.938 ms 没有包含 partial + merge，而 64K summary 的 L2 已包含单体 Top-K，因此不能直接把两个 summary L2 当完整同口径分解。

补回上述必要 kernel，按每个 repeat 先求已识别算子 kernel 和、再按臂取中位数、最后按既有两臂聚合，32K Graph NFA 约 3.415 ms，FlashInfer 约 2.905 ms，FI/NFA 约 0.851×；不能继续引用漏项图的约 1.260×。这是对已有 profile 的 CPU 重分类，不是新 GPU 实验，也不是完整模型 TPOT 或并行关键路径时延。已有 E2E 数值不会因这项分类修复自动失效，须分别核验。

### 1.4 全库其他关键发现与证据等级

以下点是当前源码/回执的针对性审查，不冒充逐文件全覆盖。标“源码风险”的项目尚未观测 GPU 故障；标“待 profile”的项目不承诺提速。

**异步生命周期是正确性门禁。** [scheduler 复制与双缓冲](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/scheduler/scheduler.py#L246-L276) 的 D2H `copy_` 不在 `_copy_stream` 上下文内，完成 event 却记在该流；事件不能自动证明另一流的复制已结束。双槽复用还需保证旧回执已消费。[GraphRunner drain / commit 顺序](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/engine/graph_runner.py#L144-L170) 中，EOS drain 释放请求与后续 pending_lengths 提交存在需验证的生命周期交错。现阶段为源码确认的依赖缺口/条件性竞态风险，未跑 GPU。取消同步前必须补延迟复制、槽复用和 EOS 对拍。

**padding slot 尾行有限定路径风险。** [cache metadata 更新](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/cache/paged_region_cache.py#L895-L951) 清了CPU slot前缀，却只把有效flat项复制到GPU，padding尾行可残留旧slot。例：B4 [0,1,2,3] → B3 [0,1,3,-1] 仍用bucket4，尾行可能指向仍活跃request3旧位置。protected-only eager 的独立write_decode与non-fused writer只检查slot非负，需防PAD覆盖历史KV；生产fused-write有active guard，不把风险泛化到全部路径。当前为源码可达风险，未跑GPU。测试要检查padding GPU slot恒为-1及历史KV逐槽不变，而非只看最后输出。

**短序列保护集有 host 反例。** [final union 的 SWA 公式](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/indexer/nfa_layer_batch_selector.py#L831-L865) 未把起点约束到非重叠保护区；L=64、sink=4、SWA=128、无 seed（或 seed 已消费）时，源码公式只产生 0..3，而 middle quota=0，不能补齐；L=128 漏 124..127，L=131 漏 130。已按公式 host 枚举，未跑 GPU；它不是 64K 断崖原因。应核 `max(sink_count, length-SWA)` 起点、去重与 count 契约，并检查 `paged_region_kernels_nfa.py` 同公式，而非简单扩大输出长度。

**显式 full 子空间模式存在接口/接线反例。** [selector 构造与调用](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/indexer/nfa_layer_batch_selector.py#L1701-L1776) 为 full L2 构造 fine_dims / q_fine，却把 L1 dims 传给 fine kernel，导致 K 子集与 Q 前段错配。已用原源码 AST 核调用参数，并以整数反例验证排序可反转：q0=1、q48=2；A 的 k0=3,k48=5，B 的 k0=20,k48=4；full 应为 A=13、B=28，错误接线为 A=5、B=4。默认 tail/same 不受这项影响。另 full L1 的 Q reshape 与固定 tail32 页统计宽度不符。应补模式契约或对尚未支持组合 fail-closed；不能把修 full 模式写成生产 tail/same 的性能收益。

**每页边界有真实长尾，不只是 64-bit Top-K。** 对本轮全部 64K NFA 12 臂（B1/B2/B4 × eager/Graph × a/b）的稳态原始序列重算，>80 ms 的峰值都落在原始 decode step i=18,82,146,210,274,338,402,466，且 (实际 prompt 65198+i)%64=0；FlashInfer 对应 12 臂没有 >80 ms 峰值。[B4 Graph a 原始回执](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/benchmark/efficiency/results/nfa_flashinfer_20261104/e2e/arms/ruler64k/nfa_a_ruler64k_b4_graph_attempt0.json) 的 median≈27.774 ms、mean≈33.147 ms，边界步均值≈366.928 ms。计算口径为 `tpot_ms_all[3:]` 共508步；p95/p99用 nearest-rank，1/64频率峰值可被p95漏掉。边界相关性已由原始值确认，根因仍待 trace 干预验证，不能直接断言是CPU或GPU。

**跨页 metadata 慢路是强候选，稳态快路已存在。** [cache.new_step / _prepare_common](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/cache/paged_region_cache.py#L813-L951) 在 request 变化或跨页时重建各层逻辑页 staging 行并复制全行；普通稳态已有快路，不能说每个 token 都全量重拷 block table。[慢路实作](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/cache/paged_region_cache.py#L891-L923) 已增量append后仍重写全历史页行，涉及按层按页的Python→CPU tensor标量赋值；首个64K边界约36×1020=36,720次/请求、B4约146,880次（源码操作量估计，不是CPU计时）。这条慢路与页边界峰值吻合，是待 profile 的首要解释，不是已完成因果证明。还需核 scheduler / GraphRunner 重复 new_step 与 GPU boolean compaction 是否造成额外分配或同步。

**Graph 与真实生成入口尚有可测结构性开销。** [B1 decode](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/engine/frozen_runner.py#L239-L274) 每请求新建 runner/capture_all，且逐 token `int(tokens[0])`；设备保留 token 不等于这个入口无主机同步。[capture_all](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/engine/graph_runner.py#L64-L126) 遍历多种 batch/visible width 组合。[capture PAD 与 protected-only 分支](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/indexer/nfa_layer_batch_selector.py#L1594-L1677) 使 both-skip eager 快路不会自动进入通用捕获图；Graph 中仍保留通用统计/选择管线，输出变空不等于执行工作消失。收益待测，不用 capture 时的 PAD 或旧 gate 错误特化真实请求。

**不要重复优化已经存在的机制，也不要复活过期叙事。** 当前 [paged_write_decode](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/ops/paged_region_kernels.py#L645-L660) 已对 K/V/slots 做 contiguous，旧 GLM V-stride 问题不能继续写成当前未修；真实 GLM/GPU验收仍未因此自动完成。[FlashInfer runtime](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/patches/qwen3_flashinfer_patch.py#L128-L207) 已固定 arena、跨层共用 plan/workspace、prepare阶段plan一次，并有稳定split的防护，不能建议“消除每层每步重复plan”作为本库收益。[gather实际循环](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/ops/paged_region_kernels.py#L306-L324) 已按 visible_lens 截短，大 MAX_VISIBLE 不证明全64K扫描。[当前 arena / 释放](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/sparse_attn/cache/paged_region_cache.py#L513-L534) 保留完整池，skip只改变visibility，请求结束才回收页ID且有全设备同步；旧docs关于物理KV淘汰的描述不能当当前能力。

## 2. vLLM / FlashInfer 一手源码调研与可迁移原则

访问日期：2026-11-04。固定检索当时快照：vLLM `66df311e82293c8b8177977d1a12cc9ad923198d`，FlashInfer `eeed657f157b0803ffe61df56ffe5ccf1196d2a1`。它们是设计参考，不是本库已安装版本或已测实现；本轮 FlashInfer 0.6.14 回执不能冒充这些新代码的结果。本次没有运行外部实现或获得新加速倍数。两库根许可证均为 Apache-2.0（[vLLM](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/LICENSE)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer/blob/eeed657f157b0803ffe61df56ffe5ccf1196d2a1/LICENSE)）；FlashInfer 还列有第三方 BSD/MIT 组件，vLLM 相关文件注明改编来源。实际移植前仍须核对应文件及第三方依赖的许可/NOTICE，保留文件级来源，不复制整套框架。

### 2.1 vLLM：按真实行形状分派，精确处理阈值桶

[vLLM indexer_topk 路由](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/model_executor/layers/indexer_topk.py#L258-L318) 的 auto 次序是 cooperative → persistent → per_row；FlashInfer / DeepSelect 是显式 opt-in，不是“vLLM 一律用 persistent”。cooperative 有 SM90+、rows≤64、K∈{512,1024,2048}、stride 等限制，其 C++ 入口只接 FP32；persistent 接受 FP32/BF16、返回 int32 局部列序号，不能直接替换本库 int64 复合 key。

[有效长度上界约定](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/model_executor/layers/indexer_topk.py#L354-L369) 特别区分 max_seq_len 与 padded row width，避免短行误走长行 kernel。本库应借鉴“编码安全条件与性能分派分离”：capacity 仍用于保证 S-token 编码安全；R、有效候选数、行数、K、dtype 和硬件单独决定性能路径。不能直接用有效长度替换 capacity 来强制 16-bit 编码。

若明确选择 vLLM persistent，R=8452、rows=8、K=1024 落在中等行长度路径，不是按外层 64K prompt 调到长行路径。[persistent 分派与长度约束](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/csrc/persistent_topk.cuh#L661-L712) 及 [medium recovery](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/csrc/persistent_topk.cuh#L143-L170) 体现可复用模式：先粗桶缩小候选，再精确细化；阈值桶超出临时容量时恢复完整精确选集，而非截取最先写入的若干项。BF16 大量同分时，这比均匀随机输入上的吞吐更重要。

[cooperative 实现](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/csrc/cooperative_topk.cuh#L264-L365) 利用 thread-block cluster、TMA 和 distributed shared memory 做协作直方图与精确恢复。但其 [实际长度分流](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/csrc/cooperative_topk.cuh#L507-L552) 对 sl≤16384 只由 rank0 运行 register-resident 短行路径，其余 rank 返回；R=8452 并不会自动获得 cluster 多 CTA 数据协作。更长行才按实际 sl / cluster_size 选择 shared staging 或 TMA 流式路径。可作为后续强基线，但额外平台、编译、同步及语义成本必须与本库真实 R 匹配，不能因为外层输入叫 128K 就认为更复杂实现一定适合。

### 2.2 FlashInfer：选中谁、输出顺序与地址映射是三个契约

[FlashInfer top_k API](https://github.com/flashinfer-ai/flashinfer/blob/eeed657f157b0803ffe61df56ffe5ccf1196d2a1/flashinfer/topk.py#L672-L729) 接受 FP32/FP16/BF16 logits；返回 int64 index 不代表支持 int64 复合 key 输入。`tie_break=SMALL` 比较局部列序号，不自动等价于本库按全局 token ID 的同分规则；`deterministic` 控制输出可重复顺序，不能替代边界同分选集验证。若通过重排 candidate 建立等价 tie，需要把重排、映射及逆映射成本一并计入，不能只测库调用。

[后端条件](https://github.com/flashinfer-ai/flashinfer/blob/eeed657f157b0803ffe61df56ffe5ccf1196d2a1/flashinfer/topk.py#L909-L1002) 进一步区分 SM100 cluster、native DSA Graph-safe、CUB 等路径。H20 不是 SM100；该 cluster 路径不支持的显式 tie / Graph-safe / 确定顺序要求不能静默忽略。其中 native `dsa_graph_safe` fallback 强制 FilteredTopK、VEC_SIZE=1，K≤2048 且要求 128KiB shared memory；这是 native 路径条件，不是整个 API 的 Graph-safe 全局限制，满足条件的 CUB 路径也可能用于 Graph-safe。运行前必须核实际选中的 backend、设备资源和接口条件，不能照搬 B200 的阈值或收益。

[top_k_page_table_transform](https://github.com/flashinfer-ai/flashinfer/blob/eeed657f157b0803ffe61df56ffe5ccf1196d2a1/flashinfer/topk.py#L1098-L1211) 展示“选择后直接映射物理地址”的融合机会，但本库 candidate ordinal、全局逻辑 token、physical page offset 必须分开。其 page table / 输出为 int32，非负与乘加无溢出约束并非全由运行时检查；page_size>1 时 score 行起点与 page-table 行起点的单位不同。本库还有稀疏页、sink/SWA、owner、补洞、union、升序和 counts，不能直接把 ordinal 当全局 token 送入 API。地址融合应晚于 Top-K 等价验收。

### 2.3 CUDA Graph：状态必须每次 replay 正确复位

[vLLM topk launcher](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/csrc/topk.cu) 把 row-state 清零放入 launch 前的 captured 序列，并按实际 kernel 资源检查驻留约束，防止跨 CTA 自旋 barrier 因超额驻留死锁。[FlashInfer row-state buffer](https://github.com/flashinfer-ai/flashinfer/blob/eeed657f157b0803ffe61df56ffe5ccf1196d2a1/flashinfer/topk.py#L42-L68) 区分 stream-local 状态与 capture 时的 graph-private 分配，避免不同 graph 并发共享计数器。固定指针本身不足以证明 Graph 正确。

[FlashInfer decode plan/run 约定](https://github.com/flashinfer-ai/flashinfer/blob/eeed657f157b0803ffe61df56ffe5ccf1196d2a1/flashinfer/decode.py#L2198-L2241) 要求 plan 在 capture 前，run 复用辅助结构。固定 split size 可稳定部分归约行为，但 KV 长度变化仍可能改变 CTA 数，不能据此直接保证 Graph 兼容或 batch invariance。selector 选集一致也不等于后续 attention 数值一致；必须同时记录 baseline 实际 plan/backend/split 配置。

## 3. 可直接交给执行方的 prompt

请在 two-level-attention 当前任务链内串行推进以下工作。总体顺序：先关闭适用路径的正确性和测量门禁，再按已测成本处理64K Top-K、页边界慢路，之后才做同步/capture/分配及其他候选；一次一个有界改变，不要求为每个模块制造修改。先核当前 main、适用指南、唯一任务索引、相关开发分支 / PR / advice / 回应与活跃写入者；复用已有同题工作，隔离副本，保护他人修改，不 reset、clean、stash 或混入他人文件。本文不授予新增 GPU、付费、Claude 或扩大实验矩阵权限；GPU 部分只在已有明确资源授权内执行，否则保存候选和最小命令并标“未运行”。不要修改人类 guide，也不要按本文修改 SGLang。

### A. 固定可重建源码，先修计时归类

1. 收集本轮运行的完整 source manifest：可解析 commit、diff 原字节及 SHA-256、实际加载文件 SHA、driver / profile / 汇总 / 绘图脚本、命令、软件、输入 token hash、所有方法 flags、实际 cache 容量、候选 R、K1/K2、dtype、kernel 分派和 Graph 开关。
2. 原始回执记录 `repo_head=c597e36…` 加 diff；本次该 commit 无法从远端解析。已归档 diff 的校验及 selector blob 对照不能替代完整程序身份。恢复可重建源码前保留这一缺口，不把当前 main 自动当作 tested source。32K / 64K 已归档 diff 的差别已定位为新增 attention-mask helper，Qwen3 返回值与旧硬编码相同，selector / cache 无差异；不要仅凭 diff hash 不同否定这两组比较。完整 base commit 和未归档 driver 的可重建缺口仍须补齐。
3. 从 raw kernels 重建阶段归类，按真实调用链定义 L1 / L2 / selected main attention；把必要的候选准备、展开、Top-K partial / merge、最终 union / 地址转换纳入明确范围。每个相关 kernel 必须恰好归属一类；unknown、漏项、重复归属必须使检查失败。模型公共算子可明确列为排除项并写理由，不能为凑总和把所有 GPU 工作塞进 attention。
4. 保留旧 JSON / 图 / 失败记录，新增带版本的更正结果，不覆盖原始数据。逐 repeat 先求和，再聚合；不能把各项中位数直接相加当总和中位数。分别报告 kernel self-device time、算子链 CUDA event 时间和完整 decode step / TPOT。
5. 添加无需 GPU 的回归：至少覆盖本次 32K 的四项漏计、64K 两项漏计、缺 kernel、未知相关 kernel、重复归类、NaN / 负时间、零调用与跨臂身份不一致。绘图入口须消费通过 coverage 校验的结果，不能只检查阶段和自洽。
6. 32/64/96K 的外部 GPU 占用轮询原始日志覆盖不足，自报 clean 不等于已独立核验；已有 128K driver 记录不能替它们补证。更正数值与运行环境证据缺口分别写明。
7. [ABBA driver 的 source receipt](https://github.com/Chosen-David/two-level-attention/blob/37a8e5d386fd80dbbef3d775e39990c27f790138/benchmark/efficiency/benchmark_nfa_vs_fa2.py#L275-L302) 在整轮结束才采集，普通git diff不覆盖staged，只额外收集特定untracked runtime，不能绑定每臂实际加载源码。改为每arm启动前/后内容manifest及运行中变更检测，覆盖必要tracked、staged和被加载untracked文件；不能读取或公开凭据/无关私有文件。
8. 先修可执行验收入口：当前 scripts/test_nfa_operator.sh 引用的7个tests文件在本次main tree均缺失，scripts/test_nfa_semantics.sh 的部分目标也缺失。不得照抄这些脚本声称回归通过。按当前tree恢复/替代原coverage，先collect-only确认非空，再执行真实host/native套件，逐项记录passed/failed/skipped/not-run。缺失测试不是删门槛的理由。

### B. 最小候选：只把宽 key Top-K 并行化，不同时改变评分

首先保留 >=65536 分支现有 BF16 fine logits、候选与过滤规则。把 64-bit filtered Top-K 扩展为 2048-slot partial + K-wide merge：

1. partial CTA 读取当前 logits + candidate IDs，使用原来的有效性过滤和 BF16 排序映射，构造原始 `(ordered.int64 << 32) + (S-token).int64` key。
2. partial 工作区改为 int64；invalid sentinel 不再硬编码 `-32768 << 16`，与 TOKEN_BITS、merge decoder、workspace dtype、launch specialization / cache identity 一起参数化。
3. 复用现有 K-wide merge 结构及 finish 语义。K=1024、5 tiles 的现有 merge 是逐次 K-wide 合并，不要误称一次全排序 5120 项。
4. 保留全序 key 后，任一 tile 中排名超过 K 的元素不可能进入全局前 K：该 tile 已有 K 个不小于它的元素。因此每 tile 取 K 再合并取 K 可保持全局结果；前提是过滤、同分规则、候选 owner 和后处理完全一致，不能只比较集合大小。
5. 旧 >=64K 单体路径保留为显式 reference / A/B fallback，不默认删除。先验证并行化这个单一变量；fine 直接产出 int64 key 的融合另作后续候选，因为它减少读取但增加写流量，不能预设收益。
6. 不直接提高 65536 阈值或强制 int32：`S-token` 在 S=65536、token=0 时已需要第 17 位，会侵入 score 位并错误解码；真实 64K 分配容量还大于 65536。不要缩短生成、降低 K1/K2、收紧 cache 容量或删保护区来规避问题。
7. 不直接启用未使用的 radix 路径：先核其 16-bit 常量、有效值契约与全向量归约开销。candidate ordinal 压缩可以另立候选，但必须证明唯一映射和同分顺序，不与首个修复混合。

### C. 先锁定语义，再验 Graph / eager

与修改前可信 reference 比较 indices、counts、顺序和后续 attention 输出。必须覆盖：

- capacity 65535 / 65536 / 65537 / 66240 / 132032，以及实际分配边界 prompt 64448 / 64449；有效长度与 capacity 独立变化，不仅测名义长度。
- R 小于 / 等于 / 大于 tile，R=8452、R_PAD=16384，K=1024；quota=0、<K、K，有效候选不足与全无效。
- 负 logits、相等值、BF16 舍入、±0、±inf、NaN；当前映射 +0 高于 -0，NaN / -inf 无效、+inf 有效。若核查后发现这些既有行为应变更，单独提交语义修复和质量证据。
- 当前 key 的 `S-token` 配合取最大值，实际上同分优先较小 / 较旧 token，源码注释“新 token 优先”不能代替真实行为。不要在性能补丁中顺手改变 tie 规则。
- duplicate 的 owner 选择、sink/SWA 强制区、near/far 边界、不足时 earliest-missing 补洞、最后 ascending IDs / counts，以及 near-only / far-only / both / no-skip。
- B1 / 多请求、dummy / padding、不同请求长度、slot 与页复用、replay 更换输入和 gate 后保持指针与状态正确。

先检查 `tests/test_nfa_current_query_selector.py::test_radix_filtered_topk_bit_exact_with_quota_and_ties` 的输入契约：其 BF16 logits 是否已不适用于 <64K 当前预 packed int32 路径。这个是静态发现，未运行不能标成测试失败。补齐正确输入及 >=64K 回归，不删除旧失败用例来过门。

现有 NFA eager / Graph 的 9 组 512-token 序列都分叉，而 FlashInfer 对应 9 组一致；同模式重复一致不足以证明模式等价。先用相同 teacher-forced token / 相同中间 Q/K / cache 快照定位首个分歧，分别核 selector 选集、attention 输出和 logits，并区分额外 RMSNorm / SiLU 实现及数值归约差异。生成 token 分叉本身不是质量下降证明，但在解释和既定质量验收前不能称“无损加速”。保持原验收门槛，不为通过而放宽。

### D. 只在授权后执行最小配对性能验收

顺序为：host 契约及 raw 重算 → 原生 selector 正确性 → 同输入边界微基准 → 完整模型配对；前一步失败不扩大矩阵。

1. 固定同一个有效 prompt / QKV / 候选 / gate / K1/K2，仅改变预分配容量跨过阈值，比较旧单体与新分块。这是区分“容量导致分派”与“长 prompt 本身”最小干预；再测实际 32K、64K、128K，不能用短序列伪装真实长上下文。
2. 首轮 B1、eager / Graph、512 输出；通过后再扩 B2/B4 与已有授权范围内的输入多样性。保持模型 revision、硬件、dtype、FlashInfer 版本、布局、page size、tensor-core 开关、时钟 / 功耗 / 占用、预热与 capture 规则一致。
3. GPU kernel profiling 与干净 timing 分开。记录 CTA / warps、寄存器、spill、共享内存、occupancy 和 launch；未采到的字段写未采集，不把逻辑张量字节数当实际 shared-memory 用量。
4. 在匹配配置预先规定独立配对次数，例如不少于 3 组 ABBA；保留每次及每步原始值、所有无效原因。报告均值、中位数和尾部，不挑最快值；相关 token 不能当作独立重复。
5. operator baseline 采用相同计时边界的 dense attention，Graph 对 Graph、eager 对 eager。当前 operator FlashInfer HND 与 E2E NHD/page16 路径不同，须披露；不能把一个算子比值移植成另一个完整模型比值。
6. 质量与速度必须使用同一候选及匹配配置。固定长度性能模式的 EOS 忽略须标明，另验证真实 EOS；全 KV 驻留与选择少量 KV 计算分开，不把 logical skip 写成 KV eviction 或显存节省。
7. 成功标准是消除已定位的异常成本且正确性、质量和完整模型配对通过，不是强行得到“长度越长绝对 ms 越小”或指定倍数。若修复后无收益、32K 回退或高 batch 变慢，保留负结果和 fallback，不晋升。

### E. 把调研转成有门槛的单变量对照，不换一整套框架

最小 64-bit partial + merge 正确性通过后，最多选择一个外部模式继续，而不是并行启动所有优化：

| 次序 | 候选 | 必须解决的准入条件 | 淘汰条件 |
| --- | --- | --- | --- |
| P1 | 本库 64-bit 分块 partial + K-wide merge | 原始 BF16 评分与 token tie 完全保留；workspace/Graph 生命周期正确 | 任何选集/count差异，或完整 selector / TPOT 回退 |
| P2 | 借鉴 vLLM 中行 histogram / exact recovery，或支持本设备的 cooperative 对照 | 以 R=8452 而非外层64K分派；明确FP32/BF16转换；阈值桶溢出仍精确；保留全局token tie | 只在随机数据快、ties退化、截断桶、需无法满足的资源/依赖 |
| P3 | FlashInfer Top-K 作为强基线 | backend实际可用；列序号到token的映射和边界tie等价；含所有转换/排序/映射开销 | 只能通过改tie、忽略Graph安全、遗漏适配开销才快 |
| 后续 | fine输出直接编码 / Top-K与地址转换融合 | 前述独立阶段通过，逐项引入；地址/保护集/去重契约不变 | 质量或完整成本不通过；只有孤立kernel收益 |

执行细节：
1. 保存 dispatch manifest：`allocated_capacity, logical_length, candidate_width, valid_count, rows, K, dtype, key_bits, backend, tile, warps, workspace_bytes, graph_mode`。真实稀疏R与KV长度不同，禁止混作同一个“长度”参数。
2. 不把 int64 composite key 转成 float32 企图调用通用 Top-K，float32 无法精确保留全部 score+token 排序位。外部浮点评分后端必须单独证明边界同分的精确 token 选择；抽样没碰到ties不算证明。
3. 在本库合法缓存/请求范围内测试均匀、极偏斜、全同分、跨tile同分、阈值桶超容量、全无效和部分mask候选。阈值恢复不得静默截断；normal 与 `-O` 的host契约均覆盖。不得用扩大tile替代证据，既有tile4096负结果应先查再复用。
4. 若引入共享row state、barrier或persistent workspace，增加同图连续replay更换长度/gate、两图交错/并发replay、两stream、空行、请求结束及新请求复用测试。确保清零操作被capture且每次replay执行；按实际kernel occupancy验证跨CTA同步不会死锁。未引入该机制的最小分块实现不强加无关状态框架。
5. profiler内同时保存实际kernel名/调用数/资源，以及每阶段临时分配和清零；Graph与eager分别核。选中KV的main attention当前约1ms，而64K Top-K约15ms，应按证据优先修selector，暂不为“融合感”重写已较小的main attention。
6. 外部库只作固定源码模式/强基线，不同时升级Torch、Triton、CUDA、FlashInfer和模型依赖。所需版本不兼容时用独立实验环境与清楚的结果身份，不污染当前基线；未授权安装或新增付费资源则只保留设计。

### F. 全库其余部分的具体优化任务链

1. **P0，异步正确性。** 对照 [vLLM D2H / event 生命周期](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/v1/worker/gpu_model_runner.py#L284-L324)，让复制、完成事件、源 tensor 引用和 CPU 消费有同一个明确依赖协议；双槽未消费时必须backpressure或有安全替代，不能覆盖旧回执。验证 EOS drain 与 commit/remove_request 顺序。对照 [vLLM pinned staging 复用保护](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/v1/worker/gpu_model_runner.py#L3563-L3576)，取消 `.item()/int()` 同步前先保护 H2D staging，防CPU覆写仍在DMA中的页/长度/计数。用延迟copy、连续超过双槽的提交、B4→B3→新加入、EOS、request ID复用对拍；“加一个全设备synchronize就正常”只能作诊断，不是最终低延迟设计。
2. **P0，padding、短序列与显式模式。** 先修/验证同bucket缩批GPU slot尾行清理；B4→B3、non-contiguous req IDs、all-both与non-fused写入路径，保证PAD不改任何旧KV。 用真实张量/reference补 L=0/1/3/4/64/127/128/131/132/133、seed有/无/已消费、四gate与mixed batch的保护集精确检查；非法长度要显式拒绝。full L1 / full L2 按实际支持范围修接线或拒绝，保存上面的整数反例。默认 tail/same回归必须不变；不要把 unsupported mode 的错误修复与无损加速混成一个补丁。
3. **P1，页面边界 metadata。** 在同请求冻结输入下采 i=17/18/19 及后续每64步的CPU/GPU/NVTX trace，记录 common_dirty原因、各层block-table重建、H2D字节、分配/同步/NonZero调用数。对照 [vLLM dirty write / 批量更新](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/v1/worker/gpu/block_table.py#L125-L158)，只更新新增/变化页、预分配固定device metadata，避免每次跨页重建全行；保留旧稳态快路。按 [vLLM batch缩小时残行清理](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/v1/worker/gpu/block_table.py#L186-L209) 和 [slot mapping/PAD约束](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/v1/worker/gpu/block_table.py#L229-L271) 验证非连续ID、padding、槽复用、不等长与page_size±1。单独报告边界/非边界均值和频率，不能只用p50/p95隐藏周期尖峰。
4. **P1，host同步与输入小算子。** 在生命周期正确后，将固定形状req IDs/位置/长度/采样及必要EOS状态保持设备驻留，评估移除 boolean动态压缩和重复new_step/小tensor创建。先有同步参考，再做有界run-ahead及事件驱动CPU消费；严格复核逐token、EOS、min_new_tokens、suffix输出。benchmark外层不读CPU与整个scheduler无D2H要分开标；真正吞吐是完成token数/实测wall，1000/median只是中位延迟倒数，B×1000/mean也仅是定义内event区间估计。
5. **P1，Graph冷启动与gate路由。** 借鉴 [vLLM合法graph keys与后端约束](https://github.com/vllm-project/vllm/blob/66df311e82293c8b8177977d1a12cc9ad923198d/vllm/v1/cudagraph_dispatcher.py#L164-L231)，统计图key使用率、capture/JIT时间及graph-private显存，再选择需求驱动capture或稳定arena内跨请求复用；不要说vLLM已为本库实现lazy capture。图缓存必须绑定model/cache指针所有权、capacity和影响捕获分支的policy，不只(B,width)。both-skip用运行时early-out或有失效机制的专用图；四gate、mixed请求、旧cache销毁/新cache同几何、首decode seed与同槽gate变化逐项对拍，trace需证明无用工作真正减少。
6. **P2，arena与工作区。** 分列full arena、occupied/free reusable slots、selector/workspace、graph-private、allocated/reserved/peak；先研究事件延迟回收代替请求退出全设备同步，再考虑有证明的bucket缩窄与buffer复用。最终selected上界须包含middle+sink/SWA+seed及fallback，不能按K2粗略截断。页复用/UAF、capacity pressure、高churn混长请求都要测试；不能把mask变小称物理显存下降。物理eviction/压缩、跨请求共享KV是更大语义/隔离设计，另提方案，不夹带入这轮性能补丁。
7. **P2，prefill、KV写入与attention本体。** 当前prefill是unpadded B1 one-shot，先量化dense attention、额外profiling/摘要、contiguous/scatter、freeze/capture在TTFT和全请求中的占比。先查是否能复用已有统计/预分配及成熟writer，再单独比较stride-aware路径；保留非连续GLM输入的正确性保障。若kernel/GEMM不是已测热点，不盲融合。chunked/mixed prefill是独立新能力，不伪装现有decode合同的小优化。FlashInfer已plan一次，继续核实际backend/layout/page/split/TC与同长batch限制；修复对照语义后再讨论更高效backend，不能用不匹配算子图支持E2E收益。
8. **P2，GQA/GLM与模型适配。** Qwen当前按KV group选集，未把KV实体复制成Q-head数；B1 main采用每程序1个query head，B>1有组内复用，这种并行度/读复用取舍不是直接bug。只有profile证明其成本重要，才按实际Hq/Hkv/head-dim比较grouped-head tile和CTA拆分，并保持当前评分/选集/GQA语义。GLM当前展开完整K/V，不能称latent MLA cache；用准确模型revision核Q/K/V维度、RoPE/NoPE、页布局和内存上界。现有pyproject的Transformers 4.56.2声明不能自动代表GLM当前运行环境；核官方支持和完整版本/依赖manifest，环境不匹配先隔离，不能拿Qwen回执证明GLM可重建。不要为解决环境问题同时升级现有Qwen基线或把GLM新实验矩阵夹带进本轮。
9. **逐项停止条件。** 每项先报告当前已做机制，确认缺口才实现。最多一个新候选进入GPU配对；正确性失败、源身份缺失、无匹配baseline或收益小于噪声则保持候选/回退。未授权GPU时完成host验证、精确运行命令和证据缺口，不能宣称全库已经加速。所有采用项沿唯一任务详情续接，不把本节变成另一套后台服务/调度。

### G. 当前可用入口

下面是已核对当前 CLI 的单配置 NFA / FlashInfer ABBA 入口示例，只在 GPU 使用已获授权时运行；模型、数据与输出路径由执行环境提供，不在本文假定私人路径。文件名仍叫 vs_fa2，但显式 methods 可选 nfa / flashinfer。它不自动完成“旧源码 NFA vs 新源码 NFA”配对，该配对另需保留两份源码身份并按既定顺序运行。

```bash
python benchmark/efficiency/benchmark_nfa_vs_fa2.py \
  --model-path "$MODEL_PATH" --data-root "$RULER_DATA_ROOT" \
  --task niah_multikey_1 --input-provenance native-ruler \
  --context-length 65536 --batch-size 1 --execution eager graph \
  --methods nfa flashinfer --sample-index 0 --max-new-tokens 512 \
  --warmup-steps 3 --visible-len 8192 --gate-epsilon 0.05 \
  --gpu-id "$GPU_ID" --output-dir "$RUN_DIR/arms" --output "$RUN_DIR/summary.json"
```

单臂入口为 `benchmark/efficiency/benchmark_nfa_end_to_end.py`，支持 `--measurement-mode latency/profile/traffic/correctness`；当前 ABBA driver 不接受这个参数。独立原始 kernel classifier / profiler driver 尚未定位为完整可重建归档，先补身份，不能假定单臂 profile 会生成同 schema。

CPU 绘图入口存在：`python benchmark/efficiency/results/nfa_flashinfer_20261104/plot_performance.py --mode all --output-dir "$CORRECTED_FIG_DIR"`。它当前读取 summary 而非重新归类 raw kernels，所以先修分类和更正产物，再使用这个入口，且保持旧图不被覆盖。核对正确输入目录后传入 `--kernel-root`；不能让修正版图继续静默吃旧 summary。

### H. 交付

交一份简洁 verdict：问题 / 代码位置 / 最小复现 / 变更 / tests / 原始证据 / 适用范围 / 未验证项。分别记录“已有 raw 重算”“host 实测”“GPU 实测”“模型质量”，不得混称。

先核对当前入口的 CLI 和依赖，再提供实际可执行的复现及验收命令。profile driver 缺失时先归档同源脚本；不要虚构不存在的命令或把提示词存在当作作业已启动。完整源码、脚本与结果身份不足时保持 pending。

由现有主执行方把 adopt / adapt / reject / defer 写入对应任务详情；本 advice 不另建进度台账。本次发布权限仅用于这份建议；后续代码修改、GPU 使用与分支/PR交付遵守届时用户授权和项目规范。发布任何修复前重新 fetch、整合并重测，只暂存相关文件，非 force 更新并读回 commit / 内容 / CI；未完成验证不能说已发布或已通过。
