# Advice：eager / CUDA Graph 性能验收与 GLM 接入审查执行 prompt

- 作者类别：GPT / AI 代码审查建议。
- 日期：2026-11-04。
- 来源版本：main `e28f5aaf124845de0e48311f57e98d485acda6b6`；提交前已回读该 head。
- 相关 PR：#39，回读 head `c77eba0ccda4f696b49a22c592b45a3b43539542`，open / draft / 未合并。
- 适用范围：two-level-attention 的真实模型推理、测量入口、相关算子与回归；不适用于据此宣称全部模型、长度、batch 已验收。
- 本文状态：供执行方逐项判定，尚未代表采纳、实现或 GPU 验收。用户此次明确授权的是将 advice 推送 main；本文不扩大后续生产代码修改、推 main 或合并 PR 的权限。
- 证据性质：远端源码与已有产物审查，加主机端布局机制核验；本次没有重跑 GPU 模型实验。性能数字是仓库记录或其重算，不能作为本次独立 GPU 复现。

## 当前判断与证据边界

当前 Graph 有局部优化证据，但不足以给出论文全矩阵“合格”；eager 尚未证明追平匹配的 FA2；GLM 必须先关闭已定位的 decode 写入布局问题，再谈其性能和质量验收。不要预设必须达到 2x 等目标，不能为达到目标修改计时范围、数值门槛或选择语义。

| 核查项 | 已有证据 | 可以及不可以得出的结论 |
| --- | --- | --- |
| iter110，Qwen3-8B，名义 32K / 实际 32612，B1，生成 128 | 两次 Graph steady median 的均值 9.91168 ms；相对归档 FA2 12.595712 ms 约 1.2708x | 是限定配置下的历史参照；ABBA 配对对象是 iter108 与 iter110，不是新跑 NFA 与 FA2。约 0.22% 的迭代差异不能直接宣称稳定提升 |
| 同配置 eager | 两次 steady median 均值 25.29299 ms；归档 FA2 eager 22.96912 ms，约 0.908x | 暗示 eager 仍有瓶颈，但需同环境新配对确认 |
| Graph 原始延迟序列 | iter110 b1/b2 排除 3 个 warmup 后的算术均值约 10.4683 / 10.4613 ms，最大值约 44.80 / 44.48 ms | “多次运行中位数的均值”不等于“逐 token 平均延迟”；必须检查长尾发生位置。CUDA event 间隔之和不是完整请求 wall time |
| iter114，生成 512 | iter110 基线约 9.90647 ms，MLP 并行约 10.27585 ms；四臂 token 相同 | 该并行候选回退成立；该轮 FA2 因缺 flash_attn 未运行，不能用 128-token FA2 数字补作 512-token 加速比 |
| skip | 上述产物运行回执 far / near / both skip 均为 0 | 可作该样本的无实际 skip 观测，不代表动态 gate 对所有输入无效，也不能代替 skip 收益实验 |
| GLM | split 后 V 是非连续 view，当前 standalone decode writer 按 packed 偏移读；PR #39 已提供修复候选 | 有明确布局错误机制；主机验证不能替代真实 CUDA、模型质量和 Graph 验收 |
| 显存 | latency 路径未在稳态起点重置 peak，末端才补记 steady_decode_start | 现有 steady peak incremental 字段不能直接解释为稳态 decode 峰值增量；须重新测量 |

主要证据入口（均固定到本次审查 SHA）：

- [iter110 verdict](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/exp/trace/results/nfa_hotpath_opt_20261102/iter110_merge16/iter110_merge16_verdict.json)
- [iter110 Graph 原始 b1](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/exp/trace/results/nfa_hotpath_opt_20261102/iter110_merge16/ruler32k_iter110_abba_graph_iter110_b1.json)
- [iter110 Graph 原始 b2](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/exp/trace/results/nfa_hotpath_opt_20261102/iter110_merge16/ruler32k_iter110_abba_graph_iter110_b2.json)
- [归档 FA2 eager](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/exp/trace/results/nfa_hotpath_opt_20261102/ruler32k_iter81_128_abba_solo_fa2_a_eager.json)
- [iter114 verdict](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/exp/trace/results/nfa_hotpath_opt_20261102/iter114_mlp_parallel/iter114_mlp_parallel_512_verdict.json)
- [迭代记录](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/exp/trace/results/nfa_hotpath_opt_20261102/ITERATIONS.md)
- [性能入口](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/benchmark/efficiency/benchmark_nfa_end_to_end.py)
- [GLM patch](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/sparse_attn/patches/glm4_moe_lite_attn_patch.py)
- [paged writer](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/sparse_attn/ops/paged_region_kernels.py)
- [frozen runner](https://github.com/Chosen-David/two-level-attention/blob/e28f5aaf124845de0e48311f57e98d485acda6b6/sparse_attn/engine/frozen_runner.py)
- [已有修复 PR #39](https://github.com/Chosen-David/two-level-attention/pull/39)

## 可直接交给执行方的 prompt

请基于执行时最新 main，按“正确性 → 测量可比性 → 性能定位 → 最小修改 → 配对验收”的顺序完成工作。目标是回答当前 eager 与 Graph 数据能支持哪些结论、性能浪费在哪里、GLM 接入是否忠实且正确；不要只给优化建议列表，也不要把程序能运行当成验收。

### 1. 建立当前基线与避免重复开发

先读适用 AGENTS.md / CLAUDE.md、CURRENT_NARRATIVE.md、TASK.md、任务详情、相关 advice 与最新实验回执，确认实际生产 selector、子空间配置、gate 和 cache ABI。核对已有结论的 SHA 和条件，不把归档方案当当前实现。

回读 main、PR #10、#39 及后续相关 PR 的 head、review、讨论与新产物；检查全部远端开发分支，含无 PR 分支。记录 head、提交时间、相对 main 的独有提交、祖先关系，阅读实际代码和任务记录；优先检查 opt/l2-fine-topk、opt/ako-pool-score、e175-skip-ring-kv 和新的独有开发。分支名或提交时间不证明任务仍在运行。祖先分支列为历史快照；其他候选标记已包含、可复用、重叠、冲突、正确性问题或待 GPU 验证。复用未变事实，避免重复搬运。

每轮仍须分析 main 的实际调用路径、张量形状、访存、launch、临时张量和生命周期。先定位 main 中问题，再判断已有 PR 是否覆盖。生产修复遵循当前用户授权与仓库规范，保留修前 SHA；同题优先更新已有 PR，不覆盖别人分支，不直接推 main 或自行合并，除非另有明确授权。单 agent 串行推进，已采纳工作并入既有任务体系，不在 advice 维护第二套进度表。

### 2. P0：GLM 正确性先于 GLM 性能

沿真实 GLM attention forward → KV projection / split → write_decode → paged writer → selector → gather attention 追踪布局。明确每一步 shape、stride、storage offset、dtype、head 数和 head dim。

当前可检验机制是：GLM 将联合投影拆成 K 的非 RoPE 部分与 V，V 的 head stride 包含被拆走的 K 段；decode writer 的 packed 偏移没有表达这一 stride。例如每头 [K0,K1,K2,K3,V0,V1] 时，正确三头 V 是 [[4,5],[10,11],[16,17]]，packed 偏移可能读成 [[4,5],[6,7],[8,9]]。必须验证实际源码仍存在此机制；不要仅靠 shape 相同判安全。

先评估 PR #39 最新实现和测试，复用其最小修复，不重复提交同题补丁。contiguous 规范化是正确性候选，会给非连续输入增加复制；随后可比较 stride-aware writer 的成本，不能为了减少 copy 删除正确性保障。原生验证至少执行该 PR 提供的：

```bash
python tests/ops/test_paged_write_layout_host.py
python -O tests/ops/test_paged_write_layout_host.py
python tests/ops/test_paged_write_layout_cuda.py
compute-sanitizer --tool memcheck python tests/ops/test_paged_write_layout_cuda.py
```

以上命令需在包含该 PR 测试的 checkout 运行；执行前核对路径与接口。保留修前失败与修后通过的同输入证据。复制 / cache 写入要求零容差；原生 CUDA 测试被 skip 不等于通过。覆盖 FP16 / BF16、B1 / 多 batch、非连续 K/V、无效 slot、Graph replay 更换输入和 slot。

再做模型级忠实性审查：

- 固定准确模型 ID / revision、Transformers 版本与真实 config。审查 q LoRA、KV LoRA、NoPE / RoPE 拆分、interleave、位置、缩放、mask、head 广播及输出投影，不凭模型名称推断维度。
- 先关闭稀疏选择与 gate，以全选有效位置对拍原生 dense 模型的 attention 输出、logits 和生成；再逐项启用 selector、near skip、far skip。全选对拍的通过不等于稀疏质量通过。
- 当前 GLM 路径展开完整 K/V，不能据此宣称实现了原生 latent MLA 缓存或其显存优势。展开路线可以有正确的数学语义，但须报告实际物理成本；latent cache 改造另立候选，不能混进最小布局修复。
- 旧 check_glm_frozen_regions smoke 使用历史接口且只覆盖短 eager 生成，不能替代当前 ABI / Graph / 质量验收。当前 benchmark_nfa_end_to_end 入口只接受 qwen3，不能用 Qwen 回执声称 GLM 已测。
- 扩展维护中的入口时，使用 GLM 实际支持且语义匹配的原生 baseline。若 FA2 不支持相应 MLA 路径，明确替代 backend 和差异，不强行标成 FA2。

### 3. P0：统一测量边界与统计定义

修复测量后重新建立基线，不把口径变化算作优化收益。

当前 NFA event 覆盖 runner.step，而 FA2 在 event start 前更新 input / position / cache position。两臂应同时报告完整 decode step 和模型 / kernel 诊断时间，完整 step 包含相同的输入准备、元数据更新、forward 与采样范围。

当前 NFA Graph 注册额外 RMSNorm / SiLU decode patch，而 eager 不启用，FA2 也没有同样补丁。因此 eager→Graph 差异包含 launch 与额外算子实现差异。分别报告：

1. 各自最佳生产栈的系统比较，列清全部启用优化；
2. 在可兼容范围内使用相同公共优化的消融，隔离 attention 方法收益；
3. NFA 相同算子配置的 eager / Graph 对照。

不得把整段差异全部归因于 CUDA Graph。当前 clean latency 路径已预分配 event、在设备保留 token history，并在末端同步；不要把不存在的逐 token .cpu() 同步当成优化成果。

指标必须明确：

- 生成 G 个 token 通常含 prefill 产出的首 token 与 G-1 次 decode；报告实际次数。
- TTFT、prefill、gate / 摘要构建、cache 转换、Graph capture / warmup、稳态 decode、完整请求 wall time 分列。首请求与可复用状态下的请求分开，capture 计费规则写清。
- 保存每步原始延迟，报告均值、中位数、p95 / p99、最大值及明确的 warmup 排除规则；分析长尾位置，不挑掉慢点。
- 512-token 请求不能用中位数 ×512 代替实测。完整请求 wall time 包括实际服务路径开销；CUDA event 区间和只作为其定义范围内的数据。
- 主要对比采用同样 512-token 工作量；128-token 可作 smoke 与历史连续性检查，不能跨长度拼加速比。
- 固定长度 latency 模式可忽略 EOS，但必须标注；另做真实 EOS / 请求终止正确性测试。
- 当前 batch 复制同一 prompt 仅是同长度吞吐测试，不等价于独立输入、不等长请求或逐请求 gate 验收。

显存另用专门的 profiling run：warmup 后同步、记录稳态 resident 并 reset_peak_memory_stats，再运行 decode 区间；另记 prefill / capture / 全请求峰值。报告 allocated、reserved、峰值定义、cache arena 与临时张量，避免把 reset / snapshot 同步插进主要 latency 路径。free page 回池复用、原缓存引用释放、allocator reserved 与驱动可见释放分别验证；不要把账面 KV 比例称作实测显存下降。

### 4. P1：按证据定位性能浪费

先在无实际 near / far skip 的匹配配置定位，再测自然 gate 分布和强制四种状态。强制状态用于机制与正确性，不代替真实输入质量结果。

优先检查以下候选，每项均需当前源码定位、机制、成熟替代方案、维护成本与可证伪实验：

- **eager 主机调度和 launch**：分解输入准备、runner / metadata、projection、summary update、selector、KV 写入、attention 与模型其余计算。将 profiler 诊断与无 profiler 的配对时延分开。确认当前混合 cache index / attention 的实际 batch 并行方式，不沿用历史“仍逐行”的说法。
- **64K 容量触发的宽 key TopK 路径**：检查 _nfa_filtered_topk / _fused_fine_scores_paged 的 capacity guard。当前选择依据可来自分配容量而非有效长度；在 PAGE=64 且额外预留 8 页的配置下，生成上限 128 时 prompt 64832→64833、生成上限 512 时 64448→64449 可跨 65536 容量门槛。先用当前分配公式重算，再测边界两侧和实际 64K，记录容量、R、kernel 分派与资源。比较 int64 partial / merge 或既有实现；不能简单删 16-bit guard，也不能直接搬用非负概率排序的零 padding 规则处理有符号 logits。
- **prefill 额外统计与复制**：检查 gate / core profiling 是否仍在原生 attention 外重算，区分论文融合设计与现有实现。评估复用 online-softmax 统计或其它受支持方式的真实增量成本；检查 paged_write_prefill 的 contiguous + scatter，比较 stride-aware scatter。保留 native attention 对原 K/V 的生命周期，不能提前释放仍使用的 storage。
- **固定形状 skip 与缓存驻留**：核对 both-skip 全层 bypass、混合请求的固定 shape 是否仍做无效读写、摘要更新和投影是否可跳过，及 ring wrap / linear 转换后旧引用是否真正释放。更改 Graph 状态前明确 pointer、capacity、slot 生命周期。
- **模型公共计算占比**：iter110 profiler 的 GEMM / reduce、selector 与 gather-attend 分解仅是该配置诊断，不是干净 TPOT。attention 微核提升须兑现到全层与完整模型，避免局部百分比放大成端到端收益。

不重复已否定方向，除非提出新机制并给出判别实验：iter109 增大 partial tile 已回退；iter111 KV projection fusion 未过既定 BF16 一致性门槛；iter113 attention+MLP 并行、iter114 MLP-only 并行均使完整模型变慢。不能用孤立 stream 微基准推翻端到端负结果，不得放宽阈值使候选晋升。

selector 语义变化独立管理：grouped-Q 先聚合再打分与逐 head softmax 后聚合一般不等价。不同聚合、低维投影、严格 compact selector 的质量必须单独对拍和消融，不能称作无损 kernel 优化，也不能用历史 gate 质量替代。

### 5. 最小实验矩阵、运行顺序和验收

串行推进：GLM 布局回归 → dense 模型忠实性 → 测量入口修正 → Qwen 与 GLM 各自无 skip 基线 → 单候选配对 → gate / 生命周期回归 → 扩展论文矩阵。已知失败应先修复或隔离，不浪费资源跑无效主表。

| 维度 | 必须覆盖或说明 |
| --- | --- |
| 模型 | Qwen3-8B；准确 revision 的 GLM-4.7-Flash，各自匹配 baseline |
| 长度 | 32K、64K、模型与配置允许的 128K；记录实际 token 数及 YaRN / RoPE 配置。超能力、OOM、backend 不支持明确列缺格 |
| batch | B1、B2、B4；更大 batch 依资源扩展。相同 prompt 与独立请求分开 |
| 执行 | eager、CUDA Graph；capture 成本与稳态分别记录 |
| 生成 | 128 smoke / 历史对照；512 主配对；完整请求计时实测 |
| gate | 无 skip、near-only、far-only、both；自然 gate 记录逐请求逐层实际冻结状态 |
| 生命周期 | 不等长、EOS、不同请求 gate、新请求 reset / recapture、slot 复用、多次 ring wrap、混合 ring / linear |
| 精度 | 全选与 native 对拍；生产稀疏配置质量；修改前后同输入配对；聚合变化单独消融 |

每个性能候选至少执行预先规定数量的独立配对（建议至少 3 组 ABBA），相同模型 / 输入 / token 数 / dtype / backend / cache 容量与实际 flags，记录 GPU 型号、软件版本、时钟 / 功耗条件及外部占用。预编译、预热策略一致，保存所有有效样本和无效原因；不能只取最快一次或把相关 token 当独立重复。小于噪声的不判收益，不跨机器、版本、长度拼接胜利。

正确性阈值运行前固定，复用既有门槛；纯复制逐位一致，算法数值差异按已有模型标准判定。kernel 正确但模型错误，或 Graph 仅首次输入正确，均不通过。memcheck、changed-input replay、指针稳定性与长生命周期测试分别有回执。

最终逐项给出：
- correctness：通过 / 失败 / 未执行；
- performance：稳定收益 / 无显著差异 / 回退 / 缺少可比数据；
- quality：通过 / 失败 / 覆盖不足；
- 适用范围、缺格与阻塞。

无 GPU 时完成可执行 host 验证并提供原生复现命令，明确 GPU 未验收，不输出预计加速比当实测。没有可靠收益不制造生产修改；正确性修复即便增加必要 copy 也应明确其成本，不能包装成加速。

### 6. 交付与记录

已采纳建议关联现有 TASK.md 与任务详情；PR #39 优先沿其 TL-PAGED-WRITE-LAYOUT-20261104 记录续接。新问题先检查是否已有任务再建记录，不复制另一套任务台账。

在仓库当前结果目录规范下保存基线 / 候选 manifest、原始 timing、质量结果、硬件软件回执和简洁 verdict；论文正式 experiment 产物遵循当前 result/experiment 约定，若当前规范已变化则按最新规范。保留历史 exp/trace 来源，不改写旧数字。给出真实可运行命令及其输出路径，不交付只含伪代码的验收流程。

提交前回读 main / 相关 PR 并整合必要增量；推送后读回 commit 与 PR 状态，交远端团队合入。只报告实质修复、重要发现与阻塞，不把状态检查做成单独提交，不修改人类 guide 或无关论文。仅在新证据足够时更新当前叙事，区分作者回执与独立复现。

## 执行方取舍记录

以下是建议处理入口，不是第二套任务清单。当前均待执行方评估；不得把空白处理视为已采纳。

| 建议范围 | 处理（adopt / adapt / reject / defer） | 依据、适用范围与调整 | 关联任务 ID / 验收证据 |
| --- | --- | --- | --- |
| GLM decode 布局与 native 对拍 | 待评估 | 优先复用 PR #39，GPU 验收未完成 | 已有候选记录 TL-PAGED-WRITE-LAYOUT-20261104；执行方确认最新状态 |
| 计时、统计、显存与 baseline 公平性 | 待评估 | 修正口径后重新建立可比基线 | 执行方关联既有任务 |
| eager / 64K TopK / prefill / skip 生命周期候选 | 待评估 | 先 profile，再按收益和维护成本选择 | 执行方关联既有任务 |
| GLM / Qwen 扩展矩阵与稀疏质量 | 待评估 | 缺格不以历史或跨配置结果填充 | 执行方关联既有任务 |
