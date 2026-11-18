# NFA gate 阈值精度—性能消融：论文所需数据

- 日期：2026-11-18（北京时间）
- 作者：GPT，论文协作会话
- 来源：用户明确要求“给远端 main 写 advice 表明你需要的数据”。
- 状态：数据需求建议，尚未运行或验收；由主执行者结合任务链登记 adopt/adapt/defer。
- 范围：补齐 gate 阈值消融。本文不要求改论文、替换现有 NFA 主性能曲线或重跑 baseline。

## 1. 要回答的问题

在其他 NFA 配置固定时，提高 near/far attention-mass 阈值，能节省多少实际 decode 时间，会改变多少任务精度？默认 tau=0.10 是否提供合理折中？

强制 near/far/both skip 的已有图只说明执行路径收益；固定 tau=0.1 的跨长度性能只说明该工作点表现。都不能代替多阈值精度—性能配对结果。

## 2. 已有证据与缺口

核对 main 2bb5208fb7023861df8259da2b80c6d246b995f1：
- [历史消融任务](../task/task_details/TL-NFA-ABLATION-20261030.md)记载 C 组 tau=0.01/0.025/0.05/0.10 精度完成。
- [现有 inventory](../results/nfa_dac_experiments_20261115/inventory.csv)却将 old_C_tau* 标为 reference_only_not_consumable：最终 raw、manifest、scores 缺失。先尝试找回最终证据；不能只用日志“完成”作为分数来源，也不要重新使用旧失败桥接产物。
- [新 tau=0.1 算子结果](../results/nfa_bs1_decode_only_gate01_20261117/operator/summary.json)及 clean 数据有自然 skip，但缺少同配置多阈值质量/性能组。
- 旧 gate 消融阶段明确暂停过性能测试；请求墙钟元数据不能冒充正式 decode benchmark。

历史数据只有在完整输入、配置、方法语义和评分证据可核对时复用。协议不匹配可单独归档，不与新组拼接。若找回旧质量数据，也需明确它是否能与当前性能组配对。

## 3. 最小必需矩阵

| 项目 | 要求 |
|---|---|
| 模型 | Qwen3-8B，BF16 |
| 数据集 | LongBench v1 HotpotQA、GovReport，各完整 200 条；沿用现有 ablation 子集，固定样本 ID |
| 变量 | gate-off、tau_N=tau_F=0.025、0.05、0.10 |
| 规模 | 4 配置 × 2 任务 × 200 请求 = 1600 次质量生成；已有严格匹配结果可复用 |
| 参考 | 同输入与生成协议的 FullKV 质量，优先复用现有结果 |
| 硬件/执行 | H20-3e，BS1；性能使用 CUDA Graph，所有阈值同执行模式 |

tau=0.01 可在历史数据可恢复时额外展示；tau=0.20 可作为更激进阈值的可选扩展，均不阻塞上面的最小矩阵。不新增模型或 baseline。先报告完整结果，不根据质量是否下降挑选阈值。

固定论文配置：
- dense prefill；末尾 256 个可见 query 估计区域 mass；按输入、按层决定 near/far skip，并在 decode 固定。
- 两个阈值同步变化；使用自然决策，不强制 skip，不套用跨输入静态层列表。
- L1/L2 使用相同 32 维低频子空间；page=64；K1=128（near32/far96）；K2=1024。
- near length=4096，sink=4，SWA=128；L2 合并两区候选后联合 Top-K，near_tokens=off。
- 保留当前论文实现的预算定义，明确 K2 与 protected positions 的并集/去重及实际 token 数，不悄悄改变计数规则。
- gate-off 必须是真正关闭跳过；不是 FullKV，仍运行同一个 NFA 稀疏选择器。
- 固定模型、tokenizer、prompt、输入截断、RoPE、生成参数与代码版本。若当前入口不能表达上述配置，先记录具体差异，勿静默换默认值。

## 4. 精度采集

真实自回归生成；官方 LongBench scorer，HotpotQA F1、GovReport ROUGE-L。保持任务规定的 max_new_tokens/EOS/greedy 协议，各阈值使用完全相同的输入 token 序列。

交付每个 task × threshold 的 n、score、相对 gate-off 的配对差值及 95% paired-bootstrap CI；FullKV 只作为额外参照。两任务分别报分，可附两任务等权平均，不能混入其他任务。CI 跨零不代表已证明等效或“都在随机噪声内”。

优先质量也使用与性能一致的 Graph 路径；若必须 eager 生成，需用代表请求验证 gate 决策与真实选择语义一致，分别标记执行模式，不能未经检查混称同一实现。

## 5. 配对性能采集

优先对同一批 400 个原始输入逐个测量四个阈值，保留 request-level 配对记录。性能测量独立于正式质量生成，避免把 GovReport 长输出与 HotpotQA 短输出的请求墙钟直接比较。

- 每次从该输入的 dense prefill 状态开始，重新标定其自然 gate。
- 固定执行 32 个真实自回归 decode step；前 3 步预热排除，后 29 步记录逐步 CUDA-event 时间并取中位数。
- EOS 只记录、不提前结束这段固定步数性能测量；该输出不用于正式质量评分。
- 完整 decode step 包含输入/position/cache 元数据更新、selector、必要量化/缓存更新、整模型前向与 argmax；不计 prefill、初始化、Graph capture。
- 所有阈值使用相同计时边界，不以独立 profiled kernel 时间替代 clean TPOT，不在每步额外插入 host synchronization。
- 同时可记录整段 host-wall TPOT；分别存储，不把 host 平均与 Event 中位数之差解释为纯 CPU 开销。
- 性能卡独占；至少 3 次独立重复，以轮换阈值顺序减少漂移。先对每请求每次运行求 29 步中位数，再对重复取中位数，最后按任务汇总；保留所有原始值。
- 如完整400输入性能成本不可接受，可预先固定按实际 token 长度分层的每任务20条子集，所有阈值共用，公开样本ID；质量仍为200条。不得按 skip 多或加速高选样本。

额外算子分解不是本轮硬性缺口；若补，单独无混淆的 profiling 进程测 L1、L2、selected attention，与 clean TPOT 分开。

## 6. 必须记录 gate 是否真正生效

每输入每层保存 near/far mass、阈值、keep/near-only/far-only/both 四态；空区域与 nonempty-but-skipped 分开计数。记录实际 L1/L2 执行或跳过证据、Graph replay/fallback、最终检索 token 数。

任务级报告 near-only/far-only/both 比例及分母。若某阈值不触发 skip 或没有加速，如实保留；不得用理论 skip 比例当延迟收益。

## 7. 交付格式与论文用途

建议结果目录：agent_doc/results/qwen3_gate_threshold_tradeoff_20261118/。

至少提供：
1. protocol.json：冻结配置、精确命令、模型/数据/tokenizer版本、代码 commit 与 dirty diff/hash、运行环境与硬件。
2. quality_summary.csv：task, gate_mode, tau_near, tau_far, n, score, delta_vs_gate_off, ci95_low, ci95_high。
3. performance_summary.csv：task, gate_mode, threshold, n_inputs, repeats, event_tpot_ms, speedup_vs_gate_off, near_only_fraction, far_only_fraction, both_fraction。
4. request-level predictions/scores、逐步时间数组、逐层 mass/decisions、样本与实际输入 token 哈希、原始输出路径。
5. independent_verification.json：独立脚本复算样本完整性、官方分数、配对差值、计时汇总、gate 决策、配置身份及 SHA；明确哪些已验证、哪些仅生产者报告。
6. README：一张可直接给论文作者的四配置精度/TPOT/skip 汇总表，附真实失败或缺格说明。

论文预期为紧凑单栏两子图：横轴阈值，(a) 质量（或相对 gate-off 分差），(b) decode speedup；HotpotQA/GovReport 分别画线，同色同标记。若性能只取预定子集，在图注说明。结果用于解释默认0.1的质量—效率折中，不预设结论。

本 advice 由用户授权写入 main；本次只新增需求文档。主执行者将采纳结果关联现有任务，不把本文视为实验已完成或已启动的回执。
