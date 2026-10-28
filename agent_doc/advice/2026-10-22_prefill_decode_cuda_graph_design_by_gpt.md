> Historical design from PR #4. Preserved for review, not current implementation status.
> See agent_doc/results/pr-unification-20261027/validation.md for current disposition.

# Prefill / decode 同时支持的 CUDA Graph 设计

日期：2026-10-22。源码快照：main `77df85772b426db007e4c1fa7b3836db27727409`。

## 最新用户目标与本 PR 范围

用户最初要求 dense prefill、稀疏 decode，并提出删除 prefill Graph；随后补充“调研一下，最好可以都支持”。最终按兼容两阶段能力处理：**保留现有 prefill 实验路径，优先完成 decode CUDA Graph + 大 batch；不删除生产代码或改写历史结果。**此前本地删除候选已撤回，未推送。

本 PR 是调研与实现交接，**不是 CUDA Graph 已实现的功能 PR**。没有变更运行默认值，没有运行 GPU，也没有提速结论。TG 会话未提交的 2a/2b 实现不在本轮可访问的 main 内，需其提供提交再集成。

## 一手来源核查

以下为 2026-10-22 阅读的在线文档；不能据此假定项目当前安装版本具有全部能力。

| 来源 | 核实的能力 | 对本库的意义 |
|---|---|---|
| [vLLM CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/) | `FULL_DECODE_ONLY` 与 `FULL_AND_PIECEWISE` 是独立模式；后者允许 uniform decode 全图、其余 batch 分段捕获；后端能力限制实际模式 | 阶段分开分发，保持 eager 回退；不能直接把 vLLM 配置移植成 HF harness 的实现 |
| [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs) | replay 依赖固定 shape/layout/地址；不安全部分可留 eager，仅捕获安全片段；捕获区不能有 CPU 同步 | 动态长度元数据与固定容量分离；支持 piecewise，不强迫全部 indexer 立即进入图 |
| [FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html) | prefill、decode wrapper 均提供 `use_cuda_graph` 与元数据缓冲；启用后 wrapper 的 batch size 固定 | 变长 prefill 并不等于不能 Graph；需要容量/批次边界、缓冲区生命周期与外部规划协议。此 API 不自动实现 LoDA 选择器 |

结论：不应把“prefill 通常计算较重”写成“prefill 不支持 Graph”；是否值得开启取决于捕获边界、复用次数、padding、冷启动和显存成本。

## 已提交代码的真实状态

- `sparse_attn/patches/qwen3_attn_patch.py`：默认 native dense prefill；`TLI_SPARSE_PREFILL=1` 只对 B=1、无 attention mask 条件接入稀疏 prefill，其他情况仍会 dense。该 fallback 不能被当成“稀疏 prefill 大 batch 支持”。
- `sparse_attn/ops/eager_prefill.py`：只有 candidate runtime、分组选择与 `TLI_PREFILL_BATCH_SELECT=1` 等前提下，才会调用批量 selector。
- `sparse_attn/ops/batch_select.py`：`TLI_PREFILL_SELECT_GRAPH=1` 捕获选择体，不是整模型。首次按实际 L/几何/schema 克隆输入、warmup、capture、首次 replay；后续原地 copy 输入并回放，可跨兼容层复用。
- selector Graph key 绑定 L、分组、池边界、预算、backend、dtype/layout、device/stream。容量最多 8 项，超限全清；这是按精确形状缓存，不是已经实现了长度分桶或全模型 piecewise。
- selector 快路径拒绝显式 near 配额、gamma=None、sink4 等当前配置；D′ skip-far 层也回退。仅打开环境变量不代表 LoDA 当前主配置用了 Graph。
- `qwen3_attn_patch._tl_decode_batched` 仍逐行 `indexer.clear → prepare_mask → eager_decoding_attn`，batch 接口不等于 batch kernel 或 decode Graph。
- `benchmark/RULER/pred_ruler_batch.py` 当前循环仍构造变长 mask/输入、处理 host list 和读取 token；尚无整步 capture/replay。
- StaticCache 现有路径仍将长度转 Python int、切实际长度；indexer 有 GPU 值转 host 的操作。仅换 StaticCache 不能直接录整步。

本轮检查：已下载 18 个 Python 文件语法解析通过；抽取真实 `_gate_ok` 做 9 个 host 配置探针，确认旧支持配置接受、sink4/gamma-off/显式配额/async/oracle 拒绝。不是张量计算或 GPU 对拍。该环境没有 Torch/CUDA，未复跑仓库 Torch 测试。历史 E139/E140 GPU 结果只作为其记录版本与配置的证据，不继承为当前固定配额方法的验收。

## 配置：算法与执行方式分开

下表是待实现的接口契约，不是现有 CLI 参数。

| 维度 | 候选取值 | 约束 |
|---|---|---|
| `prefill_attention` | `dense / sparse` | sparse 不能因 batch>1/padding 静默变 dense；必须记录或 strict 拒绝 |
| `decode_attention` | `dense / sparse` | sparse 明确使用当前 LoDA 固定配额与 D′ 语义 |
| `prefill_graph` | `off / selector / piecewise / full` | selector 只适用 sparse；piecewise/full 按后端逐项验收 |
| `decode_graph` | `off / piecewise / full` | full 要求整步图安全；不兼容时回退需可见，验收可要求禁止 fallback |

预期优先配置：`prefill=dense, decode=sparse, prefill_graph=off, decode_graph=full`。同时保留 prefill-only sparse、decode-only sparse、两端 sparse 和全 dense 的消融入口。执行方式不应偷偷改变算法身份。

## 两阶段的捕获边界

### Decode：先完成的交付

1. 固定 `[B_bucket,1]` 输入、position、有效长度、alive、KV slot/页表，以及 KV 容量。按已验收的 B 和容量桶准备图；不能每生成一个 token 就按精确 S 重录。
2. Q/K 投影、KV 写入、LoDA 选择、主 attention、FFN 至 logits 尽量进入 full graph。采样/EOS/调度可先留图外，明确计时边界；不把“至 logits”写成已捕获整个生成循环。
3. LoDA 的 S_real、near/far 边界、当前页、候选有效性与有效预算转成设备张量。候选输出分配固定上界，以 mask/有效个数裁剪；不能在捕获区做 `.item()/tolist()` 或以真实长度决定 Python slice、分支与 top-k 容量。
4. 支持 sink4/SWA128、显式 nb/nl（及实验 nt 轴）、边界页和 D′；不能复用旧比例公式冒充当前方法。区分固定容量与真实范围，保护区不能越界或重复计预算。
5. batched selector/attention 用真实 batch 维的 kernel。首版单设备单 stream，先 B=1，再 B=2/4/8 与异长请求；多设备另验，不能隐式继承。
6. warmup/capture 不得永久推进请求状态；KV 写入用指定位置幂等写，host 状态仅推进一次。不同请求重用图时重填完整有效元数据，结束/取消槽位不可污染存活行。
7. 若 selector 仍有不安全部分，显式交付 piecewise：固定安全片段图化，中间 selector eager；输出记录真实模式与未捕获耗时，不能称 full。

### Prefill：保留并逐步扩展

- dense prefill：先保留原生 FlashAttention/SDPA。若形状复用足够，评估在 attention 两侧捕获投影/FFN 等片段；不支持的 attention 留 eager。支持 Graph 与使用 sparse attention 是两回事。
- sparse prefill：保留现有 selector Graph 作为独立实验；补齐当前 LoDA 参数、保护区和 D′ 的对拍前，不扩大快路径入口守卫。
- 全图 prefill：仅对后端支持且收益明确的形状/容量桶开启；从 exact-shape replay 起步。分桶 padding 必须保持 position、cu_seqlens、causal、near/far 边界和有效 KV 完全按真实请求计算。
- 多请求 varlen：展平 Q + cu_seqlens 只提供布局，不自动让 kernel graph-safe。Graph key 至少区分请求槽位容量、总 query token 容量、KV/候选容量、执行阶段与方法配置；有效元数据留在缓冲区中更新。
- 不以跨层 shape 相同推导所有层可共图：层参数、gate、预算、backend 与捕获闭包必须兼容。保留 schema 与闭包生命周期防护。

## 共同缓存与可观测性

Graph key/失效合同需覆盖设备、stream/并发约定、B/token/KV/候选容量、dtype/layout、模型缓存所有者/地址、执行阶段和实际方法策略。数据值应原地更新；地址/容量/编译实现变更应失效，不能靠 copy_ 的广播或 dtype 转换掩盖不匹配。

按显存字节预算管理缓存，比较有限 LRU 与当前“8 项超限全清”；不要先引入所有维度笛卡尔积。分别计 `requested_mode / actual_mode / capture / replay / fallback_reason / cache_bytes`。缺省允许可见回退，验收 strict 模式拒绝回退。显存池共享与并发捕获必须单独验证。

## 接入顺序与验收

1. **TG-D1**：取得 TG WIP 提交，核对 fixed-width/valid-length 与 batch kernel。保留现有 prefill；阶段配置写进 receipt。
2. **TG-D2**：decode B=1 full/piecewise 首录及连续 replay 对拍；变长、跨页/桶、请求重用、D′、保护区与候选不足覆盖。
3. **TG-D3**：decode B>1 对比逐请求 B=1；异长/结束/取消/padding 哨兵检查，逐层选集、输出/logits/最终质量分别判定。Graph 同 shape 可要求选集逐位相同；跨 batch GEMM 可能换数值路径，容差与质量门预登记，不能假定数学上保证逐位相同。
4. **TG-P1**：prefill selector 支持当前方法；逐组参考 vs eager batch vs graph 对拍，越界/边界页/预算单独核对。
5. **TG-P2**：dense/sparse prefill 的 piecewise 能力与分桶对照；不支持组合 strict 拒绝或显式 eager，不伪报支持。
6. **TG-E2E**：同机独占，B∈{1,2,4,8} 与代表长度，分别报告冷启动/捕获、TTFT、TPOT、吞吐、峰值显存、cache 命中、padding 和 fallback。整请求计时包含输入更新、采样和同步。Graph off/on 必须同算法、同质量口径。

本轮仅完成调研和源码定位；以上任务均待实现/设备验收，不能把此设计 PR 当成功能交付或论文数据。
