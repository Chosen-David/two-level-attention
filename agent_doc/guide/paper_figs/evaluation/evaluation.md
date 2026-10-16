# 论文评测与出图执行对照

本文件规定主质量表、时延图和基线比较怎样产生可核验的数据。与 [消融对照](../ablation/ablation.md)、[Observation 对照](../observation/observation.md)共同使用；图的稳定 ID 用于绑定实验与原始产物，不代表每个 ID 都必须占一张主文图。

版本基线：2026-10-14，研究仓 `542234ac199cea9617c45d170c3329ab16a20687`。这是执行设计，不是新实验结果或 GPU 派单。既有结果先按 E0 核身份，完全匹配的格直接复用；不因文档更新重跑、不改写历史分数。负责人尚未确定的样本 ID、模型 revision、阈值、重复次数必须保持 `UNFROZEN`，填齐并审查后才进入对应实验。主方法实验归 Opus，外部 baseline 归 DeepSeek，结果由独立审查者验收。

## E0 通用冻结契约

### E0.1 当前研究对象及适用范围

- 论文目标配置为静态 `best_mavg(α=.25, β=.125, γ=.625)`、同层同 KV-head 的 L1/L2 tail32 完整 RoPE 对、最终原始全维 Q/K/V attention；默认不启用动态 far gate。gate 是单独候选与消融，不把 gate 功能关闭和某层 far 被关闭混为一谈。
- Qwen3-8B 的已核目标几何是 32 Q-head、8 KV-head、每组4个 Q-head、head dimension 128。不能直接把这些维度套到 Qwen3-32B、Llama 或 MLA。
- 标准长序列目标预算为 page size 64、L1 非保护候选页上限128、最终总 token 上限1024，包含 sink128 和 SWA128；保护区去重，middle 名义余量768。短序列、部分页、区域不足及实际候选不足均记录实际占用，不能只报配置上限。
- 标准分区下 near L1 名义16页、far112页；near L2 名义640、far128，再加保护区256。此算术只用于核对标准长序列，不替代每步实际 ID/预算回执。
- L1 为逐 Q-head 评分后组内均值；L2 为相同有效支持内逐 Q-head softmax，再组内概率均值。改变聚合次序或 near/far 归一化支持即改变方法。
- “full/sub”等标签只描述 L1/L2 的评分维度。`full/full` 仍做稀疏选择，必须另设 dense FullKV。32维量化缓存是浮点量化后反量化表示，不据此宣称 packed-int4 容量；选中 KV 仍为原始全维。

### E0.2 执行前必须填写的冻结记录

每个 `run_id` 的 `protocol.json` 至少包含下表。此表给出字段要求，不表示字段已填写或实验已获执行授权。

| 类别 | 必填字段与冻结规则 |
|---|---|
| 数据 | dataset revision、任务名称清单、source document/seq ID、split、输入文本与最终 input_ids 的 SHA-256、筛选/校准/确认集合及交集检查 |
| 模型 | checkpoint 与 revision、tokenizer revision、chat template、thinking 模式、BOS/EOS、截断方式、最大上下文与 RoPE/YaRN 配置 |
| 方法 | method、L1/L2 表示与实际坐标 ID/哈希、量化及累加 dtype、scale、GQA 聚合、page size、K1/K2、保护区、α/β/γ、gate/refill、dense layers、实际后端/回退 |
| 执行 | 生成代码 commit、dirty diff/相关文件哈希、CLI/env 的有效值、producer 版本、软件/驱动/GPU、batch、prefill chunk、输出上限与停止条件 |
| 验收 | 逐任务及总体质量容忍值 `δ`、值得投入的请求时间改善门槛 `η`、置信水平、重采样/重复方案、tie 与数值容差、失败处理 |
| 身份 | run_id、作者/执行负责人、产物清单、采集与评分时间、独立 verifier、可用范围及未通过项 |

冻结冠军后，确认集只用于评价，不再调维度、阈值、层集或参数。开发集上的“每任务最优”只能作事后 oracle 诊断；不能据此称全任务通用或把确认集重新变为搜索集。模型、代码或输出语义发生实质变化时创建新实验身份；优先定位受影响格，不盲目重跑整个矩阵。

### E0.3 统一产物和采集字段

新增产物登记在 `agent_doc/results/<run_id>/`；本文件不创建这些目录或文件。输入、参考、预测和用户反馈只在获授权的存储位置保留；公开仓库仅登记允许公开的数据、匿名稳定ID/哈希及有适当访问权限的来源，不因本指南要求留原件就自动上传私有内容。下列文件名是逻辑产物角色，可指向获授权的仓外原件。推荐最小集合：

1. `protocol.json`：E0.2 冻结值与引用；`inputs.jsonl`：匿名稳定 sample_id、document_id、token 长度与哈希。
2. `predictions.jsonl` 与 `scores.jsonl`：逐样本预测、参考、官方 scorer 身份和未聚合得分；保留失败、超长、OOM、缺失和重试原因。
3. `timings.jsonl`：逐请求/逐重复原始 wall、阶段计时、实际生成 token 数；不得只交均值或各阶段中位数之和。
4. `selection_metrics.jsonl`：按 sample/layer/query/KV-group 记录 K1/K2 实际唯一支持数、near/far/protected 分解、路由、坐标哈希、触发信息；完整 ID 可用单独压缩文件并登记哈希。
5. `memory.jsonl`：原始 KV、索引/摘要/QDQ、临时张量、峰值 allocated/reserved、CPU 缓存及传输字节，明确测量 API 和单位。
6. `manifest.json`：所有原件路径、字节数、SHA-256、producer/scorer/run_id 对应关系；`validation.json`：独立检查结论、范围、失败与未验内容。
7. `figure_data.tsv` 与绘图脚本：每个点能反查 run_id/sample/原字段。最小列为 `figure_id, panel_id, arm_id, reference_id, run_mode, metric_id, unit, estimate, ci_low, ci_high, n_independent_seq, aggregation, transform, source_run_id, source_field`；`run_mode` 区分 natural、teacher-forced、fixed-QK，合并版面时保留稳定 panel_id。既有大产物保留原路径，以真实路径/哈希引用，不复制成“新测量”。

采集模式必须分开：小型同身份 trace 用于Q/K、逐步IDs及语义验收；正式wall计时不混入同步导出、逐步调试日志。登记 `instrumentation_mode`、trace抽样规则和观测开销；如无低扰动计时模式则披露开销、不宣称已隔离。算法本身的prepare、同步、维护等成本仍须计入，不能以“去观测开销”为由删掉真实运行成本。

单位统一：质量用各任务官方原始分数，并单列0–100展示分数；差值是百分点，不混相对百分比。时间用 ms/request、ms/generated-token；吞吐用实际 generated tokens/s；显存用 GiB，传输用 bytes 或明确的 MiB。mass 是无量纲概率，距离是 token，维度是标量坐标数并同时给出 RoPE pair 数。

### E0.4 统计与停止规则

- 质量按相同样本配对；按源 document/seq 聚类重采样，保留任务权重，报告总体和逐任务95%区间、最差退化、失败率。head、layer、token 不作为独立任务质量样本。任务等权均值和样本加权均值分别命名。
- 运行重复次数、随机种子、bootstrap 次数和多重比较处理在看确认结果前登记。计时重复用于估计运行波动，不增加独立质量样本量。一次 warm run 不足以证明稳定提速。
- “CI 包含0”仅表示未检出差异。若要声称非劣效，应预先规定 `δ`，检查差值区间下界是否高于 `−δ`；等价主张需要预先规定双侧界。不得结果出来后扩大容忍值。
- 任一格的实际路由/预算/模型/输入身份与声明不符，先判该格不可用于目标比较；保留原件与负例。无可核 producer、无法配对或关键字段缺失时只作历史/局部证据，不进入配对 Pareto。
- 语义检查不过则停止该优化的速度主张。质量未达预登记范围或完整请求收益未达 `η`，停止推广对应收益，不必取消其他已授权的独立验证。

## E1 主质量表与模型覆盖

**优先级 P0，主文必须。问题：**冻结方法在完整任务集合上相对 FullKV 和有效原生 baseline 的质量代价是多少，是否有集中退化？

**数据与模型范围：**

- Qwen3-8B 主确认集保留原要求：LongBench v1 的14个英文任务及2个代码任务，LongBench v2 全503题，RULER 32K/64K/128K。任务名、版本及样本数由 E0 清单固定；缺任务/截断/失败不能从分母静默删除。若声称中英文通用，再按相同协议补齐相关中文任务，不由英文结果外推。
- Qwen3-32B 用于规模验证，Llama 用于跨家族验证；Llama 具体 checkpoint、head/RoPE 布局及支持长度尚待冻结。先复用已登记、身份成立的格，再分阶段补缺；不把每个消融都乘上全部模型/长度。代表子集必须事先登记并诚实标为子集，不能称全量模型验证。
- DeepSeek-V2-Lite-Chat/MLA 保留为后续扩展，本轮暂不做，不从 GQA 结果宣称支持 MLA。
- 128K 必须有实际 context/RoPE/YaRN 与 producer 回执；脚本当前默认值或 scorer 声明不能倒推旧作业有效配置。

**实验臂与固定量：**dense FullKV、冻结本方法、E3 中已通过对应模型/阶段验收的 baseline。所有方法使用同一 checkpoint、输入 token、生成约束、官方评分和确认样本。选择预算、保护/dense layers 差异公开，不强行将不同原生算法改成同一个方法。

**采集与统计：**复用 E0 原件；每任务 `n_planned/n_valid/n_failed`、score、paired delta、95% CI、实际上下文/生成长度、真实 token 占用。主结果优先自然生成；teacher-forced 同轨迹结果另作诊断，不与自然生成混表。

**表与面板：**主文质量表按任务组列总体与关键退化，附录给完整逐任务表；不要只画均值柱状图。模型规模/家族用独立列或小面板，显式写 supported、pending、unsupported，不用零分填缺失。

**baseline 接入状态（DeepSeek 轨，2026-10-16）：**
- **Quest（已验收，已入 E1 图 panel b）**：Qwen3-8B LBv1 41.35 / LBv2 30.22 / RULER 128K 19.33（另 32K 42.85、64K 26.25）；逐任务 13/16 任务见 `exp/results_longbench_vendor/Qwen3-8B/quest_vendor_1024_16/pred/result.json`（缺 gov_report/qmsum/repobench，标记子集口径）。
- **ClusterKV（vendor 已接入，派单中 pending）**：RULER 32K/64K/128K + LongBench 16 待补；预算对齐 token_budget=1024/sink=16（总 attend≈1025，构成=仅 sink 强制区，披露与 TLI sink+swa 差异）。
- SparQ/MInference/FASA 数据已登记（RULER 32K 60.45/49.94/24.00；LBv1 48.66/47.71/33.31），按 evaluation.md E3「额外 baseline 不堆数量」策略作附录候选，不强制入主图。

**支持或否定：**可支持声明范围内的质量—成本取舍；不能证明全局最优或跨模型普适。若有任务超过 `δ`，保留并说明适用边界，不能只展示总体平均。

**产物绑定：**`figure_id=E1`，质量表的每格绑定 `run_id + scorer_hash + sample_set_hash`；复用格同样记录原始 run_id。

## E2 请求时延与成本分解

**优先级 P0，主文必须。问题：**在相同工作量下是否真正缩短请求，收益来自哪段，长输入或短输出会不会抵消？

**最小阶段设计：**先比较 dense-P/dense-D 与 dense-P/sparse-D。prefill 稀疏候选只在 A6 的入口/语义门禁后加入，不能把索引初始化称为 prefill attention 加速。保留已有32/64/128K协议；如既有请求协议还登记32/48/64/96K，则复用其冻结长度，先选其中一个已登记长度排错，再补原计划，不另立全长度×全部消融矩阵。

**计时边界：**

- 请求 wall 从声明的模型请求入口到最后一个输出完成并同步；明确是否含 tokenization，排除模型加载和一次性编译时单列其成本。不能在 warmup 中预建被测请求索引后漏记。
- TTFT 含 prompt forward、首 token 前的建索引/标定/同步和首输出处理；首答案 logits 来源于 prompt evaluation，不能计成稀疏 decode 收益。
- TPOT 给出所用定义与分母，例如首 token 后 decode wall 除以其余实际生成 token 数；同时记录每步分布和实际生成长度。
- 分项记录 prefill attention、index prepare/update、gate calibration、L1、L2、TopK、候选构造/gather、最终 attention、CPU/GPU transfer、必要同步/分配、sampling。异步重叠以真实 wall 为准，不能把各项单独时间直接相加；分解图采用并列绝对耗时或时间轴，存在重叠时不画暗示可加总的stacked bar。
- CUDA event 用于设备阶段，host wall 用于完整边界；成对交替测量、相同硬件/负载/缓存初态、固定预热与重复。报告请求配对比值及区间，不把“阶段 median 之和”叫请求时延。
- 固定输入/输出 token 轨迹用于等工作量诊断；自然生成另报真实服务成本及实际输出长度。相同 token 轨迹也不保证不同方法 hidden states 相同，selector 对拍必须另固定 Q/K。

**采集与单位：**`request_wall_ms, ttft_ms, decode_wall_ms, generated_tokens, tpot_ms, throughput_tok_s`；各阶段原始 ms；原 KV/索引/峰值 GiB，H2D/D2H bytes。存硬件、路由、B、输入长度、输出长度、cache 状态和重复序号。

**主图安排：**(a) x=实际输入长度，y=请求 wall 或 TPOT，独立面板不混单位；(b) 一个代表长度的绝对成本分解，明确重叠项不可相加；(c) 若已有完整配对点，x=请求时间，y=质量，点标实际预算。prefill 正负收益均保留，不能只画加速的长度。

**当前证据限制：**E127/E134 历史单层合成稳态不能直接当36层 TPOT 或请求 E2E——该缺口已由 **E152v2（2026-10-16，v2 连续 wall 口径 ABBA×3，`e152v2_e2e_latency_verdict.json`）闭合**：完整 36 层 Qwen3-8B 请求 decode TPOT，dense vs TLI（dynamic）vs TLI+StaticCache（E134-pre 系统层消 72-cat/步），L=32/64/128K 全档。如实结论：TPOT tli/dense = 3.4×/2.5×/1.8×，kernel 级加速（E127 5.47× GPU / E133c -12~-15%）不 translate 到请求级，净缺口在 HF Python/launch 链（72-cat + 索引器 per-step Python），归因链见判决 JSON attribution 段；E134-pre 静态缓存臂 -5.9ms（87.9→82.0 @32K）。E2 图 (c) 面板即此数据；不做「超过 dense」主张，负结果按 spec 保留。v1（分段和 wall）数据作废不混用。旧段保留备考：`1945403` 图规划已把3.82×改为旧 sparse-mask 到 gather/prefetch 的实现比，而非相对 dense；当前 main 的 `eager_prefill` 是全 QK/PV 后 mask 的质量参考；分支中未来优化必须以它自己的运行 SHA/路由验收，不能借到 main。

**停止条件：**完整请求未改善则只报告局部 selector 收益；标定、传输、索引维护抵消收益时保留净负结果。内核快但请求慢，不提升为 serving 加速。

**产物绑定：**`figure_id=E2`；每点绑定同一冻结质量—成本配置，允许独立 timing run，但必须共享可核方法/输入/模型身份。

## E3 基线选择与比较轨道

**优先级 P0 先辨识与验收，P1 再绘图。问题：**我们是否比最接近的有效替代方案更值得用，而不只是比一个慢 dense 实现快？

| 方法 | 必要性与阶段 | 进入主表前的条件 |
|---|---|---|
| FullKV 与高效 dense backend | 必选锚点，dense-P/dense-D | FullKV 是算法语义，FA/SDPA 是实现；写清实际后端，质量与速度均使用正确且适用的基线 |
| Quest | 必选最近邻：minmax 页筛选，原生 dense-P/sparse-D | 固定官方版及 Qwen3/GQA 适配；页预算、前两层/dense 路由、Q/K norm、原始产物与运行身份通过，不把质量 mask 路径当速度内核。**已通过验收并接入 E1（DeepSeek 轨 2026-10-13）：Qwen3-8B LBv1 41.35 / LBv2 30.22 / RULER 42.85(32K)/26.25(64K)/19.33(128K)**，见 `exp/results_ruler/main_table_quest_vs_tli_draft.md` |
| SparQ 或 FASA-C | 至少一个通过验收的低维评分近邻；支持子空间贡献比较 | 固定频维/预算策略、GQA语义、补偿项、原生phase。SparQ新增 vendor 入口不等于完整验收；按已登记 k=1024及r随实际ctx映射核有效r/配置身份。FASA-C不能替换成固定低频tail后仍称原法 |
| FFD | 保留原要求，条件主基线：阈值筛选与量化内核系统 | 固定论文/代码、Qwen3适配与原生量化策略；δ对应可变实际预算。核自然logit与代码log2阈值约定；首末参考块不是强制sink/SWA保护。未验前只列待接入 |
| HISA | 保留原要求，待唯一身份核验 | 先填准确题名、作者、固定论文版、官方仓库及阶段；不能根据缩写猜成HiP或其他方法 |
| ClusterKV | 条件系统基线/相关工作：聚类检索与KV放置 | 保留建簇、重聚类、CPU offload和传输成本；官方accuracy/efficiency路径不能拼接成同一质量—成本点；缺依赖/适配时标pending。**vendor 已接入（2026-10-16 #16 completed，官方 accuracy 选择器 md5 冻结 + 纯 torch search_indices 等价实现），预算换算公式已核（总 attend=token_budget+最近区），正式派单中** |

官方来源入口：[Quest](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357)、[SparQ](https://github.com/graphcore-research/llm-inference-research)、[FASA](https://github.com/wangyifei0047/FASA-ICLR2026/tree/96c00c46db12ccee56104175470aa29579d66c58)、[FFD](https://github.com/qluoluo/faster-flash-decoding/tree/ca09458ab1536328e5f46502b27891412733cd6d)、[ClusterKV](https://github.com/sjtu-zhao-lab/ClusterKV/tree/c7380819588771e8c5dadacdbd33942d7c53adea)。这是来源入口或已核源码快照，不是本库适配验收；SparQ 的实际 vendor 上游路径/版本必须另随运行冻结。HISA 身份未核，暂不填猜测链接。

额外 baseline 不堆数量。Double Sparsity 可在低维比较仍有明确缺口时作为附录候选；AsyncTLS 的分组共享两级设计应在 related work 正面对比，但截至已核范围未找到可归属作者的实现，不能编造 native runtime。新增方法先证明必要性，不自动扩跑所有任务。

两条轨道必须分开：

1. **原生算法/系统轨**：保留作者算法、量化、补偿、保护层与offload，统一模型、输入输出、硬件和计时边界；报告差异及真实预算分布。
2. **统一后端机制轨**：相同 Q/K 回放与最终 attention backend，只隔离 selector。GQA合并、page rounding、保护策略等改动逐项声明；标签写“统一后端适配”，不冒称作者原生复现。

**预算与统计：**优先以相同质量容忍范围比较完整成本，另报同总预算诊断。记录每Q-head、每KV-group及组内union的实际 unique token 数，不能用同一个 K 名称假装访存相等。可变预算方法在开发集冻结阈值，用确认集报告分布；不得逐测试样本调到刚好相同K。

**图与产物：**来源/支持矩阵为附录表；合格方法才进入E1/E2。预算Pareto仅在已登记K档的质量—成本身份齐全后绘制；不新增海选。缺原件的论文数字可标“作者报告”用于背景，不能与本库实测连成一条Pareto曲线。

**支持或否定/停止：**faithful-port、输入/预算、评分或计时不一致先阻塞对应比较；不强迫baseline分数符合论文趋势。真实反例保留；不能因baseline更强而撤掉它。

## 图量与执行顺序

主文预算按论证职责分配：一个主质量表E1；一个2–3面板时延图E2；一个合并Observation图O1–O3；一个核心消融图A2/A3/A5。gate若有独立质量—成本证据且构成主张，再加A4面板或独立图；否则放附录/限制。A1海选、完整维宽曲线、模型逐任务表、来源矩阵和原始统计放附录。不要把所有稳定ID强制做成13张主图，也不要为了凑版面删除负结果。

顺序为：已有产物身份核对 → 方法/预算/phase语义验收 → Qwen3-8B最小受控比较 → 完整任务确认与匹配请求成本 → 已要求的规模/家族验证 → 作图。各阶段只补真实缺口；不因“计划齐全”标实验完成。

## 一手论文参考及借鉴边界

- [Quest，ICML 2024 正式PMLR版](https://proceedings.mlr.press/v235/tang24l.html)：§4.2.3/Fig.7用于预算—质量消融对照；§4.3.1–2/Figs.8–10区分kernel与模型计时。§4.3.2/Fig.10测单token decode且不含sampling，不能搬作整请求；§4.3.3/Fig.11的代理速度比较有额外开销遗漏，不能当所有方法统一实测。
- [SparQ，ICML 2024 正式PMLR版](https://proceedings.mlr.press/v235/ribar24a.html)：§5.4/Fig.7作组件消融，§4的GQA讨论及§6/Table4/Fig.8、Appendix F用于核对布局、性能和设置；该正式版性能实验是孤立attention微基准，不是完整LLM请求，双K布局另有KV存储代价。另查的 [arXiv v6](https://arxiv.org/abs/2312.04985v6) §6.2测单个生成token的整个Transformer；此节号属于该版本，不混作正式版。原生补偿与缓存布局按版本保留，不能只抄TopK。
- [ClusterKV，DAC 2025，arXiv v2](https://arxiv.org/abs/2412.03213v2)：§V-C、Fig.12–13提供包含prefill/建簇的整请求比较启发；质量与效率使用不同模型的点不能拼成我们的配对结果。
- [FASA v3](https://arxiv.org/abs/2602.03152v3)：校准选择dominant RoPE components；Appendix B.1的decode隔离协议不证明固定低频或prefill加速。
- [FFD v1](https://arxiv.org/abs/2609.00097v1)：阈值、量化和系统实现联合评价；固定代码参数及实际选择集合须独立核验。

当前源码锚点：[L2全S评分及far短路](https://github.com/Chosen-David/two-level-attention/blob/542234ac199cea9617c45d170c3329ab16a20687/sparse_attn/indexer/tli_indexer.py#L1296-L1481)、[GQA概率聚合](https://github.com/Chosen-David/two-level-attention/blob/542234ac199cea9617c45d170c3329ab16a20687/sparse_attn/indexer/tli_indexer.py#L2041-L2046)、[prefill质量路径](https://github.com/Chosen-David/two-level-attention/blob/542234ac199cea9617c45d170c3329ab16a20687/sparse_attn/ops/eager_prefill.py#L73-L172)。源码存在和作者PASS声明均不替代本文件要求的独立结果验收。
