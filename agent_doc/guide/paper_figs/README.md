# 论文出图清单：按 Fig.1 → Fig.4 → SFig 顺序执行

这是出图入口：先看“哪张图、用什么数据、跑哪个脚本、还缺什么”，详细实验条件再点旧指南。核对源码：`88c0ac625d92c51c97f8da320c9bfc44b9d0d3eb`（2026-10-14）。本文不代表实验已经完成，也不启动作业。

**图号映射：**主图编号仍为 Fig.1 观察、Fig.2 架构、Fig.3 请求性能、Fig.4 FullKV 质量；本页提出最终面板布局调整：当前稿 Fig.3(b) 的 regional/global L1 对照移入下面 SFig.2，Fig.3 集中展示请求成本。**当前 PDF/PPT 尚未按此调整，下一次稿件修改须同步面板、图注及交叉引用，不能称已一致。**旧指南中的“Fig1/2/3/4”是消融内部旧编号，不覆盖此处论文图号；其 A1–A6、O1–O3、E1–E3 稳定 ID 保留。以下 SFig 是补充出图顺序标签，尚未改写论文已有 S1–S6 补充节号。主文不机械堆满所有图。

**所有图共用：**已有结果先复用。样本 ID/文档 ID、模型与输入 token 哈希、代码/评分器版本、独立确认集、预算及事前退化容忍值，按 [E0](evaluation/evaluation.md#e0-通用冻结契约)登记；未填即待补，不用历史 AVG5 自动冒充确认集。默认 sub/sub、mavg(.25,.125,.625)、page64、K1≤128页、总 K2≤1024（含去重 sink128+SWA128），gate 默认关闭。每图交 PDF/SVG、绘图源、figure_data.tsv、原件路径/hash；命令里的大写路径变量必须绑定真实且已核原件，输出目录应为新目录。

## Fig.1 为什么需要近远区与低维检索？【O1 + O3】

- **数据：**已登记 LongBench 小型 dense Q/K trace；不是 LongBench 任务得分。现采集器默认四任务 hotpotqa、narrativeqa、passage_retrieval_en、gov_report，各取最长2条、上限32768 token、9层。这是既有诊断采样方式，具有长样本偏差，不是全量代表集；确认 sample/document/query 清单后才复用。
- **脚本：**采集入口 `exp/trace/collect_trace_lb_v.py`（GPU，现只采 prompt/prefill）；O3 分析入口 `exp/figures/o3_retrieval.py`。已有合格 trace 可离线运行：
  `python exp/figures/o3_retrieval.py --protocol "$O3_PROTOCOL" --out "$FIG1B_OUT" --plot`。
  `O3_PROTOCOL` 是待补齐/核验的真实协议文件，不是仓库已存在的默认配置；格式见 [O3工具说明](../../../exp/figures/o3_retrieval.md)。
- **测什么、怎么画：**(a) 距离 token → 每有效 token 的原始 dense attention mass density，另标保护区/实际 near-far 边界；(b) tail/high/paired-random 同32维 → retained mass（无量纲）及文档级区间。O3 输入 q/k/qpos、RoPE 回执、trace 哈希与种子，输出 selection_metrics.jsonl、summary.json、figure_data.tsv 和 o3_diagnostic.svg/png。
- **验收/缺项：**O1 的新口径距离分箱/保护去重/density 汇总与出图链仍缺；旧 `exp/figures/fig1_motivation.py` 把 near 标成 SWA，并挑 far-heavy 样本，不能直接作为本图。O3 目前是全 causal 支持上的 global TopK 离线评分诊断，没有生产 L1/L2、保护或区域配额；不能当生产等价/任务质量。真实 trace 的 document/input/producer 身份仍须核验。详见 [O1/O3](observation/observation.md)。

## Fig.2 我们的方法如何工作？【架构，不做实验】

- **数据：**无需 LongBench；以已核实现及当前17页 PPT 可编辑架构图为输入。
- **脚本/源：**沿用当前 PPT/论文源码内的可编辑图源。仓库 `exp/figures/fig7_architecture.py` 是旧图生成器，包含旧图号和硬编码输出位置，不能直接运行覆盖新图；最新版可编辑源与仓库的一键导出入口尚需绑定。
- **画什么：**同层/KV头共享表示 → 子空间/可选投影 → near/far → 两区 L1 页粗筛与 L2 token 精筛 → 原始全维 Q/K/V attention；sink/SWA 独立保护；gate 仅候选旁支。
- **验收/缺项：**输出紧凑双栏总图及可编辑源；核科学语义、最终字号、箭头和灰度。不能由箭头声称当前 L2 已只计算候选、packed INT4 或请求加速。当前图可继续用，最终外观仍待用户确认。

## Fig.3 实际请求快多少，时间花在哪里？【E2；阶段对照 A6】

- **数据：**与 Fig.4 同模型/静态配置的已冻结请求输入与输出工作量；沿用既定长度，不新增网格。先 dense-P/dense-D 对 dense-P/sparse-D；其他阶段须先过路由与语义门禁。
- **脚本：**`benchmark/LongBench/pred.py` 是自然生成入口，但不能仅凭它的预测文件生成完整请求成本图。已核 `exp/bench/e127_stage_bench.py`、`exp/trace/e126_stage_timing.py` 只作单层/selector 微基准来源，不能代替请求采集。**匹配 TTFT/TPOT/request wall 的完整采集→汇总→出图入口尚未验收，暂不提供伪完整命令。**
- **测什么、怎么画：**(a) 实际输入长度 → request wall（ms/request）；(b) TPOT（ms/generated-token）；(c) 代表长度的 prepare/update、L1、L2、gather、attention、同步/传输绝对时间。保留每次重复、实际生成 token 数、cache 状态、失败和峰值 GiB；重叠阶段用并列图/时间轴，不相加阶段中位数。
- **验收/缺项：**需要同配置真实请求原始 timings、生成长度与质量绑定；预热不能藏掉每请求建索引/capture 成本。随机 QKV、暖态 CUDA Graph 的算子快于 dense，只能标“算子诊断”，不能填成本图。详见 [E2](evaluation/evaluation.md#e2-请求时延与成本分解)。

## Fig.4 sub/sub 相对 dense FullKV 的任务质量【E1 + E3】

- **数据：**主结果按 E1 原范围：LongBench v1 14英文+2代码任务、LongBench v2 全503题、RULER 32K/64K/128K；分阶段补缺，不把全矩阵重跑。已有五任务 qasper/hotpotqa/gov_report/musique/repobench 是筛选/历史子集，不能当完整确认结果；冻结清单中的正式名称需映射到生产入口。
- **脚本：**预测用 `benchmark/LongBench/pred.py`，逐样本评分用 `benchmark/LongBench/eval.py`；sub/sub–dense 文档配对统计/诊断图用：
  `python exp/figures/paired_quality.py --protocol "$PAIRED_PROTOCOL" --out "$FIG4_OUT" --plot`。
  源码支持此接口；`PAIRED_PROTOCOL` 必须先补真实两臂预测/inline runcap、document/input 身份、scorer 哈希和事前容忍值。工具不接受 RULER 作为 LongBench 输入；RULER 原管线另行验收。
- **测什么、怎么画：**逐任务官方分数与 sub/sub−dense FullKV 的百分点差、文档簇配对95%CI、有效/失败数量；目标主面板为森林图，零线=FullKV，事前 −δ 单独标示，并附绝对分数/完整任务表。现有工具实际输出 summary.json、figure_data.tsv、figure_bindings.json 和 paired_quality.svg；其绘图仅有差值、CI 与零线，**尚无 −δ 标线、绝对分数表及 PDF 导出，这些是待适配项，不能称已经生成。**Quest、低维近邻及其他已验 baseline 按 E3 另列表，不混入本工具的两臂统计。
- **验收/缺项：**已有预测，不等于已完成配对。仍需 independently recovered document/input 与 dense 运行身份、预先冻结 δ；CI 跨0不能写无损。full/full 稀疏筛选属于 SFig.1，不是 FullKV。详见 [E1/E3](evaluation/evaluation.md)。

## SFig.1 哪一级降维起作用？【A2a】

- **数据/对照：**确认样本清单与独立划分仍待登记，冻结后比较 FF/FS/SF/SS 四表示格；历史五任务仅在身份一致时复用，不自动视为确认集。生产模型、scorer、预算、保护和量化口径固定。FF 仍是稀疏筛选，FullKV 另列。
- **脚本/原件：**复用 `exp/trace/results/e133_subspace_verdict.json`、`exp/trace/results/e134_l1sub/`、`exp/trace/results/e134_l2full/` 和 `exp/trace/results/e134_perf_reverdict.json`，逐样本重评分入口仍为 LongBench eval。现有 paired_quality 只接受 SS 对 dense，**不能直接传四臂**；四臂文档配对绘图适配尚缺。
- **测/画/验收：**2×2 面板展示质量百分点差和完整成本（分轴）；绑定实际坐标/dtype/支持。缺 FF 验收、完整请求及独立确认的格保留待测；E134 稳态摘要不能当请求成本。[详见 A2](ablation/ablation.md#a2-两级表示与频段维度)。

## SFig.2 分区配额与内容选择是否真的有用？【A3 + A5】

- **数据/对照：**复用 Fig.1 的固定 Q/K 做局部对拍，自然生成的独立确认样本/划分仍待登记并冻结；历史五任务仅在身份一致时复用，不自动视为确认集。A3：同 scorer、同候选/softmax 支持、同预算，仅区域/全局配额不同。A5：相同保护与总1024预算，本方法 middle 对预冻结朴素 middle；256保护-only仅诊断。
- **脚本/原件：**`exp/trace/analyze_e59_l1_ablation.py` 是旧 scorer replay；`exp/trace/analyze_e135_sinkswa_grid.py` 是旧2048-budget trace 网格，不能直接充当以上新控制。**纯配额及等预算朴素-middle 的生产臂/配对汇总仍待补齐与验收，无合格一键命令。**
- **测/画/验收：**逐任务质量差/CI、实际 unique near/far/protected token 数、局部 mass/输出误差及请求时间；两紧凑面板。当前 near-avg/far-minmax 不能直接混池称只改配额。缺失保留，不以保护-only弱基线替代等K控制。[A3/A5条件](ablation/ablation.md)。

## SFig.3 方法选择、频段/宽度与保护参数【A1 + A2b + A5敏感性】

- **数据/脚本：**仅复用已登记搜索与 trace：`exp/trace/analyze_e125_screen_verdict.py`、`exp/trace/analyze_e128_tradeoff.py` 及对应 results；频段诊断复用 Fig.1 的 O3 工具。旧 `exp/trace/analyze_e85d_selection_methods.py`、E135 grid 仅按原协议保留，不混入新预算。
- **测/画：**开发集合法候选 → 冻结后确认质量/成本；tail16/32/64 单轴曲线，固定32维的 tail/high/paired-random 另面板，sink/SWA仅已登记网格。每个面板独立标数据/预算，不乘成大矩阵。
- **验收/缺项：**新宽度/paired-random 生产臂、确认样本与种子未验处标待测；历史脚本有硬编码路径/旧协议，不作为立即重跑指令。正文只简洁披露128+128，未经验证不称最优。[详细条件](ablation/ablation.md)。

## SFig.4 Gate、阶段覆盖与工程边界【O2 + A4 + A6，条件性附录】

- **数据/脚本：**仅复用已存在 E137/E129、阶段四臂和工程对拍原件；`exp/trace/analyze_e129_ablation_verdict.py` 的 cclustersim 静态消融不等于 mavg prefill 标定 gate。当前 prompt-only collector 不能补 decode 漂移。
- **测/画：**far-on/off-refill/off-no-refill 的质量差与实际预算；校准→后续 decode far mass；dense/dense、dense/indexed、indexed/dense、indexed/indexed 路由及完整成本分别成表/面板。
- **验收/缺项：**G2、逐seq真实触发轨迹、完整成本与匹配质量未验保持候选；新增动态工作仍暂停，不因列在出图清单就排队。只有主张确实使用 gate 时才考虑提升到主文。[O2](observation/observation.md)、[A4/A6](ablation/ablation.md)。

## 现在按什么顺序补

1. 先补 Fig.4 已有预测的配对身份/独立确认与 Fig.3 完整请求原件，决定主结果能说到哪里。
2. 复用小 trace 完成 Fig.1；Fig.2 仅绑定最新版可编辑源，不重新采数据。
3. 只补 SFig.1/2 中能区分主贡献的缺格；SFig.3/4 优先复用。暂不具备验收条件的主图使用“待实验结果”占位，绝不画预期曲线。

旧分类文件保留为细则：[Observation](observation/observation.md) · [Evaluation](evaluation/evaluation.md) · [Ablation](ablation/ablation.md)。图序优先看本页，科学条件与失败记录沿用旧稳定 ID。

