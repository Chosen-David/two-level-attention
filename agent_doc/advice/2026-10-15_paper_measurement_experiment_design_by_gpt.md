# 有限增量实验设计候选：按论文主张闭合证据

日期：2026-10-15。状态：**已完成独立设计审查；实验尚未执行或验收。本文件是实验设计候选，不是实验结果、派单或运行授权。**

## 0. 本次优化的实质

最重要的增量不是新增方法、长度或参数，而是把已有图规约转换为可拒收的执行门禁：

1. 先恢复已有 FullKV/sub/sub 配对身份和正式确认集合，不能拿五任务筛选均值直接完成 Fig.4。
2. 将 Fig.3 的单 kernel、全 attention pipeline、模型 TPOT、完整请求分开验收；不把 E139 64K 局部 1.64×升级为请求收益。
3. 本法表示只补 FF/FS/SF/SS 中的真实缺格；near/far 只做能隔离配额的控制，不顺手扩成 scorer×参数×模型矩阵。
4. 现有 candidate-L2、candidate-topk、compact-block 是已有实现。先通过 Graph 身份/输出正确性门禁，再判断是否值得专门优化 near/far 执行。
5. 论文测量细节的逐字段对照另见附后的《论文测量口径逐字段对照》；本文件把采用规则落到本项目的协议、字段、产物和停止条件。任何文献 NR 都不转写为本项目默认值。

**范围锁：**Opus 负责本法，DeepSeek 负责外部 baseline；此文仅提交方案审阅，不向二者直接派单。已有完成格直接复用；只列缺格候选，不批准任何新 GPU 作业。新增动态 α/β/γ 与 D′研究保持暂停；现有 prefill 标定/decode 复用 gate 仅按已测范围描述。

## 1. 来源冻结、冲突解决与 advice 去重

本轮设计输入快照的远端 main 头核为 `ed15763fcd7c8c42bbbeca89f7d3a246b5c08dca`。已实际读取最新 `agent_doc/guide/paper_figs/README.md` 及 evaluation、ablation、observation 三细则、advice README 与两份10-11建议、E137建议；公开来源路径和固定版本见下文。最新入口优先决定论文图号；细则的稳定 A/O/E ID 保留。定稿前复核main已前进至`b4406658aa0e028879cd7ba34c75747a5d67c6e4`；四份paper_figs规约的Git blob SHA逐份相同，方案所依赖规约未变，历史数据仍锁上述原快照。

- Fig.1=O1/O3观察；Fig.2=架构；Fig.3=E2请求成本；Fig.4=E1/E3主质量。regional/global 从旧 Fig.3(b) 移至 SFig.2。guide中“当前稿/PPT尚未同步”是其历史状态提示，不作为当前成稿事实；最终应以发布稿/PPT实际面板核对。
- main 普通 L2 全S评分后mask；实验 `e1edf9c316fc4190dd0c29bc5d47477136f8bdab` 已有候选 L2/candidate-topk；`93e4a364ea3be4897a076ef62feddb60145eccef` 已有活跃块遍历。最新状态须绑定实际执行 SHA，不把 main 的限制泛化到所有分支。
- 较旧 A6/E2 的“E139原件未取到”已被本地 Step2 harness+verdict 部分补齐：能画作者局部摘要；逐重复 raw 和请求计时仍没有，不能因旧文字忽略新证据，也不能过度升级。

| 已有建议/规约 | 已覆盖内容 | 本轮处理与真正增量 |
|---|---|---|
| `2026-10-11_benchmark_paper_scope_by_gpt.md` | phase、faithful port、同机请求、native/统一后端两轨 | adopt，不重开竞品接入；新增逐格身份合并表与NR阻塞传播 |
| `2026-10-11_phase_matched_experiment_design_by_gpt.md` | 四阶段、前缀语义、请求成本、旧实验身份 | adopt；只先验 dense/dense 与 dense/sparse，其他阶段条件解锁，不扩跑 |
| E134 已归档子空间判决 + S-T022 | 三表示格与紧凑实现已存在，身份/更广质量开放 | adapt；先找FF及旧运行回执，不再建议“实现紧凑32”或整矩阵重跑 |
| `2026-10-13_1040_E137_gate_composition_scope_by_gpt.md` | 覆盖bug已修、pending/freeze/clear、标定不保证未来安全 | adopt；不再报静态覆盖未修；现有结果按版本标注，不恢复新动态研究 |
| Graph两份既有审查 + 既有kernel审查 | 几何/schema键、打印不失败、fallback冒充Graph | adopt；复用已交付CPU修复与独审，不重复运行；真实路径待另获授权验收，非新性能创新 |
| paper_figs E0/A3/A5 | 最终unique K、配额纯控制、等K朴素middle | adapt；显式区分跨方法可不同K1与因果消融固定K1/P/L2/scorer，防止公平性口径串用 |

没有引入第二套远端任务清单；下文是这份候选的验收顺序。

## 2. 三种比较必须分别登记

### 2.1 算法公平性：same-K

K指每个查询注意力行最终去重后实际参加attention的token数，包含保护token；组共享时说明各行复用同一集合，跨头union/访存另报。预算分开记为上限`K_cap`和每步/每head或KV-group实际支持`k_real(t,h)`；二者均包含去重保护token。仅相同上限叫cap-matched；逐比较单元实际支持数相等才叫realized-support-matched（严格same-K）。page rounding、候选不足或短序列clamp使实际数不同的格不能仍标严格same-K，也不能事后补token硬凑。标准长序列本法目标总上限1024，sink128+SWA128，非保护名义768；短序列/重叠/候选不足时记实际数，不能用名义1024冒充相等。

跨方法 K1/page粗筛预算是内部参数，可以不同；记录K1、page size、候选数、选择/TopK/搬运/维护/最终attention全部成本。不能用同K1强行抹掉方法设计，也不能以最终同K推断compute、traffic或resident KV相同。报告每Q-head、每KV-group、group union及整层实际读取/共享粒度。方法原生保护/dense层不同，在原生轨保留且披露；无法匹配最终K的原生方法只报告实际分布，不逐测试样本调阈值硬凑K。

### 2.2 真实系统：同质量成本 frontier

只在已有登记预算档内，先在开发集冻结候选/阈值，确认集评价。只让质量身份与成本身份完整且相同配置的点进入frontier；满足逐任务及总体预登记质量容忍δ后比较请求成本和显存。不能用一个实现的quality点配另一个kernel的速度；不能插值虚构未测预算，不能只保留优势长度。没有δ或独立确认集时只画描述散点，不称“同质量更快”。

### 2.3 因果机制：受控消融

A2只改变L1/L2表示；A3只改变指定层级的区域/全局配额；A5只改变middle内容选择策略。A3若区域配额改变导致可用候选不足，必须记录实际支持差异；此时是包含支持量变化的整体配额政策效应，不是纯内容排序收益。A3区域效应固定统一page scorer、K1、page size、L2 scorer/softmax支持、表示、保护、精度和共享规则。near-avg/far-minmax不能混池后声称“仅关闭区域”。统一scorer的控制是机制实验，不冒充生产mavg语义。混合scorer生产结果与控制分轨。

## 3. 全图共同冻结记录：未知不得填默认

下列已知目标不是旧run生效回执。静态本法意图为 mavg(.25,.125,.625)、SS tail32、page64、K1≤128、K≤1024含保护，gate关闭；Qwen3-8B目标32Q/8KV/D128。所有旧臂须从原运行逐项恢复，不能回填当前默认。

| 字段组 | 逐run必需值 | 证据/拒收条件 |
|---|---|---|
| 模型 | checkpoint revision、tokenizer revision、template/thinking/BOS/EOS、RoPE/YaRN | 权重/配置与生产回执；仅模型名不够，128K不能由现脚本反推旧factor |
| 输入与划分 | dataset revision、task/split、sample/document IDs、input文本与token hash、校准/开发/确认交集 | 按document隔离；旧五任务标开发/历史，未证独立不能改名确认 |
| 工作量 | 实际ctx、输出上限/EOS规则、固定轨迹或自然生成、B、chunk | 同工作量微基准与自然服务成本分开；token轨迹相同仍不等Q/K相同 |
| 选择语义 | L1/L2实际坐标、pair IDs、scale、QDQ及累加dtype、scorer、GQA次序 | L1逐Q score再组均值；L2逐Q softmax再组概率均值；生产异构near/far不偷换 |
| 预算/阶段 | K1/page/最终unique K、protected/near/far IDs，first/dense layers逐层列表，P/D真实route | flags与实际route同存；统计fallback/不足/重复；first-dense层不是模型缺省猜值 |
| kernel/环境 | 实际backend及实现SHA、tile/精度、GPU/driver/CUDA/Torch等版本、resident/offload | 相同GPU/负载下匹配；不拿质量mask实现借原生kernel速度 |
| 测量 | timer/sync边界、warmup/repeats、cache初态、compile/capture/prepare/update归属 | 每重复raw；host wall与device event分清；不合计阶段中位数 |
| 统计 | 主指标、δ逐任务/总体、η、CI、bootstrap/seed、多重比较、失败规则 | 确认前冻结；CI跨0不证明无损；反调δ/阈值/层集禁止 |
| 溯源 | protocol、run_id、producer/dirty diff、scorer hash、产物hash、独验结论 | 原件/身份缺失只做历史描述；软件验证通过不自动科学验收 |

δ、η、模型revision、确认ID、生成长度、bootstrap数、timing重复数当前具体值均未在本任务得到冻结回执，保持 UNFROZEN。阻塞按产物逐项传播：历史摘要重绘只需原件/hash与描述标签，不等δ/η；CPU软件审查需代码与测试契约，不等确认集；O3诊断需trace/RoPE/分析协议而不需要任务质量δ；配对描述需运行/样本/scorer身份，但没有事前δ仍可描述差值；正式非劣效需要事前δ与独立确认，净收益推广另需η；请求时延采集本身可在测量协议冻结后独立进行，只是未绑定质量前不称同质量优势。文献数值只作测量设计参考，不替用户决定这些值。

## 4. Fig.1：观察 O1 + O3

**问题。**保护区以外的近远距离需求是否不同；固定32维完整RoPE对是否保留检索信息。观察不承担配额/质量因果结论。

**可复用。**既有dense Q/K trace。collector `exp/trace/collect_trace_lb_v.py` 目前为prompt/prefill，默认四任务各最长2条、最多32768、9层；此采样有长样本偏差，只能按实际清单描述，不能外推LongBench总体或decode。先找原件和document/input/producer/RoPE回执，不新采GPU。

**最小缺口与入口。**
- O1仍缺按当前规约实现的离线分箱/去重/density汇总；旧`fig1_motivation.py`误把near当SWA且挑far-heavy，不能直接跑来填图。候选工作仅是离线汇总适配，不改selector。
- O3现有`python exp/figures/o3_retrieval.py --protocol "$O3_PROTOCOL" --out "$NEW_OUT" --plot`可用，但需真实冻结协议；本轮读取其代码blob `fe2bce897810f21aa65a868723a835f219e02be0`与说明。它是CPU float32、all-causal/global TopK、无保护/无L1/无区域配额诊断，不是生产BF16/Triton等价。禁止将其输出贴成A3生产分区结果。
- O3协议字段：`model_revision,dataset_revision,producer_commit,split,rope_receipt_path/sha256,query_phase=prefill,support=all_causal_global_topk,quantization=minmax4_k_float32,budget,bootstrap_repeats/seed,pair_seeds,traces`；trace含真实hash/sample/document/input/layer/query_indices。脚本至少2个random seeds/2文档仅是软件下限，不代表统计充分。原collector meta缺身份不能伪造补齐。


**O3进入当前Fig.1的具体缺格：**现有measure()仅输出全causal总retained_mass；当前论文图注要求protected mass separately，仍需在不改变候选支持、不重新采trace的前提下，按冻结保护ID分别计算互不重叠的`selected∩protected`与`selected\protected`的原始full-reference mass；若另报全保护集reference mass则单列，不能与selected总mass重复相加。适配未验前只交全causal局部诊断，不能称Fig.1最终面板完成。

从仓库根执行的CPU分析入口如下，变量必须先绑定审核协议及仓库外不存在的新目录。NPZ需要NumPy，PT另需PyTorch，绘图需要Matplotlib；缺依赖不自动安装或运行。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tla-mpl XDG_CACHE_HOME=/tmp/tla-cache python exp/figures/o3_retrieval.py --protocol "$O3_PROTOCOL" --out "$O3_NEW_OUT" --plot
python exp/figures/figure_data_contract.py --bindings "$O3_NEW_OUT/figure_bindings.json" --check "$O3_NEW_OUT/figure_data.tsv"
```

**冻结与指标。**O1固定距离箱/任务权重/query位置和当前真实near边界；保护sink/SWA先去重，非保护near/far不重叠。每箱mass、有效token数、mass/token density及总非保护mass并列，零距离处理明示；原始full-D参考，不用低维softmax代替。O3固定32标量/16完整pair、scale、支持、量化/tie，tail/high/paired-random各seed均保留；raw记录选集/Jaccard/recall/mass和最差Q-head。

**画法与验收。**(a)距离density+边界范围，附区域mass/token份额；(b)三种32维retained full-reference mass，文档簇CI。O3软件现为文档等权、每seed独立CI，尚非task-balanced/paired-delta非劣效；不把seed间变化混进其文档CI。O1质量守恒、支持合法；O3pair/RoPE核验和固定支持通过后可作局部动机。若tail不胜random/high或far弱，保留并收缩主张。优先P1，依赖身份清单；不阻塞优先的Fig.4证据回收。

## 5. Fig.2：架构语义，不新增实验

复用当前17页PPT/论文可编辑源，先绑定源文件hash及导出入口。旧`fig7_architecture.py`硬编码路径和旧编号不可直接覆盖。图中明确：同层/KV-head共享表示→near/far→两路L1/L2→原始全维Q/K/V attention；sink/SWA独立保护；D′仅far候选旁路。near avg、far minmax标不同reduction，不画near全维attention/far低维attention。

关键区分：共享坐标不等物理cache复用；32维浮点QDQ不等packed INT4；算法箭头不等main已候选内省算。现有candidate分支可在实现脚注注明SHA与未验边界。完成标准为源码语义、可编辑源、PDF/SVG、双栏字号/灰度检查，不是新增质量或速度证据。P1，无模型运行依赖。

## 6. Fig.3：四层账单与完整请求 E2/A6

### 6.1 四层一一对应，不做跨层倍率嫁接

计时层命名T0–T3，与算法L1页筛/L2 token筛分开，避免重名。

| 层级 | 实际计时边界与字段 | 可支持的结论/不能替代 |
|---|---|---|
| T0 单kernel | 同输入布局/精度/输出，device event或明确同步host；`kernel_ms,kernel_name,launch_count,bytes` | 仅算子实现变化，不替代selector/模型 |
| T1 全attention pipeline | prepare/update分摊或单列 + L1/page TopK + L2/token TopK + merge/dedup/layout/gather + full-D attention +必要同步/传输 | 同QKV/同支持的实现收益；不等模型TPOT |
| T2 模型decode | 全模型逐步wall，明确embedding/MLP/sampling范围；`decode_wall_ms,generated_tokens,tpot_ms` | TPOT分母与首token剔除规则固定；不等prefill+decode请求 |
| T3 完整请求 | 模型入口→最后输出完成并同步；`request_wall_ms,ttft_ms,output_tokens,peak_gib` | 最终服务收益；tokenization是否含入显式记录，加载/编译另列 |

**冷/热定义按事件生命周期而非笼统warm：**
- 进程冷：加载/一次性编译/初次backend plan单列，并另报包括它们的启动体验（若已有原件），不必新增冷启动网格。
- 请求冷：新输入的subspace/摘要/索引准备、分配、必要capture/metadata全部归该请求；即使进程热也不能预热掉。
- 稳态热：已正确可复用的graph/缓存，计每token metadata/索引更新、buffer copy、replay和必要同步；重捕获发生时归触发请求。只做同shape预捕获的微基准明确适用范围。
- break-even仅作派生量：若每步净节省s>0，则额外一次成本C/s为回本步数；s≤0无回本，不拟合负例为收益。

阶段重叠用时间轴/并列绝对耗时，主结论取真实wall，禁止阶段median相加；instrumentation trace与低扰动计时分开，观测同步不混入正式统计，算法自身同步不能删除。

### 6.2 可复用与最小缺格

已有D-A6图保留8/16/32/64K全部：Step2对dense约0.32/0.54/0.94/1.64×（由四舍五入ms算；作者verdict前两档0.31/0.53×）。harness固定随机BF16 QKV，B1/Hq32/Hkv8/D128，K2=1024，warmup2、timed7中位、`perf_counter`前后CUDA同步，fresh indexer每iter，但capture在warmup完成；chunk2048、select_group64、prefetch0等以原脚本为准。此为局部prefill链路作者摘要，缺逐repeat raw，**没有模型tokenizer/样本任务质量，不进Fig.3请求主面板或质量Pareto**。

请求先核现有实际route（当前公开入口[`pred.py L229–253`](https://github.com/Chosen-David/two-level-attention/blob/b4406658aa0e028879cd7ba34c75747a5d67c6e4/benchmark/LongBench/pred.py#L229-L253)拆分question，[`L349–388`](https://github.com/Chosen-David/two-level-attention/blob/b4406658aa0e028879cd7ba34c75747a5d67c6e4/benchmark/LongBench/pred.py#L349-L388)先material forward再逐question token、再生成；具体attention路由仍须结合patch和flags；旧SS来源另见[`35b39bd8 pred.py L225–245`](https://github.com/Chosen-David/two-level-attention/blob/35b39bd8e43cf478bb4f0ab64a8457f08d87c864/benchmark/LongBench/pred.py#L225-L245)，q_pos取任务锚点与尾100字符边界的较大者，不能笼统称整个question。当前源码不替代旧run回执）：LongBench入口可能为material dense预填充、question suffix进入稀疏路径，再自然decode；不得把整个prompt自动标为dense-P。将既有原生material/question/decode路由另列，与受控整prompt dense-P/dense-D及整prompt dense-P/sparse-D分开。旧混合prompt质量不能绑定到整prompt-dense成本。受控两臂只在沿既定范围、路由与对应质量已核后比较，先从已登记请求长度选一个代表格核路由；具体长度/输出量必须由已冻结协议确定，不新选64K只因局部好看。原范围后续只补缺失长度，已完成且身份合格的不重跑。稀疏prefill另需真实route、suffix不变性或明确whole-prompt近似协议，未验不加入严格在线前缀结论。

`benchmark/LongBench/pred.py`是自然生成入口；`e127_stage_bench.py`、`e126_stage_timing.py`只是微基准。**完整请求采集→汇总→Fig.3入口尚未验收，不能给出伪完整benchmark命令。**最小工程候选是在现有请求入口补低扰动timing回执，关联相同run/config/input哈希，不新建数据矩阵。

### 6.3 正确性先于快慢

Graph修复候选已有CPU交付：以e1edf9c为基线，补丁SHA-256为`02b9349be026bef99fedb312f55c03c725152a768d40497d5422545b7119786a`；签名/schema/fail-closed的6个测试方法和22个子进程场景，经作者与独审的普通/-O四轮检查通过。此处只承接该CPU交付状态，不重复执行、不称CUDA/真实selector/生命周期或性能通过；真实capture依赖及GPU行为仍须对应验收。无GPU验收不得叫Graph已正确。随后在另获授权范围内，同QKV/同candidate集合/同L1mask/softmax支持/保护/tie/GQA/dtype，核最终IDs、finite输出和既定数值容差。覆盖边界/短序列/部分页/候选不足/NaN、alpha/gamma/保护变化导致几何变化；fallback计数必须为预期，不能将fallback成功当Graph成功。

全S、已有candidate-L2/candidate-topk、compact活跃块分别命名，先“同输出实现”比较再接动态selector。生产near avg/far minmax语义保持；若提两独立kernel/分区CTA/融合等新优化，先由正确分支的profiler定位瓶颈，再限选一个最小候选，不能默认全做新矩阵。当前仅列研究候选，未授权实施。

**成功/失败。**先正确性通过，再同质量达到δ、请求改善达到预登记η才称真实收益；kernel快而pipeline/request慢分别报告，不撤负例。报告paired请求比值CI、median/尾部（具体分位预冻结）、actual output长度和显存账单。Fig.3收益主张依赖同身份质量与所用路径自身正确性门禁；Graph门禁只阻塞Graph路径，已正确的非Graph路径可独立推进。优先P0；若只有T0/T1，主图保留待测，局部图放附录。

## 7. Fig.4：sub/sub对DenseFullKV主质量

**已知可复用而未闭合。**E123五任务FullKV原预测和result存在，但无邻接run/config sidecar；不能说FullKV不存在，也不能直接当匹配锚点。FS47.910、SS47.588、SF47.666历史均值身份不闭合，差0.322分不等价于无损。FASA LB16均值33.305625可作单臂任务谱，不能当同身份排名。RULER FASA六任务均值24与Quest十一任务均值不可混。

**最小动作。**先恢复FullKV/SS producer、sample/document/input/scorer/phase/dense-layers/精度。若旧记录恢复足够支持有限描述但不满足工具inline runcap，保留历史描述，不伪造inline sidecar、不降低门禁。不能通过补写今天的metadata把过去运行变成已冻结确认。无法恢复才登记最小缺格，并由现有负责人在已有授权范围另行决定，不自动重跑。

既定benchmark覆盖范围保持LBv1 14英文+2代码、LBv2全503、RULER32/64/128K；覆盖范围不等独立确认集。已用于选择配置的document不能因属于公共benchmark而重新标为确认；须逐document标开发/标定/未触碰确认，既有全量分数仍可作覆盖描述，无法证明独立时不承担确认推论。规模/家族只沿原登记验证，不把每个消融乘上32B/Llama。五任务只作历史/开发或经证明独立的既定子集，不能代替完整主结果。失败/超长/OOM保留分母。128K逐run核有效RoPE/YaRN与实际上下文。

**入口与实际能力。**
- 生成/评分：`benchmark/LongBench/pred.py`、`benchmark/LongBench/eval.py`，仅引用既有生产CLI并填实际冻结值，不发新增运行命令。
- 配对：`python exp/figures/paired_quality.py --protocol "$PAIRED_PROTOCOL" --out "$NEW_OUT" --plot`。本轮代码blob `1d46857515a6029b2586ad8fc3df27bfcb50eb1e`，L64–92冻结检查、L94–153原件/scorer/runcap/配对、L161–169只给诊断状态。要求两臂sub_sub/dense、`selection=dense_fullkv`，不接受FF代dense、RULER或四臂。
- 诊断协议需冻结`schema=paired-longbench-v1,status=frozen,run_mode=natural,identity,tasks/samples,seed,bootstrap_draws`；描述性诊断可令`noninferiority_delta_pp=null`，仅声明非劣效时要求事前δ及`preregistration`，逐文件及inline runcap哈希、记录哈希、document/input身份；preregistration保存事前证据，不以自报时间戳代替。
- 已有绘图仅差值/CI/零线；需适配−δ标线、绝对分数表、PDF导出，不能称已有完整论文图。

**统计与拒收。**逐任务官方分数、SS−dense百分点差、文档簇配对95%CI、n_planned/valid/failed、最差退化与总体任务等权汇总。CI下界>−δ才支持事前非劣效，跨0只说明未检出差异；等价另需双侧界。当前工具是marginal percentile CI且`noninferiority=not-assessed`，不自动完成多任务联合非劣效决策；需独审预登记多重比较规则及总体/逐任务门槛。若primary claim为所有预设任务都非劣效，采用交并检验逻辑，各任务均通过才通过整体，不机械追加Bonferroni；若挑部分任务、多个候选或多项优越性结论，另预登记多重性控制。当前双侧95%percentile区间下界对应单侧2.5%尾界，不自动等于单侧5%非劣效；冻结α和区间口径并核簇数与估计精度，至少2文档只是软件下限，不代表统计充分。未完整配对/失败不足时工具阻塞，禁止静默删样本形成主结果。P0，先于新性能故事。

**baseline最小闭合。**Quest128K先回收原有raw+receipt，MInference先回收修复后raw与frozen-pattern运行绑定；既有作者摘要不当验收。FASA 16任务原件可继续核身份，gov_report先补独立计数/原件可读性；不新增64/128K派单。ClusterKV依赖/HISA身份等保持pending，不能为凑baseline扩集成。原生first-dense层、保护、GQA及实际预算披露；统一后端轨另列而非假称native。具体状态以baseline_progress冻结快照为准，不猜宿主在跑情况。

## 8. 附录逐图：最少且能辨识主贡献

| 图/稳定ID | 复用与最小缺格 | 固定/字段/入口 | 成功失败与优先依赖 |
|---|---|---|---|
| SFig.1 / A2a | E133/E134 FS/SS/SF原件先恢复；找FF，不能先判不存在；确认与完整成本缺处保留 | 2×2实际128/32评分维度；固定K1/P/K/保护/αβγ/GQA/scale/量化算法/backend；坐标/QDQ形状和dtype实核。eval重评分，四臂配对汇总适配尚缺，不向paired_quality传四臂 | 各主对比预登记δ/η；若完整成本无改善不称4×降维=4×加速。P0语义，依赖身份/独立确认；只补缺格 |
| SFig.2a / A3 | 按当前L1分区主张优先同一page scorer的global/regional L1；仅另有L2配额独立主张时，条件解锁同QK同candidate L2区域/全局TopK | L2控制固定L1 candidates、softmax支持、GQA、K；L1控制统一scorer并固定K1/page/L2政策/表示/保护。保存candidate/selected IDs、near/far/protected unique counts、mass/output误差、选择成本；生产自然生成另轨 | 语义仅目标因素变化才准归因；自然生成状态可能分化，不继续称同QK。区间不足以区分有意义效应则标inconclusive；只有排除预定有意义增益或同质量下被更低成本支配才在测试域缩小必要性主张。旧e59是scorer replay，合格新控制入口尚缺。P0 |
| SFig.2b / A5 | 复用M2=本法与M0=dense；M1=256保护-only保留诊断；仅缺M3等总K朴素middle时列缺格 | 保护相同，总unique K1024；M3在看确认结果前选定规则及候选来源，不后挑弱者；长序列middle768，短序列按实际去重余量。保存实际IDs/quality/request成本 | M2仅胜M1不能证明内容选择；M3同质更快保留反例。旧E135的2048 trace与256-only不可补M3。生产臂/汇总仍缺，P0 |
| SFig.3 / A1+A2b+A5敏感性 | 复用E125/E128搜索；tail16/32/64既定轴、SS32复用；固定32频段；保护已登记网格 | 不做频段×宽度×模型笛卡尔积。paired-random生产臂缺则BLOCKED，旧unpaired random32不能替代；seed/坐标事前固定 | 仅说明有限候选；不在确认集重新挑冠军/保护。e125/e128/e85d/e135脚本原协议和硬编码先审。P1，主结果以后 |
| SFig.4 / O2+A4+A6 | 复用E137/E129及阶段四臂原件；G0/on、G1/off-refill、G2/off-no-refill仅有真实实现/轨迹者展示 | 保存calibration/selection far IDs、pending/freeze/clear、真实first decode token、每seq/layer/head漂移、阈值/rows、校准同步成本及实际预算；G2少token非sameK。prompt-only trace不能补decode漂移 | 静态cclustersim不替mavg prefill标定；标定低mass非安全证明；动态新研究暂停，未验G2不排队。已有gate仅限测过范围。P2条件性 |
| 工程附录 / A6 | D-A6-Step2全长负例、Graph候选修复测试、compact已有报告 | 同输出门禁→T0/T1→模型/请求；记录capture/replay/fallback、cache几何/schema、内存与流支持域 | CPU逻辑候选不等CUDA/数值/生命周期通过；仅改善旧慢实现不称vs高效dense胜出。依赖独审与另行授权 |

## 9. 最小优先清单：实际操作、交付、停止点

此表是待采纳执行顺序，不是新派单。每行仅在前置满足且已有授权覆盖时实施；此轮可以完成的仅文档/离线原件审查。任何需要GPU的新格保持BLOCKED，不因有脚本就运行。

| 优先/依赖 | 动作与最小产物 | 可执行入口/明确阻塞 | 完成判据 |
|---|---|---|---|
| P0-1 无 | 建`reuse_ledger`：每个已有run归为reusable / identity_missing / semantic_failed / absent，并列证据/原hash | 从source_manifest、baseline_progress及已有result/raw回执逐项连接；不重评分前就下missing结论 | 每个主图点指向真实原件，所有unknown保留；完成格锁为reuse |
| P0-2 P0-1 | 冻结E0身份、独立确认划分、δ/η和测量计划 | 恢复原协议；缺负责人决策则UNFROZEN。若历史仅开发，禁止事后“预注册” | 文档交集检查、原始时间/来源可信；所有必要字段非猜值 |
| P0-3 可并行 | 复用已交付Graph CPU修复正负门与独审结论，核补丁hash并输出支持域/未验项 | 沿现有修复任务，不重复造candidate-L2/compact；真实CUDA验证未授权 | 普通/-O正负门、缓存重用/变更/回退观测正确，仅标CPU范围 |
| P0-4 P0-1及诊断协议；确认性结论另依赖P0-2 | 用已有匹配两臂原件生成Fig.4诊断数据 | paired_quality现成接口；inline回执/独立身份缺则阻塞，不伪造 | 图数据绑定通过、未损失分母、独审接收；非劣效结论另按预注册审 |
| P0-5 P0-2；仅Graph路径另依赖P0-3 | 补请求计时采集设计与一个既登记代表格的执行包 | pred.py真实入口；完整timing harness尚缺，先实现候选/审查，不运行GPU | 四层边界、冷/热事件归属、同配置质量绑定、无重复统计；获得许可才测 |
| P0-6 P0-1/2 | SFig.1四格逐一按ledger判定，仅补真实不可复用格；SFig.2仅纯配额/等K朴素middle真缺臂 | 固定QK控制入口及四臂汇总有缺；先语义审查，缺实现保留BLOCKED | 只改目标因素；不得将历史三均值画成因果验证 |
| P1-1 P0-1 | Fig.1离线O1适配+O3既有trace，Fig.2源绑定 | O3现有CLI；O1新口径适配尚缺；不采新trace | provenance/支持/统计单位齐全；无decode或生产等价越界 |
| P1-2 主结果闭合 | 仅沿原范围补确认/长度缺格与必要单轴附录 | 已完成锁复用；需GPU仍待已有负责人授权 | 质量及请求主张在已声明范围通过或如实失败 |
| P2 条件 | gate/阶段四臂/进一步kernel专用调度 | 新动态研究暂停；无主张则不新增 | 相应主张不成立即收缩，不为图齐全而继续扩展 |

离线分析可复用的真实CLI只有O3、paired_quality及figure_data_contract。变量必须绑定真实审核原件和新输出目录；没有协议或缺依赖时停，不用示例值运行。现阶段没有合法的一键GPU总跑脚本，也不应构造一个。

## 10. 最终论文验收的可证伪结论

- 若SS质量通过预登记容忍而请求无净收益：论文可报告质量/选择机制与实现限制，不能称serving加速。
- 若局部kernel获益仅64K：完整保留8–32K负例，结论限定该算子/该配置/该范围，不能“推测请求也会更快”。
- 若区域控制与global差异的区间跨越预定有意义增益，结论为inconclusive；只有区间排除该增益，或朴素middle在匹配质量下以更低成本支配，才在所测域收缩相应必要性主张。均不得改变确认集或另选更弱对照。
- 若旧身份不能恢复：历史图保留“描述/作者报告”，正式点留待验证，不能把NR填成默认来制造完整论文。
- 任何新增实现/实测都在独立审查后另建run身份；本设计完成不等于实验完成、结果采用或用户授权GPU执行。

## 11. 逐论文测量细节如何真正改变本设计

本节与附后的《论文测量口径逐字段对照》联读。其P/C/NR表逐字段保存论文披露、当前官方代码和未知，**不把三者拼成一份原论文运行配置**。Quest固定v2和代码01c1623，TokenSelect固定EMNLP2025正文和代码7e16108，Twilight固定NeurIPS2025正文和代码ee9b1c4；对应论文实验commit均NR。下面Q/T/W文件代号及准确固定链接均在对照表第1节，不使用浮动main引用。

| 细节/出处 | 采用、调整或不适用 | 本项目具体落地及验收 |
|---|---|---|
| Quest §4.3.1图8/9、§4.3.2图10（PDF物理页7–8）把kernel/attention/模型decode分开 | **采用并扩充边界** | Fig.3执行T0–T3四账单；只有T3可称request收益。图10排sampling不可沿用为我们的完整请求定义；sampling开销存字段，TPOT分母写清 |
| Quest Q1 L19–36模型FP16本地路径，无revision；T1 L36/50 tokenizer无revision；W1合成输入无模型 | **调整** | 本法与baseline都锁checkpoint/tokenizer hash及revision；合成kernel运行模型字段记N/A而非伪填Qwen3，正式质量运行缺revision=identity_missing |
| Quest Q1 L81/90随机embeddings；Twilight W1随机Q/KV/labels/score与独立后级输入 | **仅作局部诊断** | 复用Step2随机QKV的T1证据，不进入E1/请求质量；同输出控制用真实同一QKV/IDs并传递上游输出，不能分别随机后级再相加当pipeline |
| TokenSelect 图7 chunk512、KV128K/512K/1M；T2 chunk8192与论文不同 | **不适用其规模；采用配置显式绑定** | 本项目沿已登记长度，不补1M/batch新轴；Step2 chunk2048只作该harness身份，不移植到正式请求默认。producer实际chunk必须回执 |
| Quest Q2 decode256/iteration5；Q1默认iteration10被覆盖 | **不照搬参数，采用effective-config核验** | 输出长度/repeat以现有冻结协议为准；同时保存CLI/env解析后的有效值，禁止从脚本默认猜旧运行。没有正式值不启动新格 |
| TokenSelect T1 L262–269 resume跳过旧pred；L310–314超长skip；L360记录prompt+max_gen非实际输出 | **调整并加拒收门** | 每请求timing必须绑定本次真实completed样本ID与实际生成token数；历史resume样本可复用quality但不可计本次速度分母；所有skip/OOM/失败保留planned分母 |
| TokenSelect T2 initial128+local512+k2048；T3 L438–448组合；Quest Q3 L112–119预算含末页 | **采用实际支持记账，不能抄名义K** | same-K按最终union含保护；选集实际去重、页padding、每head/group union存回执。跨方法K1不必一致；native预算不对齐者另报分布而非强凑 |
| Quest首两层dense；Twilight W3 skip_first_two_layers；TokenSelect该入口未声明相同规则 | **保留native并披露；受控轨才匹配** | 逐层route/first-dense列表；不能向所有方法植入“前两层dense”默认。两轨分名，dense层差异不得混成selector因果收益 |
| TokenSelect T3 L361 soft vote/L371统一TopK；Quest Q3 head buffer；Twilight W1 flatten head/num_qo_heads=1 | **不强行统一native，机制轨显式匹配** | 本法32Q/8KV组内mean的实际次序固定；记录每head及group union。W1算子几何不证明Qwen3 GQA serving适配。改变共享语义则新方法身份 |
| Quest图10 FP16/4-bit weights；TokenSelect T2 BF16；Twilight W1 FP16+packed INT4 pruner | **不拼接dtype收益** | 分开weights/KV/selector QDQ/accum dtype及物理存储，主attention固定原维。统一后端消融不得顺手换量化；本法浮点QDQ不能标packed-int4 |
| Quest Q1 L80–95 host perf_counter无外层显式同步；图8用NVBench | **调整，不能盲抄timer** | 模型/request wall前后清楚同步并声明边界；device event用于stage。baseline同规则。外层未同步不证明Quest底层无同步，本项目不据此判论文错，只不沿用未闭合timer |
| Twilight W2 L18预热后同步、L24–29 timeit；W1 L644–645 warmup100/repeats5000覆盖默认5/10 | **采用独立warmup/repeat披露，不抄次数** | 从现有登记预算/方差目标冻结repeat与warmup；按成对交替顺序测；记录实际Torch版本及timeit同步行为。计时重复不是新增独立文档N |
| Twilight W1 L562 metadata、constructor plan在测量外；L588–604三部分测后求和 | **只作T0参考，不作为T1/T3方法** | 完整attention以连续数据流wall测；prepare/plan/metadata更新分别归offline/每请求/每token；重叠不stack相加；禁止sum-of-means/medians等于请求 |
| TokenSelect T1 L396–448 dataset wall含load/Runtime/tokenizer创建与销毁；图7专用计时入口NR | **调整** | 区分进程启动与请求冷、稳态热。主图模型入口到末输出；模型加载/编译另列。不可据T1断言论文图8实际用了此边界；我们提供精确入口/结束事件 |
| Quest均值、Twilight统计声明无error bars、TokenSelect每样本端到端但有效分母NR | **增加可审计统计** | 本项目质量按document簇配对CI；timing按request配对比值/重复波动，记录尾部和失败；预注册CI/δ/η，不因文献无区间就省掉本法区间 |
| Quest图11预算映射FlashInfer代理；Twilight图8完整TPOT入口NR；TokenSelect当前代码早于正式版 | **不作为faithful复现验收** | 论文最大倍率只作相关工作，不能排本库名次。每个baseline方法×模型×kernel支持分native/已验port/pending；现有作者摘要与本库实测严格分开 |

### 11.1 请求原始计时行的最小接口候选

以下是待实现采集器的逻辑字段，不是声称现有脚本已输出。选择这些字段直接针对上述论文口径差异，避免仅泛称“做E2E”：

- 身份：`run_id, protocol_sha256, pair_id, sample_id, document_id, input_ids_sha256, config_sha256, producer_commit, backend_id, instrumentation_mode`。
- 实际工作：`phase_route_by_layer, first_dense_layers, actual_ctx_tokens, actual_generated_tokens, eos_reason, max_new_tokens, fixed_workload_or_natural, batch, chunk`。
- 计时边界：`wall_start_event, wall_stop_event, tokenization_included, sampling_included, sync_policy, timer_api, repetition, order, warmup_count, cache_state`。
- 四层读数：`request_wall_ms, ttft_ms, decode_after_first_wall_ms, first_output_timestamp, last_output_timestamp, per_token_wall_ms`；无第二token时TPOT记N/A而非零。
- 生命周期：`compile_ms, plan_ms, capture_ms, recapture_count, prepare_ms, metadata_update_ms, injected_copy_ms, replay_ms, fallback_count`；各字段声明included_in_request/overlap，不默认可加。
- 连续pipeline：`selector_pipeline_wall_ms`及L1/page_topk/L2/token_topk/merge/dedup/gather/attention阶段事件；原始event顺序保留，独立kernel profiling结果另run，不能拿其和覆盖pipeline。
- 支持与内存：实际unique protected/near/far/total，K1 page/candidate token、Q-head/KV-group共享方式、原KV/摘要/QDQ/临时缓冲allocated与reserved峰值、H2D/D2H bytes。
- 失败：`status,error_kind,retry_of,planned,measured_this_run`；resume旧记录不可写成measured_this_run。原件manifest保留失败，不在汇总前消失。

TPOT候选定义为首输出之后的decode wall除以`actual_generated_tokens−1`，需以实际程序第一/第二token事件核实，不能凭公式把prompt最后logits算成decode。自然输出长度不同则同时报告真实请求成本与既定固定工作量诊断，不按长度比线性“校正”成未测速度。

请求速度比固定为同一pair的`dense_wall / method_wall`，统计时保持同输入配对和运行块结构，不能把所有replay当独立请求。δ/η究竟用百分点、绝对ms或相对改善需先在协议明确；η尚未冻结时不宣布“达到实际价值”。选择分解profile若影响wall，独立保存，不用instrumented wall替代主计时却不披露。

## 12. 首批复用账本：已经定位的原件，而非从零再盘点

以下路径全部相对公开仓库，除Step2另注明外锁在 `ed15763fcd7c8c42bbbeca89f7d3a246b5c08dca`。本次目录回读确认存在不等于完整运行身份验收；Git blob SHA与文件SHA-256不可混用。

| 臂/用途 | 精确原件入口与已知身份 | 当前状态/缺项 | 下一项只读动作；通过后能做什么 |
|---|---|---|---|
| DenseFullKV/五任务 | `exp/trace/results/e123_trial_raw/pred_fullkv/result.json`，blob `4f6de1a220a0befb2ef6c3d20df95ac7ef07b835`；五个`{gov_report,hotpotqa,musique,qasper,repobench}-none-1010201318.jsonl`；另有`exp/trace/results/e123_trial_raw/e123_fullkv_gpu1.log` | identity_missing，目录没有run/config旁车；非“没有FullKV”。checkpoint/tokenizer/input/document/scorer/phase与当前SS匹配未证 | 先读该log及原producer版本，逐sample核输入/实际prompt路由；不能恢复inline provenance则保留历史quality，不喂paired_quality伪造配对 |
| FS历史锚点 | `exp/trace/results/e123_trial_raw/pred_mavg/result.json` blob `1ef509cea7cdae15c667647089ed45121073bbae`；同目录五任务`*-tli_64_128_1024_c4_a0.25_b0.125_g0.625-1010201318.jsonl`；`exp/trace/results/e123_trial_raw/e123_mavg_gpu0.log` | identity_missing，47.910是名义FS，目录无sidecar；不能默认所有有效参数同当前 | 读log及生产文件哈希绑定actual L1/full与L2/sub、QDQ/GQA/保护/phase；不以文件名和AVG补运行身份 |
| SS主臂历史五任务 | `exp/trace/results/e134_l1sub/pred_mavg_l1sub/`；如`qasper-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h0a1708cf4e-1012180214.jsonl` blob `96710ee2701412caeb5e66c0a2842c03358caffe`，同名`.runcap.json` blob `6f54671a1f5ca262f6dc8f33c1fd56d9f9f08b03`、`.tli_manifest.json` blob `cd5bdef9fdd938b4e9dd981e94c33cbaed856c85`；五任务均有对应旁车 | identity_missing至独验完成；不因有旁车自动可与旧FullKV配对。该目录未发现result.json，不再猜文件存在 | 先读这组旁车逐项对齐producer/cap/scorer/实际路由，再核其余任务document/input。47.588仅历史描述，旧E133的task-bootstrap不替document配对统计 |
| SF历史五任务 | `exp/trace/results/e134_l2full/pred_mavg_l2full/`；如`qasper-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h2d1dc6ec58-1012195423.jsonl` blob `57494d0b1fbf4ddcfb5c3535e77b3aa8c5482696`，同名runcap blob `16e558df6ec5eef9272f0f9864f0d5e1b0e1b0dd`、tli_manifest blob `6e310e18478e2bb421a6c264ef24faef47c4b715` | identity_missing；L2full模式与实际生成代码绑定未闭合，不预判semantic_failed | 比较manifest实际producer/dirty文件与L2full实现，核QDQ形状/坐标及phase；闭合前47.666不得作2×2因果点 |
| FF稀疏臂 | 当前给定产物中未定位合格原件；不等于断言全仓/宿主绝对不存在 | absent_in_reviewed_artifacts；四格逐一判断，FF不是唯一可能需补的格 | 仅查既有任务索引/producer产物清单是否有FF，不全面重搜/重跑；确无且必要才登记缺格 |
| Step2工程诊断 | 实验SHA `e1edf9c316fc4190dd0c29bc5d47477136f8bdab`：`exp/trace/results/e139_step2_cand_verdict.json` blob `1ecb5851eb9c1de087ca2a0ba34982b2dd034556`；`exp/bench/e139_step2_wall_lsweep.py`；原件清单保留两者SHA-256 | reusable_descriptive：8–64K作者摘要/harness可复用；逐repeat raw、真实模型请求、质量绑定缺；不是absent整条结果 | 先查已有原始timing日志回执（不重复测速）；找不到则继续只画摘要且无CI，不升级Fig.3/请求Pareto |

### 12.1 请求统计的具体预登记条款

采用同prompt/同配置的AB/BA交替或固定种子的随机交错顺序，精确顺序和运行块写入协议。重复嵌套于请求/源文档，而不是把每次replay算独立样本。建议主estimand为预登记文档权重下的配对log速度比均值再指数化，即几何平均`dense_wall/method_wall`；绝对paired毫秒差另报，不在看到结果后挑有利汇总。若既有协议已冻结其他estimand则保留并明确，不事后替换。文档簇重采样保留同一请求的运行块和配对关系；计时重复只估计运行波动。

自然生成的EOS/输出长度变化属于真实服务成本结果，单列输出分布及对应质量；**早停节省不能归因为kernel加速**。kernel/selector归因依赖另外的固定工作量或固定QKV控制，不用token长度比例代替实测。


---

# 论文测量口径逐字段对照：Quest / TokenSelect / Twilight

核验日期：2026-10-15。范围只覆盖与当前分层selector、候选attention、Graph成本最相关的核心测量；不是扩展实验矩阵。所有代码只读，未运行实验。

**读表规则**：P=论文明确披露；C=本次固定官方代码快照披露；NR=在本次核查的对应论文实验/源码中未报告或未能确定，不表示作者绝对没有资料。P与C不能拼成一个虚构的“原论文复现配置”。官方当前commit不等于发表实验commit，后者三篇均未建立可验证映射。下列页号统一采用PDF物理页（从1开始），避免出版社页码变更。

## 1. 版本锁定与出处

|项目|论文版本|固定官方代码快照|对应关系限制|
|---|---|---|---|
|Quest|[arXiv 2406.10774v2 PDF](https://arxiv.org/pdf/2406.10774v2)，2024-08-26；[ICML2024正式记录](https://proceedings.mlr.press/v235/tang24l.html)|[01c1623bf9395009520874e989e29f683203b357](https://github.com/mit-han-lab/Quest/commit/01c1623bf9395009520874e989e29f683203b357)，2025-07-10|本表页码/实验细节以v2为准；正式摘要页曾反写两项倍率，按正文图9/10解释。原实验commit NR|
|TokenSelect|[EMNLP2025正式PDF](https://aclanthology.org/2025.emnlp-main.1079.pdf)；[arXiv v4](https://arxiv.org/html/2411.02886v4)，2025-10-09；此前报告核v3|[7e16108d6e1e0ee6a514404f34961d57679028f4](https://github.com/pzs19/TokenSelect/commit/7e16108d6e1e0ee6a514404f34961d57679028f4)，2025-03-11|本表论文页码以正式PDF为准；代码早于最终版本。不能假设当前配置等于图7配置|
|Twilight|[NeurIPS2025正式PDF](https://papers.nips.cc/paper_files/paper/2025/file/ad2f57116d62a9d8dcfaba6a168715e4-Paper-Conference.pdf)；[arXiv v5](https://arxiv.org/html/2502.02770v5)，2025-11-04；此前报告核v4|[ee9b1c42e2f770cde7b3e028260dc6876d2dccda](https://github.com/tsinghua-ideal/Twilight/commit/ee9b1c42e2f770cde7b3e028260dc6876d2dccda)，2026-08-31|当前代码比论文晚；不得用它倒推论文TPOT脚本的所有计时细节|

固定文件与行号：

- Q1：[scripts/bench_textgen.py](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/scripts/bench_textgen.py#L13-L103)
- Q2：[scripts/bench_efficiency_e2e.sh](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/scripts/bench_efficiency_e2e.sh#L1-L10)
- Q3：[quest/utils/controller.py](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/quest/utils/controller.py#L97-L120)
- Q4：[quest/models/QuestAttention.py](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/quest/models/QuestAttention.py)
- T1：[benchmark/pred.py](https://github.com/pzs19/TokenSelect/blob/7e16108d6e1e0ee6a514404f34961d57679028f4/benchmark/pred.py#L245-L455)
- T2：[config/qwen-token-retrieval.yaml](https://github.com/pzs19/TokenSelect/blob/7e16108d6e1e0ee6a514404f34961d57679028f4/config/qwen-token-retrieval.yaml#L1-L15)
- T3：[patcher/token_retrieval.py](https://github.com/pzs19/TokenSelect/blob/7e16108d6e1e0ee6a514404f34961d57679028f4/patcher/token_retrieval.py#L338-L448)
- W1：[benchmark/efficiency/bench_breakdown.py](https://github.com/tsinghua-ideal/Twilight/blob/ee9b1c42e2f770cde7b3e028260dc6876d2dccda/benchmark/efficiency/bench_breakdown.py#L467-L645)
- W2：[benchmark/efficiency/bench_utils.py](https://github.com/tsinghua-ideal/Twilight/blob/ee9b1c42e2f770cde7b3e028260dc6876d2dccda/benchmark/efficiency/bench_utils.py#L5-L29)
- W3：[benchmark/configs/config_quest_twi.json](https://github.com/tsinghua-ideal/Twilight/blob/ee9b1c42e2f770cde7b3e028260dc6876d2dccda/benchmark/configs/config_quest_twi.json#L1-L20)

## 2. 核心论文测量：先区分测了什么

|字段|Quest|TokenSelect|Twilight|
|---|---|---|---|
|主要算子测量|P：§4.3.1，物理页7–8，图8独立kernel、图9组合attention|P：§6.4，物理页9，图7 single-chunk prefill；图5另有分解|P：§5.2，物理页8–9，图7 self-attention；§5.3页10图10分解|
|模型级测量|P：§4.3.2页8，图10 decode单token平均|P：图8，每样本端到端；附录H页16–17表8另测prefill|P：图8 TPOT，decode服务，不是含prefill请求|
|模型|P：Llama2-7B效率配置；质量模型为LongChat/Yarn，不能混为同一模型|P：图7/8 Qwen2-7B-Instruct；表8 Llama3-8B|P：§5.1列Longchat7B、Llama2-7B-Chat、Llama3.1-8B；图7/8各柱完整checkpoint映射NR|
|模型revision/tokenizer revision|P：均NR|P：均NR|P：均NR|
|样本及长度|P：8K/16K/32K等长度；效率输入的样本ID/hash NR|P：图7 KV128K/512K/1M、chunk512；图8 InfiniteBench，逐样本ID/hash NR|P：Qasper/GovReport/LCC，10K/20K/30K；抽样ID、截断/补齐细则NR|
|budget/保护|P：图9关键点2048；page16；首两层dense|P：默认k2048+local512，另保留initial128；θ0.9；不能将k当全部token数|P：首两层dense；Longchat p=.85，Llama2/3 p=.95；图10 selector8192→约256|
|head粒度|P：page打分/选择，代码Q3按head buffer|P：per-head softmax后跨head soft vote；不是每头独立最终集合|P：自适应head预算；当前W1微基准另有flatten-head实现|
|precision|P：图10 FP16权重与4-bit权重分列；不能假定KV也4-bit|P：论文核心效率图完整dtype组合NR；C见下一表|P：pruner低精度机制见§4.2；图7/8完整权重/KV/累加dtype组合NR|
|硬件/backend|P：RTX4090 CUDA12.2 kernel；Ada6000模型级；FlashInfer / RAFT / NVBench|P：A100；SGLang+FlashInfer+PyTorch/Triton；>1M另用TP|P：单A100，CUDA/Triton；SDPA FA2/MEA、FlashInfer、改为batch支持的Quest|
|warmup/repeat/同步|P：图8 NVBench；图9 PyTorch profiler；具体次数/同步边界NR|P：图7/8具体计时器、warmup/repeats/同步NR|P：图7/8具体计时器、warmup/repeats/同步NR|
|统计/有效分母|P：图10平均每decode token；误差条、完整排除规则NR|P：图8每样本；实际成功/失败/跳过数量及平均分母NR|P：TPOT；统计检查单物理页23明确不画error bars；具体输出token总数/异常排除NR|
|核心基线|P：图9/10 FlashInfer full；图11其他算法为预算映射到FlashInfer的定性估计|P：图7 FlashInfer/InfLLM；图8 SDPA/InfLLM；不能当作对Quest的同机速度排名|P：图7 FA2/FlashInfer/Quest；图8 FlashInfer/Quest|
|收益解释|7.03×组合attention；2.23×对应量化decode，非整请求|23.84×对应图7 prefill；2.28×相对InfLLM图8|图7正文15.8×相对FA2，图8最高3.9×相对FlashInfer；相对Quest增益较小|

上述不披露字段不得用其他论文的默认值“补齐”；也不能用当前repo默认值冒充历史实验值。

## 3. 固定代码逐字段：可复现入口实际做什么

|字段|Quest Q1/Q2|TokenSelect T1/T2/T3|Twilight W1/W2/W3|
|---|---|---|---|
|模型及revision|Q1 L19–36：本地Llama-2-7b-chat-hf路径；from_pretrained无revision参数|T2 L3：Qwen/Qwen2-7B-Instruct；T1 L36/50 tokenizer同路径，无revision锁|W1是算子合成输入，无完整模型checkpoint|
|tokenizer|Q1 inputs_embeds绕过tokenizer|T1 L50 AutoTokenizer；L291–308按任务套chat模板和special tokens|W1不适用；真实服务tokenizer NR|
|数据/seed|Q1 L81/90每次torch.randn；无seed设置|T1 L413/415本地InfiniteBench/LongBench；数据revision/hash NR|W1随机Q/KV、label、topk分数、logits；无seed固定；不是真实prompt前向轨迹|
|长度/生成|Q2 8192/16384/32768；decode256；Q1不autoregressive采样，而随机单步embeddings|T2 max_len1048576；T1 L303–334 tokenize后保初始128+尾部截断，可能decode/re-encode；无truncation时超长直接skip|W1 default seq_len30000、head32、dim128；batch16/32/64；不是图7全部batch配置|
|预算保护|Q1 page16；Q3 L112–119 clamp有效页数，预算包含最后页，topk只选budget−1页|T2 initial128/local512/k2048；T3 L377短历史全保留，L438–448拼初始/候选/local；最终必须按实际去重后数量记账|W3 selector8192、chunk16、INT4估计、p.85；W1 b0=8192/b1=256/p.85，b1仅profile路径，不代表top-p实际结果|
|dense层|论文首2层；Q4由controller预算控制，不能把单文件无显式layer判断当全部稀疏|T1/T2未声明首两层dense；不能套用Quest规则，完整layer逻辑需另审|W3 skip_first_two_layers=true；W1本身无层循环|
|head粒度|Q3 topk buffer [num_heads,budget−1]|T3 L361跨head求和，L371统一topk并L372排序；GQA映射需结合kernel，不是简单每头各k|W1 top-p结果flatten batch×head；稀疏attention num_qo_heads=1；不能把该分解当完整GQA serving|
|precision|Q1 ModelConfig float16；该入口未显式4bit加载，不可复现图10量化柱|T2 dtype=bfloat16；图7论文chunk512而T2 chunk_size8192，已存在配置差异|W1 FP16 Q/KV/metadata；pruner INT4 packed into INT8；FP16 scale/zero；FP32累加细节需kernel另审|
|backend/Graph|Q4 custom CUDA wrappers；Q1无Graph配置；编译/底层同步NR|T1 L69–84 SGLang Runtime：disable_cuda_graph/radix/disk/flashinfer_sampling等，max_running_requests=1，TP参数可变|W1 FlashInfer paged decode NHD/use_tensor_cores=False；W2 torch.utils.benchmark；未走CUDA Graph replay|
|计时范围|Q1 L80–95 host perf_counter，randn与model call均在内；load/init在外，clear在外；prefill与decode分列|T1 L396开始dataset计时，含load数据、L427建Runtime/tokenizer、tokenize/HTTP请求、L445删除model及empty_cache；L448结束，结果文件写入在外|W1 L532 selector+topk；L588–595 SpGEMV/top-p/attention分别测；L604求和；并非连续执行一条输入数据流|
|同步|Q1计时区无显式torch.cuda.synchronize；底层可能同步但未完成闭包审计，不能证明host time是完整GPU时间|HTTP同步等结果不等于单kernel同步；图7低层timer未定位|W2 L18 warmup后显式同步；L24–29用torch benchmark.Timer.timeit；同步细节受安装torch版本实现约束|
|warmup/repeats|Q1无独立warmup；Q2 iteration5×decode256；Q1默认iteration10，被Q2覆盖|T1无独立warmup/repeated runs；每个剩余样本一次HTTP调用|W1 L644–645 warmup100/repeats5000覆盖W2函数默认5/10|
|误差统计|Q1 np.mean，无方差/CI/quantile|T1 dataset elapsed标量，未输出per-request方差|W1 L469只打印timing.mean；无CI/尾延迟|
|有效分母/排除|Q1 decode均值覆盖iteration×decode_len；随机数据、无EOS早停|T1 L262–269已有preds被跳过但保留在返回列表；L310–314超长可skip；L360 token_length是prompt+max_gen而非实际生成数。总耗时除返回pred数会混入历史记录，不能直接如此计算|W1每组件timeit重复调用的平均；sum-of-means不包括阶段衔接开销；不是每请求或每模型token分母|
|prepare/gather范围|Q1模型初始化在外；Q4 forward包含KV append等，但是否完整计入GPU完成时间未知|T1请求范围含服务处理，未提供图7selector/gather/plan独立账单|W1 L562 metadata构建、L336/398/450 plan、allocation都在forward计时外；独立随机topk/pruner输入，不保证selector输出真实传给后级|

## 4. 对当前项目实验的直接约束（不扩矩阵）

1. 现有8–64K测量继续保持原定范围；补齐固定模型/tokenizer revision、prompt token IDs/hash、实际输出token数及相同成功样本ID。无须为了匹配三篇而额外加入百万token或batch256。
2. 把普通全S、e1edf9c候选内、93e4a3活跃块分列；Graph缓存几何/schema与fail-closed修复验证完成前，不接受其速度作为主结论。
3. 在同一批既定样本上同时记录原有算子口径及完整链路口径：L1/L2/topk/merge或去重/gather/main；prepare与Graph capture另列。既不能剥去我们额外成本，也不能把baseline额外成本漏掉。
4. 同一预算定义中记initial/local/dense层/页padding/两路重叠后的实际token数；先按匹配质量比较，再给固定预算诊断。不要套用TokenSelect“2K+512”而漏initial128。
5. 主结果用有明确CUDA完成边界的计时；hot replay和cold prepare/capture分开。不要照抄Quest当前host timer；不要把Twilight sum-of-kernels叫完整请求；不要按TokenSelect当前dataset wall-time除含resume旧结果的数量。
6. 对论文中NR项，我们自己的run manifest必须明确；这属于可审计性补充，不是据此指责论文结论错误，也不构成新的算法创新。

## 5. 必须保留的不确定性

- 三个官方快照与发表实验未建立commit映射；正式代码release/tag与artifact数据是否完整还需作者或归档证明。
- Quest底层同步闭包未审完；本结论仅限公开计时脚本外层缺同步。
- TokenSelect图7低层专用benchmark未定位；T1是仓库评测入口，不能断言图8以其dataset计时生成。
- Twilight完整模型TPOT实现入口未定位；W1只能审计当前operator breakdown，不可替代图8复现。
- 没有把当前项目1.64×与上述论文最大倍率相除，也未据此宣称领先、落后或创新成立。


---

# 最小执行优先清单（设计候选，非派单）

沿用既有要求：最终去重K含保护、SS主方法/DenseFullKV质量锚点、同身份同phase、固定确认集/容忍、完整请求成本、既有矩阵不扩、完成格复用。

本次具体化：精确原件复用账本；Graph仅阻塞对应路径；prompt-material/question/decode路由分账；论文逐字段P/C/NR表与采用规则；请求AB/BA及嵌套统计；O3保护质量拆分缺口。

1. 先读详细方案§12的FullKV、FS、SS、SF原件及旁车/log，逐格判reusable / identity_missing / semantic_failed / absent；不重新全面盘点，不从当前默认反填历史。
2. 核实际material/question/decode路由及dense层，禁止旧混合prompt质量配整prompt-dense成本。benchmark覆盖与未参与选型的document确认集分别登记。
3. 推进非Graph请求采集设计，同时复用已交付Graph候选CPU检查、保留真实GPU未验边界；Graph门禁只阻塞使用它的路径。所有新GPU执行仍需原授权覆盖；本文件不启动。
4. 先用身份合格的SS–Dense原件形成配对诊断；无事前δ可描述差值，不能宣称非劣效。正式质量收益需独立确认；历史图、CPU检查不必等δ/η。
5. 仅从已有登记请求协议选代表格，冻结完整timing字段、AB/BA顺序、输出工作量、重复/统计；完整请求采集入口未验收，先补入口候选而不是伪造一键运行命令。
6. 仅补A2四格中真正不可复用格，以及必要的纯配额/等K朴素middle控制。跨方法允许不同K1；配额因果控制固定统一scorer、K1/page/L2/表示/保护。
7. 用既有trace离线作O3；当前仅全causal质量诊断，进入Fig.1还缺保护质量拆分。O1新口径离线汇总缺；Fig.2只绑可编辑源，不新采GPU。
8. 保留8–32K负收益与64K局部摘要。Gate现有测量限原范围；新增动态α/β/γ、D′、新baseline、大规模网格继续不排队。

执行入口与冻结条件见本文第4—9节，原件账本见第12节；文献固定版本/源码行号见附后的《论文测量口径逐字段对照》。未知值保持NR/UNFROZEN，不补默认。
