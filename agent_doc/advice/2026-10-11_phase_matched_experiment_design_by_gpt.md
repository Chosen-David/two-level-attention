# 阶段匹配的静态实验设计与基线接入建议
@DeepSeek

日期：2026-10-11 UTC。核查基线：[two-level-attention main e5d09f8](https://github.com/Chosen-David/two-level-attention/tree/e5d09f8c62ee05211e5f6c2a5f3f7792485da664)。
本文供既有 S-T023 与静态实验链修订使用，不新建任务，不代表实现、模型运行或收益验收。
本文更正此前baseline addendum对SparQ补偿项的一概要求，并补清MoBA实际phase等边界；冲突处以本文固定版本核查为准。
当前顺序为静态 method 筛选、合法参数枚举、冻结配置、机制对照、阶段匹配确认；动态计算器和动态 gate 继续暂停。
本说明仅提出设计，不代表已启动、停止或改变外部作业；已确认污染缺陷仍按已有停跑/隔离/修复/新身份重跑规则处理。
不据此启动实验、训练、下载或付费资源，也不替作者修改实现；后续执行须沿用已有授权与门禁。

## 1 先纠正研究问题与预算口径
目标是检验：共享评分表示、near/far 两区 L1 页筛和 L2 token 筛，能否在最终原始全维 KV attention 下改善质量与完整请求成本。
同层同 KV 头先构造共同表示，再分 near/far，两区各做 L1/L2；不能以两个独立表示替代这一研究对象。
D′ gate仅作用far；静态门控若纳入候选，固定其开关/规则，不借机制实验恢复动态gate。
prefix/SWA 是去重后的共同保护条件，包含在最终 K2 总预算中，不把“保送、不竞争”写成“预算外免费增加”。
例如 K2=1024、有效保护集合为256时，middle竞争名额为768；短输入、重叠保护、部分页与不足候选须按真实集合计数。
完整 KV 仍驻留时，少选 token 只可能减少读取/计算，不能直接宣称 resident KV 容量下降；FullKV是dense锚点，不保证任务分数上界。
当前 HF `full/c4` 是 L1 full、L2 tail32；真正 full/full 对照需先验收两阶段全维评分支持，不能拿名称代替实际路径。
Qwen3 D=128 的 paired tail32 为 [48..63]∪[112..127]，即16个完整低频RoPE对，不是连续最后32坐标；`nope`切半标签不证明物理NoPE。
MLA若另做迁移，先核潜变量、RoPE/NoPE和GQA接口；S-T023的分别索引设想是独立架构变体，不能静默改写共享表示契约。

## 2 固定论文和官方实现范围
以下版本用于本轮机制核查；论文实验、作者代码与本库适配分别登记，代码存在不等于接入已通过。
- Quest：[论文 v2](https://arxiv.org/pdf/2406.10774v2)，[官方库 01c1623b](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357)。
  无模型训练，dense prefill、query-aware page sparse decode；论文§4.3.2的E2E是整模型单token decode平均延迟并排除sampling，不是完整prompt+generation请求。
- ClusterKV：[论文 v2](https://arxiv.org/pdf/2412.03213v2)，[官方库 c7380819](https://github.com/sjtu-zhao-lab/ClusterKV/tree/c7380819588771e8c5dadacdbd33942d7c53adea)。
  论文系统prompt仍做dense attention，同时建簇/异步offload并尝试重叠，decode选簇；建索引不等于sparse prefill。§V-C/Fig.12另报包含prefill的请求时延，accuracy代码的建簇时点另见附录。
  原系统CPU full-KV与GPU cache须保留其成本边界；GPU-resident selector重实现另标。仓库整体许可未核清前，不默认可直接vendor。
- FASA：[论文 v3](https://arxiv.org/pdf/2602.03152v3)，[官方库 96c00c46](https://github.com/wangyifei0047/FASA-ICLR2026/tree/96c00c46db12ccee56104175470aa29579d66c58)。
  无模型训练但需离线频率块校准；Appendix B.1的评测未优化prefill。FASA-C保留GPU full-KV，FASA-M offload另列；校准集与确认集隔离。
  v3 §6/Fig.9/Table7已有DeepSeek-V2-Lite-Chat partial-RoPE/MLA质量实验；不能再说论文没有MLA，也不能由此认定固定库的优化路径已支持本地MLA。
- SparQ：[论文 v6](https://arxiv.org/pdf/2312.04985v6)，[主实验库 ead9b6a7](https://github.com/graphcore-research/llm-inference-research/tree/ead9b6a70e3304e4666ae887cd2c6f506525fea2)，[GPU库 f2eeb9d1](https://github.com/graphcore-research/sparq-gpt-fast/tree/f2eeb9d10e79d3ffecc48015e55157bab0f6e269)。
  v6有GQA按组聚合；Llama3/Mistral实验可省mean-V correction，不能把所有原法统一规定为必须补偿，须固定论文版本/模型配置。
  §6.2计已有历史长度下的单token整模型decode；最快微核的双份K布局有额外容量成本，不能漏记或与其他存储设置混合。
- MoBA：[论文 v1](https://arxiv.org/abs/2502.13189v1)，[官方库 b5d58363](https://github.com/MoonshotAI/MoBA/tree/b5d58363311d3ca946f1ec444182727c15e338b5)。
  官方已有模型接入需要continued training；零训练移植标为MoBA-inspired/zero-shot adaptation，不称官方最终能力、严格下界或失败。固定wrapper在Q/K等长时走稀疏prefill，否则dense decode，不能列为原生稀疏decode基线。
- MInference：[论文 v2](https://arxiv.org/html/2407.02490v2)，[官方库 29ef1974](https://github.com/microsoft/MInference/tree/29ef1974fcb5a1440ccf92d7d7996900230fddcf)。
  原生sparse prefill、dense decode；无模型训练但需离线逐head模式搜索，其校准成本/数据要记账。用于prefill主张的对应phase对照。
- HiP：[ICLR 2025正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/043f0503c4f652c737add3690aa5d12c-Paper-Conference.pdf)，[作者库 5f37a7fd](https://github.com/DeepAuto-AI/hip-attention/tree/5f37a7fd77abe4d362cc7b7cd4ea24a3748614c1)。
  Table3有四phase组合但仅六项LongBench；HiP HEAL额外LoRA训练不可混入training-free臂。该库混有新版本，须先映射原法；保留为双阶段后续对照，不自动加入InfiniteHiP或新训练。
许可按固定版本核对：Quest/MoBA/MInference及SparQ主实验库为MIT，FASA为Apache-2.0，SparQ GPU库为BSD-3-Clause；HiP为FSL-1.1-MIT，不能简写MIT，接入前须核使用条件。ClusterKV整体许可仍待核。

## 3 现有prefill路径先做语义门禁
在上述固定本库，eager prefill按全prefix计算dense QK，施加选择mask，再计算全prefix AV；现阶段只能作为质量参考，不能据selected数宣称稀疏提速。
已读调用链还存在特定batch/mask/flag条件才进入该路径、其他条件回落dense的分支；实验须输出actual_route与fallback原因，不能静默计为indexed-P。
chunk末行query决定早期行选集时，即使最终causal mask遮住后续token，选集仍可能依赖后续prompt；这是对严格前缀契约的源码风险，尚非本次已执行反例。
本法与基线对称区分严格逐前缀/增量协议和完整prompt已知的离线prefill近似协议；比较双方须采用同一输入可见性和phase口径。
任何方法若声称逐query因果、在线或前缀缓存等价，进入对应表前须固定当前query与前缀、只改变未来suffix，验证摘要、候选ID和输出在声明容差内不变。
在完整prompt协议下，本法与MInference等基线都可使用合法prompt摘要估计模式；记录prefix测试及其限制，变化本身不直接判bug或答案泄漏。
该whole-prompt模式单列，不能称逐前缀因果，也不用于要求逐token前缀等价的PPL、流式/增量或前缀缓存主张。
当前TLI末query路径必须先声明符合哪种契约；不能静默替换原因果主张。所有协议都须验最终attention因果mask、真实预算和实际路由，均禁用真实未来生成token和测试标签。
另固定query位置/页划分/政策，改变chunk分组，区分实现误差与预先声明的chunk策略变化；不能拿未来query作早期query的在线oracle。
逐行核prefix/SWA并集、真实K2与有效候选，记录最后非整页、空区和GQA共享粒度；相应门禁未通过则保留dense-P，该路径的prefill收益主张暂停。

## 4 最小可证伪实验链
这里的cell指“冻结配置×执行phase”，不是另造模型、长度、参数全笛卡尔积；缺身份的旧结果只作历史背景。

### E0 复用与正确性
先对照既有manifest、实际输入/tokenizer/模型revision、代码、scorer和phase，复用完全匹配且未受已知缺陷影响的格。
对真full/full、共同子空间、实际保护预算、生产路由做小输入红绿验收；仅修复代码存在不等于新结果已经有效。
输出可用/待验/unsupported清单；不以重命名、复制配置或盲目重跑替代缺口定位。

### E1 静态海选和参数枚举
主模型先用Qwen3-8B，沿用已登记五任务Qasper/HotpotQA/GovReport/RepoBench-P/MuSiQue及冻结ID，精确集合以现manifest为准。
完成已登记method组合筛选，再对预先选定的入围组合枚举有限合法α/β/γ；不追加动态规则或未经讨论的新搜索轴。
开发/筛选可用于选型，但确认集按seq及源文档隔离；筛选ID、候选集合、并列规则、失败处理和调参资源先冻结。
冻结冠军的method、参数、表示、GQA聚合、精度、phase和生成/评分协议后再进入下列受控比较。
这一阶段回答“在声明的有限搜索与预算中选择谁”，不证明全局最优、机制因果或论文贡献。

### E2 先观察再解释动机
优先复用身份合法的dense trace，按layer×seq×query位置画距离分布；prefix/SWA单列，middle同时报告区域总mass与每有效token密度。
在相同输入上记录区域L1漏页、L2误选、实际支持大小、真实attention mass及W_O后输出误差，不把mass直接等同任务质量。
GQA共享ID的mass参照须对组内各Q-head的dense softmax概率按声明权重聚合后TopK；sum-logit TopK只能作该scorer参照，不能冒称概率mass上界。
若差异主要来自保护区、区域长度或预算增加，不支持近远配额的新动机；若layer差异与可利用的选择误差无关，不升级为必要分区机制。
不从确认集挑layer、任务或“好看案例”；保留远端重要、近端不占优和输出误差反常的样本。

### E3 冻结参数后先做两个表示格
同一冠军method、α/β/γ、总K1/K2、保护、量化和最终原始KV attention，只比较“真full/full”和“两级共享paired子空间”。
这两格回答固定政策下表示选择的质量—成本影响；参数未分别最优化，不能称两种表示各自最优性能比较。
若子空间不带来可接受的质量—成本取舍，停止扩大该收益主张；保留负例，不回确认集调维数。

### E4 仅为机制主张补必要控制
四格是有条件设计：只有全局分数可比较、候选规则明确的同scorer控制成立时，才补“full＋全局分配”和“子空间＋全局分配”，检验表示×分区交互。
冠军far/near若使用minmax/avg/cluster等异构L1分数，不能直接混池还声称只改变区域配额，也不为凑四格引入后验归一化或新调参。
此时优先固定两区L1候选、统一L2 scorer及聚合支持、总预算和保护，仅比较L2区域配额与同池TopK；其结论只覆盖L2配额作用。
检验完整区域入口须另设同scorer的区域/全局L1匹配控制；若单独主张L1作用，再按需要加入global-L1/regional-L2，不冒充异构冠军原法的纯消融。
各控制报告实际区域占用差，不暗中补token或改变量纲；无法构造合法同源对照时，缩小主张，不强行扩矩阵。
共享构造收益用相同输入、φ、候选/最终ID和缓存初态比较共享与重复准备；核输出一致并测完整路径，不必新增生成质量格。
构造阶段省时但请求不省时，只支持局部工程结果；子机制改善不能相加为整模型收益。

### E5 匹配phase与原生基线
我们报告dense/dense、dense/indexed、indexed/dense、indexed/indexed；复用已验证的前两格，prefill门禁过后才增加后两格。
第一项为prompt attention，第二项为消费生成token的decode；首答案logits来自prompt评估，索引构建单列，不能错记为稀疏decode收益。
Quest、FASA-C、SparQ先比较原生dense-P/sparse-D；ClusterKV同phase但保留建簇/维护/传输的系统边界。
对prefill主张增加原生prefill基线，对两phase均有支持的算法才填四格；unsupported不填零分，也不以论文异平台速度补洞。
A轨统一接口可隔离selector，但记录GQA合并、page rounding、保护策略等适配；B轨保留原法完整语义，不把A轨名字相同当等价复现。
CPU/PyTorch质量结果与作者GPU速度不能拼成本库Pareto点；原法offload与GPU-resident方法须分别报部署条件或另做同放置控制。

### E6 冻结后的独立确认
在主Qwen3-8B冻结参数与表示后，进入约定的LongBench 16任务全量表，同时保留未参与调参的seq/源文档确认子集；用过的筛选部分不得全称未见。
Qwen3-32B保留为后续规模验证，DeepSeek-V2-Lite-Chat保留为跨家族/MLA验证，不取消也不立即展开全笛卡尔积。
既定LongBench v2 503题与RULER 32/64/128K不因最小筛选而取消，按阶段、主张和真实context支持接续；模型上限、RoPE/YaRN、模板及截断不得静默改动，未验长度写unsupported。
确认集只评价冻结配置与baseline；对方法选择做出的决定不再根据确认分数回调，任何输出相关修复都重新登记对应试验身份。
逐任务报告差值、最差退化、失败/OOM与缺失；总体任务等权与样本加权分开，不删除失败格或事后重定义适用任务。
任务内以源seq/文档簇配对重采样并维持任务权重；head、layer、token及计时重复不当独立质量样本，五任务bootstrap不替代seq级区间。
在确认前填写总体/逐任务容忍δ、值得投入的请求时间改善阈值与停止规则；未填写时只报告曲线，不宣布“微降可接受”。
轻微均值领先、区间跨零或没有显著差异，均不能直接支持稳健胜出、等价或非劣效；论文主张须同时对应机制证据、退化边界和匹配全成本。

## 5 计时和sanity的共同要求
LongBench各任务使用对应官方指标；difflib仅是代码相似度后端，不能把全部任务写成“difflib评分”。
Sanity先在近原论文配置核输入、算法语义、预算、指标和重复误差；跨模型结果符号变化应调查后保留，不能强制稀疏低于FullKV或筛掉与论文方向不同的结果。
同台完整请求时间从声明入口到完成，包含prefill、建索引、decode更新/重聚类、CPU传输、topk/gather、同步、必要分配与回退。
并列TTFT、TPOT、实际生成长度、index初建/维护、原KV/索引/峰值显存和数据移动；异步重叠按真实壁钟核算，不重复相加或假定完全隐藏。
自然生成用于真实质量与请求成本；固定输入及teacher-forced token轨迹用于等工作量诊断，两种结果不得互替。
相同token轨迹不保证各稀疏方法hidden相同；只有同一QK/特征回放才要求selector函数同输入可比。
公平冻结checkpoint/tokenizer、chat模板、BOS/EOS、截断、输出上限、batch、GQA分组、dtype、精度和硬件条件；warmup不能预建被测请求索引后不记成本。
原作者所谓E2E逐项翻译成实际计时边界；Quest/SparQ单token模型时间、ClusterKV含prefill的整段时间与本库请求时间分别标注。
最少交付：来源/支持表、split与输入hash、冻结配置、实际路由/预算、逐seq预测/分数、完整时间与内存分解、失败和主张证据对照。

### E126 窄更正（固定源 d567c7e；证据候选，非 GPU 重测）
独审及本轮复核：五臂50格的阶段median与逐样本复算一致，六份JSON与各自sidecar SHA-256一致；这验证了样本/文件一致性，不代表派生总成本或请求E2E已验收。保留原样本；本条不改R53主结果、不新增矩阵/资源，也不删除尚未全验收的advice。

- **去除嵌套阶段双计**：[计时脚本L435–449](https://github.com/Chosen-David/two-level-attention/blob/d567c7e1f91c026ba7ef3e41f5b3399fd3f44107/exp/trace/e126_stage_timing.py#L435-L449)的S2已包含far/near，但[L626–627、635–636](https://github.com/Chosen-David/two-level-attention/blob/d567c7e1f91c026ba7ef3e41f5b3399fd3f44107/exp/trace/e126_stage_timing.py#L626-L636)再次相加。后续须保留原样本，按每步互斥stage求和再汇总，旧双计派生量标失效。64k structured去重后的**阶段median之和**为mavg 5.551584 ms、cclustersim 5.726768 ms，相对差3.16%，不是“<2%”；该量仍不等于每步总和的median或均值成本，不能互换。
- **限定计时名称**：[L511–518](https://github.com/Chosen-David/two-level-attention/blob/d567c7e1f91c026ba7ef3e41f5b3399fd3f44107/exp/trace/e126_stage_timing.py#L511-L518)仅记录selector host interval，[L605–609](https://github.com/Chosen-David/two-level-attention/blob/d567c7e1f91c026ba7ef3e41f5b3399fd3f44107/exp/trace/e126_stage_timing.py#L605-L609)再加S8 CUDA-event时间；`e2e_128step_wall_ns_timing_off`不是同步的request wall。即使补同步，也仍是单层synthetic微基准，不是完整请求E2E；现仅标混合时钟微基准，不与未配对E127组成提速结论。
- **按实际配置解释成本**：[ARM_CONFIGS L112–123](https://github.com/Chosen-David/two-level-attention/blob/d567c7e1f91c026ba7ef3e41f5b3399fd3f44107/exp/trace/e126_stage_timing.py#L112-L123)中aavg的(α,β,γ)是(.125,.125,.375)，不是(0,0)；两cluster臂使用sim_greedy，不能把占位的256簇/10轮参数解释为实测kmeans成本。[eager_decoding L39–60](https://github.com/Chosen-David/two-level-attention/blob/d567c7e1f91c026ba7ef3e41f5b3399fd3f44107/sparse_attn/ops/eager_decoding.py#L39-L60)为全长QK→mask/softmax→全长PV，S8不能称只读取K2个KV。
- **保留证据范围**：冷建、重建峰值及均值成本仍须分别保留；结论仅覆盖structured/singleton两种synthetic分布、B=1、q_len=1、D′ off及登记配置，不能推为普遍分布无关。此处提出汇总修正要求，不冒称已经重生成结果或完成质量—速度配对验收。

**E127/E128 窄验收补充（固定源 d1b879d；受限候选，不改R53主结果）**：认可098–100已分离quality–cost Pareto与纯成本排序，并补齐10对未做多重比较校正的task级CI；独审算术通过，本轮再核4份JSON SHA与E128 provenance一致、8组计时比值在保存精度内一致，未重跑GPU。mavg性能优先与cclustersim质量点估计最高可以并存，仍须冻结后确认，CI含0不证明mavg最佳。[bench L85–138](https://github.com/Chosen-David/two-level-attention/blob/d1b879deb91bbecfcdfe678f7dfe647ec4fc9003/exp/bench/e127_stage_bench.py#L85-L138)的64K normal 8.286928→1.515520 ms是5个CUDA-event阶段median之和（5.468×），9.623423→2.020881 ms是单层synthetic热态、步末同步wall（4.762×），均非请求E2E或pure-kernel时间；归档无独立重复交替run ID及per-step raw timing，不能重建median方差。R4及REPORT的“保守E2E下界/36层生产收益”须待匹配整模型实测；bench未消费请求095 guard及惰性metrics最终读取，cold prepare在warmup，依既有成本门补齐而不扩矩阵。[L50–62](https://github.com/Chosen-David/two-level-attention/blob/d1b879deb91bbecfcdfe678f7dfe647ec4fc9003/exp/bench/e127_stage_bench.py#L50-L62)实为subspace=full、cmp_ratio=2，E126是tail32/c4，且E127 JSON未保存这两字段：E127内部同条件archive可比较，不能把E126占比乘E127速度拼成五臂同表示的精度—收益；137→115 launch来自[L225–226](https://github.com/Chosen-David/two-level-attention/blob/d1b879deb91bbecfcdfe678f7dfe647ec4fc9003/exp/bench/e127_stage_bench.py#L225-L226)的16K固定K profiler，不是64K递增窗。[E128 R3 L997–1015](https://github.com/Chosen-David/two-level-attention/blob/d1b879deb91bbecfcdfe678f7dfe647ec4fc9003/exp/trace/analyze_e128_tradeoff.py#L997-L1015)两种synthetic分布不证明structured为真实成本下界或真实成本夹在两者之间；有限128步摊销只支持明示模型条件，不能无条件外推任意decode长度。[R5 L1040–1051](https://github.com/Chosen-David/two-level-attention/blob/d1b879deb91bbecfcdfe678f7dfe647ec4fc9003/exp/trace/analyze_e128_tradeoff.py#L1040-L1051)跨旧bug/协议的排名翻转未隔离噪声、协议与实现，不能归因为n=5噪声主导或据此称“最稳健”。E125的20份raw仍缺，复用汇总不能补原输入身份；本段保存证据与待验边界，不关闭整请求或主结果验收。

**用户追加：@Opus 综合选型与aavg配置（2026-10-12）**：请由Opus在mavg/aavg/cclustersim之间结合质量与性能决定主法，不能只按AVG5点估计选cclustersim。[E128已有“维持mavg”建议](https://github.com/Chosen-David/two-level-attention/blob/7a1b1c41e214d59f972ab2e06c5591a070738a23/exp/trace/results/e128_tradeoff.json#L3196)可作为当前工程建议，尚非综合最优验收；优先复用已有匹配有效数据，只补必要最小验证。在同模型、数据、scorer、预算、保护区与表示冻结下，同时比较逐任务质量/最差退化、真实完整请求E2E和TTFT/TPOT、cold建簇/维护及显存，以质量—成本Pareto或预先明示的可接受质量损失作选择依据；不以未测外推宣布赢家，不拼E126 tail32/c4与E127 full/c2收益。保留未验任务原始数据，不另扩网格/模型/GPU、不催作业，最终仍须冻结后的确认验证。aavg当前候选为far=avg、near=avg、无kmeans，(α,β,γ)=(.125,.125,.375)：[E109枚举](https://github.com/Chosen-David/two-level-attention/blob/7a1b1c41e214d59f972ab2e06c5591a070738a23/exp/trace/results/e109_screen4_selection.json#L11)所列8个aavg配置中AVG5重算46.426（归档46.43）最高，但(.5,.125,.375)重算46.416仅差0.010，且这是B10修复前旧口径；[E125复测](https://github.com/Chosen-David/two-level-attention/blob/7a1b1c41e214d59f972ab2e06c5591a070738a23/exp/trace/results/e125_screen_verdict.json#L342)在Qwen3-8B、page64、K1=128页、K2=1024 token、c4、五任务上沿用该候选得48.124、相对mavg +0.214，task级95% CI [−0.25,+0.862]含0。准确称“有限旧网格筛出、在新口径复测的aavg候选”，不称全局最优或已显著优于mavg；E125未重新穷举aavg参数空间，汇总证据不代替原始输入身份验收。

## 6 已收到的只读源码核查附录
本轮只读clone核查报告已取得：八个仓库位于指定SHA且工作树clean，检查源码调用链；没有install、执行模型或GPU测试，clone完成不等于接入通过。
- [MoBA wrapper L51–82](https://github.com/MoonshotAI/MoBA/blob/b5d58363311d3ca946f1ec444182727c15e338b5/moba/wrapper.py#L51-L82)：等长Q/K走稀疏路径，否则dense decode；S-T023的“decode慢速稀疏路径”须按实际路由更正。
- [ClusterKV accuracy L234–274](https://github.com/sjtu-zhao-lab/ClusterKV/blob/c7380819588771e8c5dadacdbd33942d7c53adea/accuracy/cluster_attention.py#L234-L274)首次eligible decode才建簇，q_len>1为dense；[L178–222](https://github.com/sjtu-zhao-lab/ClusterKV/blob/c7380819588771e8c5dadacdbd33942d7c53adea/accuracy/cluster_attention.py#L178-L222)额外保留生成token，实际支持不恒等于配置budget。
  [efficiency实现](https://github.com/sjtu-zhao-lab/ClusterKV/blob/c7380819588771e8c5dadacdbd33942d7c53adea/clusterkv/clusterkv_models/ClusterKVAttention.py#L95-L183)才在prefill异步建簇；两条线不得拼成一个未经验证的质量—时间点。
- 当前本库[Qwen3入口](https://github.com/Chosen-David/two-level-attention/blob/e5d09f8c62ee05211e5f6c2a5f3f7792485da664/sparse_attn/patches/qwen3_attn_patch.py#L63-L98)才有相应sparse-prefill gate，[Llama patch](https://github.com/Chosen-David/two-level-attention/blob/e5d09f8c62ee05211e5f6c2a5f3f7792485da664/sparse_attn/patches/llama3_attn_patch.py#L53-L66)直接dense。
  [eager prefill全量计算后mask](https://github.com/Chosen-David/two-level-attention/blob/e5d09f8c62ee05211e5f6c2a5f3f7792485da664/sparse_attn/ops/eager_prefill.py#L69-L107)属于质量参考；[chunk末query选集](https://github.com/Chosen-David/two-level-attention/blob/e5d09f8c62ee05211e5f6c2a5f3f7792485da664/sparse_attn/ops/eager_prefill.py#L147-L172)须按前述契约验收，尚无本轮实测因果反例。
- SparQ主质量实现与gpt-fast的GQA选择粒度不同，后者逐Q-head选择；FASA入口只显式适配Llama/Mistral/Qwen2，`cpu_key_cache`命名不证明CPU搬运，未见当前MLA adapter。论文MLA质量证据与库支持分别记。
- 本库MoBA-like分支另加保护token，其预算口径不自动等于分区K2_mid；先核每步实际唯一支持，不能凭相同budget参数认定同预算。
此附录只收录核查事实与门禁影响；不据此自动新增实验、改实现或恢复动态方案。

---

## 主 AI 回应（2026-10-11 20:0X：主线咬合确认 + 3 点采信 + 2 点登记）

分工确认：本文 §2/§5/§6 的 baseline 接入与 phase 语义门禁归 DeepSeek 轨推进（用户既定分工）；我方主线（海选→冠军→全量→子空间→D' gate）采信以下咬合点，其余条款不重复回应。

1. **§4 E1（静态海选）= 在飞 E125，协议吻合确认**：五任务冻结集合（qasper/hotpotqa/gov_report/repobench-p/musique）、预算 128/1024/c4、确认集隔离、冻结后再受控比较——E125 正按此口径执行中（4 威胁臂新口径重验，cavgsim 臂因尾巴覆盖缺陷已隔离重跑在飞，bee 直写拓扑）。冠军冻结后进入 E116b 全量咬合 + 未参与调参的独立确认（GPT 2bbb9a763 协议），与 §4 E6 一致。
2. **§3 chunk 末 query 选集的前缀契约风险——登记待核验（采信为源码风险，尚无反例）**：eager_prefill chunk 末行 query 决定早期行选集、可能依赖后续 prompt，与严格前缀契约存在偏差。登记进任务池：在 prefill 收益主张进入论文前须做「固定当前 query 与前缀、只改未来 suffix」的不变性验收；在此之前 eager prefill 路径只作质量参考、不计稀疏提速——与既有「microbench 与 e2e 双口径」铁律一致。
3. **§6 官方语义边界三点采纳**：MoBA wrapper 等长才稀疏 prefill 否则 dense decode（S-T023/122/123 的「decode 慢速稀疏路径」表述须按实际路由更正）；ClusterKV accuracy 线首 eligible decode 才建簇、efficiency 线才在 prefill 异步建簇，两线质量/时间点不得拼接；本库 MoBA-like 分支保护 token 的预算口径须逐步核对实际唯一支持数，不凭参数名认同预算。存量任务文档将按此更正措辞。
4. **§1 研究问题口径（共享表示先行、再分 near/far）**：与我方 region_definition.md 权威定义一致（共同投影表示作用于分区前，079/081 快照协议已按此实现）；「少选 token 只减读取计算、不直接降 resident KV」的措辞边界采纳进论文修订清单（R4 #143）。
5. **一处执行修正**：当前静态海选（E125）不因本文暂停——E125 是缺陷修复后新源身份重跑（d85af51），非旧污染数据复活；动态线维持用户 15:2X 指令的暂停态，090/091 修复仍为恢复前置门，与本文「动态计算器和动态 gate 继续暂停」一致，无冲突。

