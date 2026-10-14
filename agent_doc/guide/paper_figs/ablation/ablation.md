# 论文消融实验与面板对照

目标是说明各设计在什么条件下有用、何时失效，而不是预设“冠军最好”“质量损失在噪声内”或“某层可以安全丢弃”。本版把原四组图要求保留并拆成六个稳定实验 ID；它们可以合并成有限主文面板，完整曲线放附录。

版本基线：研究仓 `542234ac199cea9617c45d170c3329ab16a20687`，2026-10-14。所有格先服从 [evaluation 的 E0 冻结与产物契约](../evaluation/evaluation.md)。本版不执行新实验、不恢复动态 α/β/γ 搜索。样本ID、δ/η、种子/重复数、精确模型revision及尚缺实现的臂仍为 `UNFROZEN/BLOCKED`，负责人补齐后再核验；既有身份一致的结果直接复用。

## 公共设置与优先级

- 原方法固定 `best_mavg(.25,.125,.625)`、page64、L1上限128页、K2总上限1024、sink128+SWA128，最终原始全维attention；保护包含在最终总预算中。所有消融记录去重后的实际支持、短序列clamp、区域不足、fallback及dense层。
- 主消融先使用Qwen3-8B、dense-prefill/sparse-decode、冻结五任务及未参与调参的确认样本；具体ID由E0登记，不能把“已跑过五任务”当独立确认。原主质量表完整任务要求仍按E1接续，不将每格扩成所有模型×所有长度。
- P0为主贡献必要的语义与因果控制；P1为机制解释和已要求敏感性；P2为仅在相应主张成立时需要的附录/工程诊断。优先级不是新增派单权限，也不取消原任务。
- 所有质量格采 `predictions/scores`，所有成本格采完整selector与request wall；指标单位、置信区间、原始路径/哈希与停止规则沿E0。图注必须区分自然生成质量、固定轨迹回放和相同Q/K的函数对拍。

## A1 方法选择和参数可迁移性

**原图对应：Fig1a/1b。优先级：P1，主文简述选型，完整结果放附录。**

**问题与必要性：**为何选择mavg与这组固定参数？它是否仅适合被用于选型的任务？这是选型说明，不承担near/far或子空间的因果证明。

**实验臂：**复用已登记海选各method的合法最佳开发配置及固定冠军；不新增方法/参数轴。参数通用性比较固定全局冠军与预先登记候选，确认集上不再挑赢家。若画“每任务最好”，仅使用开发集得出的配置在该任务确认集评价；直接在确认集取最大值的点必须标“事后oracle”，不得作为可部署臂。

**固定变量与门禁：**相同开发/确认划分、总预算、保护、表示、dtype、phase、模型/scorer；记录不同method的初建簇与维护成本。旧口径和新口径数据不拼同一排名；候选集合/搜索资源/失败处理和winner规则先冻结。若历史raw只够证明有限候选点估计，则只写该范围。

**采集字段：**`method_id, config_hash, split_hash, task_id, sample_id, score, delta_to_champion, init_ms, selector_wall_ms, request_wall_ms, peak_gib, search_budget`，以及有效/失败样本数。

**图形：**(a) x=候选方法，y=确认质量差值，误差线=配对95%CI，旁列完整成本；(b) 行=任务，列=固定参数配置，色值=相对同任务固定冠军的百分点差。完整热图放附录；不得每行使用不同色标掩盖退化。

**支持/否定与停止：**支持有限搜索范围内可解释的质量—成本选择；不能证明全局最优。参数在确认任务中出现明确退化时如实保留，缩小“通用”主张，不追加确认集调参。

**产物：**`figure_id=A1`，每格绑定E125/E128等实际原run_id或新run_id；历史报告只作索引，确认其预测、评分和配置身份后才能复用。

## A2 两级表示与频段维度

**原图对应：Fig2。优先级：P0表示四格；P1频段/宽度。**

**问题与必要性：**L1和L2降维各自影响什么？tail32的效果来自低频选择、维数减少、RoPE配对完整性，还是实现路径变化？

### A2a 四表示格

| 臂 | L1评分维度 | L2评分维度 | 显式配置意图 |
|---|---:|---:|---|
| FF | 128 | 128 | `tli_subspace=full` 且 `tli_l2_subspace=full` |
| FS | 128 | 32 | `tli_subspace=full`，L2使用原tail32回退 |
| SF | 32 | 128 | `tli_subspace=tail` 且 `tli_l2_subspace=full` |
| SS | 32 | 32 | `tli_subspace=tail`，L2同一完整pair集合 |

这四格都保留稀疏选择与最终128维Q/K/V，dense FullKV另列。逐臂核实际坐标、K摘要/QDQ形状与dtype，不能只相信`full`名称。固定α/β/γ、K1/K2、保护、GQA聚合、原始`1/sqrt(128)` scale、量化算法及最终attention。量化步长可因所含坐标改变，这是表示干预的一部分，必须记录；不能额外换精度/后端来美化某一格。

### A2b 频段和宽度

- 先固定32个标量坐标、16完整RoPE pairs，比较tail、highfreq（高频）及paired-random。Q/K使用同一pair集合，L1/L2按该集合构建各自摘要和QDQ；每层/KV-group共享规则不变。
- 当前 [random实现 L514–517](https://github.com/Chosen-David/two-level-attention/blob/542234ac199cea9617c45d170c3329ab16a20687/sparse_attn/indexer/tli_indexer.py#L498-L527) 是固定seed20261002的`randperm(128)[:32]`，不保证pair完整，且不随`cmp_ratio`改变宽度。它只能标为“unpaired-random32诊断”，不能作为纯频段对照或声称完成random16/64。合格paired-random实现与坐标回执未验收前，该臂保持BLOCKED；本指南不授权修改代码。
- 随机pair集合的种子列表在确认前固定，报告跨集合变化；不挑最好seed。集合数量由资源与精度目标预登记，至少不能把单次幸运集合称为随机方案稳健性。
- 宽度仅沿tail做16/32/64；SS32复用A2a，FF作为全维参考。优先使用嵌套完整pair集合；高频/random不再乘上所有宽度，除非结果确实需要检验交互。
- 不用DCT重新定义RoPE频率；D128下pair坐标关系和频率排序以实际模型RoPE实现为准。换模型重新核对，不照搬索引区间。

**采集字段与统计：**每样本质量及相对FF/FS差值；实际坐标ID/哈希、标量维数、pair数、quant范围/dtype、prepare/update/L1/L2/TopK/gather/attention完整时间、峰值内存。辅助记录最终token Jaccard、相对共同参考的retained mass和`W_O`后输出误差；逐位mask一致率受大量共同未选位置影响，不能换成Jaccard或recall。采用E0配对统计，随机集合方差与样本方差分别报告。

**图形：**主文一个2×2表示格质量差值/成本面板，加一个固定32维频段比较面板；tail宽度x={16,32,64,128}、y=质量/完整成本的曲线放附录或与主文Pareto合并。两种单位分面板，不用双轴制造相关性。

**支持/否定与停止：**仅在质量达到预登记δ、完整成本达到η时支持该范围的取舍。128→32表示点积宽度降4倍，不自动等于完整selector快4倍。若tail不优于paired-random/highfreq，保留负例，不能再以FASA为固定tail32提供结论性背书。

**产物：**`figure_id=A2a/A2b`；每个维度格必须绑定实际坐标文件而非参数名。E133/E134三格只能按原实际配置复用，缺FF或新随机臂不得补成已完成四格。

## A3 Near far配额与两级选择的必要性

**新增补足原near/far主贡献。优先级P0，主文核心。**

**问题：**收益来自区域配额，还是near/far不同scorer、保护区或候选数变化？L1缩小选择支持是否也真的缩小了计算量？

**最小实验臂：**

1. 固定同一条Q/K回放与同一L1候选、保护区、L2概率、总K2，比较“区域配额TopK”和“候选全局TopK”。L2逐Q softmax的支持集合相同，GQA组内概率均值相同；只干预选取配额。这是L2配额效应，不是完整两级分区效应。固定Q/K回放只验IDs、局部误差与选择成本；任务质量另用同一冻结干预做自然生成，不能把回放指标称为任务精度；自然生成的隐藏状态与实际候选可能分化，届时仅称政策与预算固定，不继续宣称逐步Q/K或候选相同。
2. 若主张完整L1区域设计必要，再预先选择同一L1 scorer（如全区域minmax或全区域avg），比较global-L1与regional-L1，先固定L2政策。当前冠军far-minmax/near-avg不能直接混池并声称只关闭分区，也不为对照临时引入后验分数归一化。
3. 如单独主张L1筛选价值，可在同一L2 scorer和总K2下加“无L1限制”的单级臂；记录额外候选与实际成本。这一臂不与1的“同候选纯配额效应”混为同一个问题。

**固定与冻结：**先固定所检验的层级、scorer、K1/K2、保护、同候选/全候选定义、tie规则和短区处理。语义对拍完成前不跑大确认集。当前 [L2 L1412–1481](https://github.com/Chosen-David/two-level-attention/blob/542234ac199cea9617c45d170c3329ab16a20687/sparse_attn/indexer/tli_indexer.py#L1412-L1481) 通常先算全S再mask；因此减少L1候选只能直接说明选择支持变化，不能按候选比例宣布L2点积省算。

**采集字段：**逐层实际candidate/page/token IDs与unique counts，near/far/protected占用、L1漏掉的参考重要token、L2最终支持、mass、输出误差；逐样本质量和含prepare/TopK/gather的完整时间。不同阶段scorer比较时，明确参考是概率mass还是该scorer的分数上界。

**图形：**主文x=区域/全局控制，y=配对质量差值或请求成本；另小面板显示实际near/far token占用，避免只报名义K。漏页/预算扫参完整曲线放附录。

**支持/否定与停止：**只有同scorer/同预算的合法干预才能归因到配额；候选、归一化支持或评分来源一起变时降级为整体方法比较。若全局控制同等或更好，不宣称区域划分必要；可以报告其在特定任务上的效果。不能用仅mass改善代替任务质量。

**产物：**`figure_id=A3`，保存每个控制臂的语义差异清单。候选内L2优化仅当对应工程主张需要时，在A6附录控制中验证，不自动增加新主算法格。

## A4 Gate与near回填

**原图对应：Fig3。优先级P0语义，P1候选质量/成本；主文是否使用由证据决定。**

**问题：**关闭far的实际效果是否来自把预算给near？prefill校准能否预测后续decode风险，其成本是否值得？

| 臂 | far策略 | near L2 | 解释 |
|---|---|---|---|
| G0 | far-on | 原预算 | 固定策略参照 |
| G1 | far-off + refill | 接收原far余量，受可用候选限制 | 当前回填主线，检验远端选择换为近端选择 |
| G2 | far-off，无refill | 保留G0的near额度 | 机制诊断，实际总token通常更少，不能称同预算质量比较 |

**实施顺序：**先在固定Q/K与预先冻结的同一skip决策轨迹下对拍G0/G1/G2，分离策略作用；G2只有在明确实现/实际ID验收后才可用，不能从现有开关名推定已存在。随后单列完整策略“gate功能关闭”与“prefill校准gate开启”，保留各自自然生成质量和完整请求成本。固定Q/K三臂只产局部诊断，自然生成三臂/完整策略的状态与实际候选允许分化，图中分轨命名。手工/静态层集不能冒称动态策略，静态/动态gate组合按代码互斥。

**固定变量：**best_mavg、表示、保护、L1 near页额度、阈值/校准行数和输入可见范围、实际边界取整规则、随机种子。当前far-off不扩near L1，只让L2剩余额度进入near；不能把G1的效果解释成只删far而其他都不变。

**校准与风险记录：**逐seq/layer记录校准mass、样本行位置、阈值、pending/frozen状态、首/后续decode实际mask与clear；记录问题token是否已在校准输入。分别保存校准定义的far区与真正跳过ID集合，不能假定取整后边界完全一致。报告后续decode的far mass、head-mean/head-tail、触发率及错误跳过率，阈值只在开发集选。

**成本与采集字段：**在E0字段外加入`gate_enabled, skip_far, refill, calibration_ms, calibration_peak_bytes, host_sync_ms, far_lo/hi, actual_K1_near/far, actual_K2_near/far/protected`。校准是full-D逐Q softmax，且float标量读取会同步；当前far-off仍有全域prepare、score buffer及repeat/float转换。测完整selector、TTFT/TPOT/request wall，不按省下的far FLOPs推断净省时。

**统计和图形：**主质量图x=G0/G1/G2或完整策略，y=相对G0的逐任务百分点差，配对CI；并列实际预算/触发率。成本另面板报绝对时间与包括校准的请求差；逐层mass全景引用O2，不只选低mass层。校准触发的“样本×层”比例不写成固定可跳层比例。

**支持/否定与停止：**CI含0不证明无损；平均mass不证明每个Q-head或未来decode安全。达到预登记非劣效δ且完整成本改善η才支持该策略；任务退化或校准抵消收益时保留负结果，gate继续作候选/限制，不回确认集挑层。Llama当前不因Qwen入口存在就视为支持动态gate。

**产物：**`figure_id=A4`，绑定触发轨迹/阈值/有效mask哈希。E129 cclustersim静态gate结果不能代替mavg动态gate；只交校准JSON不能代替任务质量。

## A5 Middle与保护区归因

**原图对应：Fig4。优先级P0，主文核心。**

**问题：**完整方法的收益来自多看一些token、固定保护，还是middle中的内容选择？

**最小臂：**

- M0：FullKV质量锚点。
- M1：sink128+SWA128-only，作为删去middle的诊断；标准长序列实际只有256 tokens。
- M2：相同保护区+本方法的middle选择，最终总预算1024。
- M3：相同保护区+预先确定的朴素middle策略，middle同为768，例如最近768个非保护token；也可预先选择随机middle作替代，但不能看确认结果后挑更弱者。若用随机，固定种子并报告变化。

M1与M2不等预算，只能说明新增middle支持的整体价值；M2与M3才检验内容选择超过简单补足相同token数的作用。FullKV不是这个等预算因果控制。

**固定与门禁：**M2与M3固定相同总unique K；M1保留256诊断、M0保留FullKV，不补足成等K。其余比较固定保护区定义、最终全维attention、模型/输入/phase/GQA共享口径；明确M3是在完整可见middle上选择还是同候选集合内选择，二者回答不同问题。质量对照不要求选择成本相同，成本差必须实测并保留。保护区重叠/短序列按实际ID去重；M1若indexer被旁路或仍照算评分，须在时延解释中说明。

**保护区敏感性：**原sink/SWA网格作为P1附录，不用来重新挑确认集参数。当前生产保护长度要求page64对齐，既有trace中的sink16不等于生产支持；需要固定总K并把保护增减反映到middle余量。若单独主张保护机制必要，才补“保护开/关、总K固定”的合法对照；不能假定SWA=0已被当前入口支持。

**采集与图形：**逐任务score/paired CI、实际unique K、各区域mass/输出误差、选择和请求成本。主文x=M1/M2/M3，y=任务质量差；旁注明确256与1024，M0为参考线。附录保护网格x=sink、y=SWA、色值=同总K下质量或成本，不将2048 trace mass与1024任务分数混用。

**支持/否定与停止：**M2仅胜M1不足以证明智能middle选择；若M3达到相同质量且更快，应保留该反例并缩小索引贡献。mass差不能直接当任务质量差。

**产物：**`figure_id=A5`，记录朴素策略完整定义/种子、候选来源与实际ID。E135旧产物按其2048 trace或256-only协议独立标注。

## A6 阶段组合与工程成本边界

**承接原Fig2的prefill/decode追问和各模块单独成本。优先级P0路由核验，P1阶段结果，P2候选内工程优化。**

**问题与臂：**dense-P/dense-D、dense-P/sparse-D、sparse-P/dense-D、sparse-P/sparse-D。P指prompt attention，D指后续单token消费路径；建索引和gate标定单列。先选一个已登记代表长度和固定样本子集核路由/质量，再决定是否补既定长度；不与频段/维数/gate做全笛卡尔积。

**当前路径门禁：**main默认dense prefill；`TLI_SPARSE_PREFILL=1`仅支持其声明的B=1无padding路径，当前eager实现仍计算全QK/PV后mask。chunk末Q决定共享选集，对更早行的严格前缀不变性需要“固定当前query/前缀，只改未来suffix”的独立检查；行级causal mask不自动证明选择前缀等价。未通过时只能明确标注whole-prompt近似质量协议，不能称严格在线前缀等价或高效稀疏内核。其他分支的gather/prefetch实现需自己的SHA、非默认保护区与边界验收。

**成本固定量：**同输入/输出工作量、batch、模型/backend、原始full-KV、保护、预算及正确路由。记录P/D实际route和fallback，不按flag标签填表。按E2计入index初建/更新、校准、select/gather、transfer、同步和sampling；必要时同时给“相同实际支持上的最终attention”诊断以分离selector成本。

**候选内L2附录控制：**仅当要声称L1减少L2点积工作量时才需要。当前全S评分后mask与候选内计算必须在相同Q/K、相同L1mask、相同量化/dtype、softmax支持、GQA概率聚合和tie规则下对拍有效候选、最终IDs与输出；覆盖短序列、部分页、候选不足及NaN。语义通过后测候选构造/gather/launch/完整selector，不只测dot kernel。实现未存在则保持BLOCKED，不以复杂度图代替结果。

**共享构造附录控制：**若声称物理缓存复用收益，固定坐标、IDs、缓存初态和数值语义，比较当前实现与明确的重复准备实现。共用坐标不等于L1摘要与L2 QDQ共用同一物理缓存；无实现和实测则仅作机制说明，不新增主质量格。

**采集/统计/图形：**E0+E2字段，另存P/D路由与前缀测试原件。四阶段质量差以小表呈现，成本用绝对ms面板；有负收益也画出。候选内/共享构造对拍及微基准放附录。`figure_id=A6`。

**停止条件：**路由错误、前缀承诺不成立、数值/tie门禁不通过时阻塞对应主张；实现优化只快于旧慢路径而不快于高效dense时，按实现改进报告，不写vs-dense加速。新图规划中的E139 wall数字在原件与身份链未交付前不进入实测图。

## 出图组合与交付检查

1. 主文优先组合A2的表示/频段、A3配额、A5等预算middle四个紧凑面板；图中不要堆全部参数网格。
2. A1完整海选、tail宽度、保护敏感性、阶段四臂和工程对拍放附录。A4只有在独立质量—成本证据齐全且作为主贡献时进入主文，不为画一张漂亮gate图提升未验主张。
3. 每图交付可编辑脚本、`figure_data.tsv`、源run_id清单、caption和独立检查结果；字体/颜色/坐标一致，同类对照固定顺序，负例不裁掉，单位不混。具体总图数与evaluation中的主文预算一致。
4. 图注逐项回答：干预了什么、哪些固定、N是什么、误差线是什么、是否包括保护/初建/维护、哪些是作者报告或未验。缺一项时先补元数据，不靠caption猜测。

## 一手比较的用途

[Quest正式PMLR版](https://proceedings.mlr.press/v235/tang24l.html) §4.2.3/Fig.7与§4.3.1–2/Figs.8–10提示分开预算消融、kernel及模型计时；单token计时不是本项目请求E2E。[SparQ正式PMLR版](https://proceedings.mlr.press/v235/ribar24a.html) §5.4/Fig.7的组件消融与§4 GQA讨论提示保留补偿/布局控制；[ClusterKV v2](https://arxiv.org/abs/2412.03213v2) §V-C/Fig.12–13提示建簇和传输不能被隐藏。借鉴的是受控比较与成本边界，不搬用他人的数值或预设本方法必须赢。
