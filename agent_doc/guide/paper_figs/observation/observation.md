# 论文 Observation 与动机图执行对照

本文件回答三个动机问题：
注意力随距离怎样分布，
哪些层/头可能不依赖远端，
以及32维完整 RoPE 对是否能保留检索信息。
它规定怎样得到可审查的观察，不预设“低频一定最好”“低 far mass 可以安全删掉”或“分区一定有效”。因果与任务收益分别由 [消融 A2–A5](../ablation/ablation.md)验证；模型、数据、统计、原始产物和成本身份统一遵守 [评测 E0](../evaluation/evaluation.md)。



## O0 采样与共用口径

- 开发/标定样本和确认样本按 source document/seq 分离。先登记任务、样本 ID、query 位置、模型与 RoPE 配置，再看图；不能挑一张符合故事的 heatmap 充当总体。
- 主要观察使用相同 checkpoint、输入轨迹和固定 Q/K 的 dense/full-attention 参考。prefill query 与 decode query 分层报告，teacher-forced 固定轨迹与自然生成轨迹分开；token 相同不代表不同方法的隐藏状态相同。
- 保留每个 Q-head 的原始量，再给同 KV-group 聚合。不能把组均值正常推成组内所有头安全；head/layer/query 是重复观测，统计独立单位仍为 document/seq。
- sink、SWA、near、far 必须是明确的 token ID 集。先去除保护区再划非保护 near/far，记录边界、完整/部分页、有效 causal 支持和实际长度。用固定方法的边界观察，不从同一确认图重新选择 α 或断点。
- 质量参考概率来自原始128维 Q/K、正确 scale 与完整合法支持；32维 selector softmax 是另一分布，不能混用来证明“保留了 full-attention mass”。GQA 的 full-reference 先逐 Q-head 求概率，再在声明的组内聚合。
- 原始记录沿用 E0，至少补充 `query_phase/query_position/layer/q_head/kv_head/region_start_end/n_valid_tokens/actual_ids_hash/reference_kind`。mass 无量纲、距离 token、频段坐标/维度为整数，误差需注明相对分母与零分母处理。

## O1 距离分布、保护区与 near/far 动机

**稳定 ID：O1；优先级 P0；主文候选一个紧凑面板。问题：**在保护区以外，注意力质量是否既有局部集中又存在不可忽略的远端贡献；现有 near/far 边界是否具有可解释的资源分配意义？

### 实验与固定量

1. 对预登记的 full-attention trace，计算每个合法 key 到 query 的距离 `d=q_pos−k_pos`。保护 sink/SWA 单独记账；不要把 sink 高质量混成近距离，也不要重复统计两保护区的交集。
2. 同时计算原始距离直方图、非保护 near/far 总 mass 与该区 token 数。距离箱边界在开发集冻结，报告每箱总 mass 和每 token mass density，避免较宽的远端箱仅因 token 多而显得重要。
3. 画当前固定 α=.25 所对应的实际边界；其语义按当前代码的有效长度/页取整计算，不能直接把图的横轴25%当成实现边界。短序列和区域不足单独标注。按有效长度和query位置分层展示；混合trace若共用绝对距离轴，边界画范围/分位带而非一条常数竖线，不能只标“α=.25”掩盖变化。
4. “高权重 token”如需展示，只采用预登记定义，例如相同个数的 full-reference TopK，或达到预定累计质量的最小集合；定义和阈值均需负责人冻结，不使用看结果后挑出的阈值。

固定模型/RoPE、样本/query、full-reference、距离箱、聚合权重；本图不改变 near/far selector，不是参数海选。不同任务与长度先分层，再给明确权重的汇总；不能把所有长序列的 token 一起平均而掩盖任务差异。

### 数据、统计与出图

- 采集 `mass_sink/mass_swa/mass_overlap/mass_near/mass_far`、各区唯一 token 数、每箱 mass/density、full-reference TopK 的区域分布及采样清单。去重后各区总和应在数值容差内为1；仅画非保护分布时注明其是否重新归一化，并同时给非保护总 mass。
- 按 document/seq 聚类配对重采样，画均值或中位数及95%区间；另给远端质量的上分位/最差头摘要，具体分位预登记。不得用数百万 head-token 点构造虚假的极窄区间。
- 主面板：横轴 key-query 距离（tokens，若用 log 轴明确零距离处理），纵轴每 token mass density；标注保护区和已冻结的 near/far 边界。小 inset 或附录：各区总 mass 与 token 比例。不同图中统一颜色，不用断轴夸大 far。
- 原件用 `figure_id=O1`，每一距离箱可回查 trace、query 以及分箱规则；完整 head/layer 热图放附录，不要求全部进主文。

### 可支持、不可支持与停止条件

该图可以支持“局部与远端检索需求不同”的动机；不能仅凭分布证明当前 α、β、γ 最优，不能证明 far 应关闭，也不能证明两级选择优于全局 TopK。若非保护 far 很弱、不同任务无稳定分界或密度差异不明显，保留结果并缩小动机，不重新筛样本制造断点。near/far 配额收益由同 scorer、同总预算的 A3 判断。

## O2 层/头远端依赖与 gate 候选

**稳定 ID：O2；优先级 P1；gate 进入主贡献后才升为主文必要。问题：**远端依赖是否具有可重复的层级差异，prefill 标定是否能预测后续 decode 的风险？

### 实验与固定量

1. 先在不关闭任何 far 的固定 full-attention 轨迹上，覆盖预登记的所有目标层和 Q-head，测每 query 的 far mass。保留 raw Q-head、KV-group 均值和层均值；不先选“可关层”再画热图。
2. 按当前 gate 的真实时间线分开记录：prefill 末 `min(rows,S)` 行标定、pending 状态、首个进入 decode 路径时冻结、后续复用与新输入清空。首个 decode 入口可能仍消费 question token；不把它无条件称为“首生成 token”。
3. 独立保存 `calibration_far_ids_hash` 与真正跳过的 `selection_far_ids_hash`，核边界/页取整/保护规则是否一致。不同则分别报告，不假定标定 mass 就是被删质量。
4. 开发集冻结 rows、ε、层策略及允许的失败阈值，再在确认集测触发率和后续 far-mass 漂移。若当前候选只用一个固定 ε，不新增阈值×任务×层海选；若阈值待定，明确 `UNFROZEN`，本图仅作探索。

固定模型、输入、query 采样、保护/nearfar 定义和轨迹；首先不执行 gate 干预，避免“关 far 后 attention 自然无 far”这个循环论证。实际任务效果和 refill 资源变化另由 A4 的 G0/G1/G2 比较。

### 数据、统计与出图

- 采集校准和 decode 的逐 head/group/layer `far_mass`、`mass_on_actual_skipped_ids`、标定样本数、触发决定、真实入口 token、随 decode 步的变化；标记不适用、空 far 与短序列，不能都记为安全0。
- 按独立序列计算触发率、触发层的后续风险质量分布及组内最差头；风险阈值在开发集冻结。给跨样本区间，头/层不当作独立样本。相关性只作预测性证据，不代替非劣效任务试验。
- 主候选面板：横轴 layer、纵轴原始 full-reference far mass，展示跨序列中位数/区间并附组内最差头摘要。标定→decode 对应散点或漂移曲线放附录；若图太拥挤，主文仅留一个面板。
- 原件用 `figure_id=O2`，每个触发点绑定标定与后续 query。gate 校准耗时、全维计算、同步和后续成本归 E2/A4；本图不把低 mass 直接转换成预计速度。

### 可支持、不可支持与停止条件

低平均 far mass 只能提出 gate 候选，不能证明任务无损；小概率 token 仍可能影响关键输出，输出误差也受 value 与后续层影响。若确认集上漂移大、少数头风险明显或标定/实际 far 支持不一致，停止“该层可安全关闭”的推论，保留静态双32主方法。gate 未通过质量和完整成本验收时，本图只能列为探索性观察或附录。

## O3 完整 RoPE 对、频段与低维检索保真

**稳定 ID：O3；优先级 P0；主文候选一个面板。问题：**在同样的32维预算下，tail 完整 RoPE 对相对其他坐标能否保留后续 token 检索所需的信息，而非仅因用了不同宽度或破坏配对占优？

### 实验与固定量

1. 主观察固定同一组 post-RoPE Q/K、scale、候选支持，比较32维 tail、32维 high、32维 paired-random；这些 selector 臂使用相同量化/累加规则。mass/output 的参考始终是原始未QDQ的128维 full-attention 概率；如另加128维QDQ selector，明确命名为量化评分参照，不能替代前者。坐标必须以实际 pair ID 与标量坐标 ID 记录；Qwen3-8B 的16个完整 pair 对应32维，不能仅凭数组位置宣称频率排序，必须结合 checkpoint 的 RoPE 参数验证。
2. 当前 main 的 `subspace=random` 使用固定32个任意标量坐标，不保证完整 RoPE 对，且不会随比较宽度改变。因此它只能命名为旧实现 random-coordinate 诊断，不能冒充纯频段对照或随机16/64控制。paired-random 对照在独立实现/数值审查通过前标记 `BLOCKED`，不得用旧随机结果补空格。
3. paired-random 需预登记种子清单，并独立报告种子间变异；不从多次随机抽样挑最好或最差。为控制工作量，主频段观察仅32维；16/32/64宽度留 A2 的 tail 单轴敏感性，不做全部频段×宽度×任务网格。
4. 检索主参考为当前 GQA 语义：每 Q-head 评分/softmax、相同支持、组内聚合，再得到固定预算的 full-reference 选集。逐头偏好冲突另报，不能把组共享参考与逐头 oracle 混成同一分母。

### 数据、统计与出图

- 同时采集组级选集与 full-reference 的 intersection、recall/Jaccard、所选原始 full-attention mass、各 Q-head 最差保留质量与组内偏好冲突；L1 页和 L2 token 分开，预算/保护是否包括在指标中明确写出。L1默认用候选页覆盖token的full-reference mass；若用page recall，先冻结参考页生成规则、页预算及保护处理并保存reference_page_ids，不能把token oracle暗换成页oracle。
- 若另作近似分数相关性，记录 rank/correlation 的支持与 tie 处理；相关性高不等于 TopK 一致。固定预算下 overlap 是检索诊断，不是最终任务准确率。
- 可选 attention-output 相对误差与投影后误差，必须固定原始 V、W_O、数值分母和精度。它们是局部代理指标，不作端到端误差上界或无损证明；没有已登记原件不强制新增昂贵 trace。
- 每个数据点记录 `dim/pair_ids/coordinate_ids/seed/quant_dtype/scale/valid_support_hash/topk_ids_hash/reference_ids_hash`，以及 per-head 与 group 指标。独立序列配对区间，随机种子作为额外变异来源说明，不把所有 head 当重复实验。
- 主面板：横轴 tail/high/paired-random（全部32维，随机显示跨种子区间），纵轴 full-reference retained mass 或检索 recall，二选一预登记主指标，另一个放附录；必要时分 L1/L2 两小面板。不能把不同含义指标共用一条纵轴。`figure_id=O3`。

### 可支持、不可支持与停止条件

若 tail 在配对确认集上稳定保留检索信息，可为固定 tail32 提供经验动机；任务质量与完整成本仍由 A2/E1/E2证明。若 paired-random 或 high 表现相当，诚实说明固定 tail 的简单性/部署成本，不声称低频独有优势。若 RoPE 配对/排序、支持、预算或 scale 不一致，先停止该频段结论。FASA 的校准式 dominant component 选择不是“固定低频总优”的外部证据。

## 最小主文组合、复用与审查

1. 主文最多把 O1、O3 和已验必要的 O2 合为一张2–3面板动机图；不把每层、每头、每任务都扩成主图。O2 的 gate 仍是候选时保留附录或省略。
2. O1 可复用 A3/A5 的同身份 full-reference trace；O2 可复用 A4 的无干预轨迹；O3 可复用 A2 的固定 Q/K 对拍。复用必须 identity 完整，2048-token 选择 trace 不能冒充1024预算的任务质量结果。
3. 每个图交付 E0 的数据切片、绘图脚本、caption、原件哈希、统计单位、选择规则与独立检查。图注写明“观察/诊断”，不把设计、回复、图规划或未公开原件写成实测。
4. 停止规则面向主张：出现负例时保留并收缩结论；不通过更换样本、阈值、层集或坐标重新包装确认结果。允许机制假设被否定，这本身是本指南的有效交付。

## 一手论文对照与边界

- [Quest正式PMLR版](https://proceedings.mlr.press/v235/tang24l.html)，§3.4/Fig.5：页级 min/max 查询感知选择说明应关注被遗漏重要 token，而不是仅看平均 logit。它直接在选中页上做 attention，不能借它证明本项目 L2 token 选择已减少全S评分成本。
- [SparQ正式PMLR版](https://proceedings.mlr.press/v235/ribar24a.html)，算法与§5.4/Fig.7组件消融：先选查询重要通道，再检索 token，最后全维 attention 并带遗漏 value 的近似补偿。可借鉴检索保真与最终输出分开评价；其动态坐标和补偿不同于本项目静态 tail32，不能当固定频段最优证据。
- [ClusterKV v2](https://arxiv.org/abs/2412.03213v2)，方法与§V-C/Fig.12–13：语义聚类和系统缓存/传输共同决定收益。它不是本项目 near/far 边界的理论证明；只借鉴重要 token 分布的可解释展示及完整成本核算。
- [FASA v3](https://arxiv.org/abs/2602.03152v3)：校准得到 dominant RoPE components，再做稀疏选择。用于提醒频段结论需要模型/校准依据，不等于预先认定最后16对最优。




