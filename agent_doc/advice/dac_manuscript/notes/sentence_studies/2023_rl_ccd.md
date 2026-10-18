# RL-CCD (DAC 2023 Best Paper) 逐句精读笔记 — Abstract + Introduction

> 2026-10-18 第 26 轮逐句学习，直接读 PDF 原文（`2023_RL_CCD.pdf`，DAC'23，GT + Synopsys）。
> RL-CCD 是「RL 替代商业工具内建启发式」类 best paper（端点优先级选择替代均匀 CCD），观察其 However 直判 flaw 句、Not-all-X-are-equal 一行论点句与公平比较声明句。

## Abstract（4 句，极紧凑）

| # | 原文（缩写） | 功能 |
|---|---|---|
| A1 | "Concurrent Clock and Data (CCD) optimization is a **well-adopted** approach in modern commercial tools that resolves timing violations using a mixture of clock skewing and delay fixing strategies." | **well-adopted 信誉句**（先给实践背书——商业工具已广泛采用，再谈缺陷——让 flaw 判词有分量） |
| A2 | "However, existing CCD algorithms are **flawed**. **Particularly**, they fail to prioritize violating endpoints ... correctly, **leading to** flow-wise globally sub-optimal results." | **However 直判 flaw 句 + Particularly 收窄句**（一个词的判词 are flawed——极 blunt；Particularly 把判词落到具体失败点；leading-to 后果链） |
| A3 | "In this paper, we **overcome this issue** by presenting RL-CCD, a Reinforcement Learning (RL) agent that selects endpoints for useful skew prioritization using the proposed EP-GNN, ..., and a Transformer-based self-supervised attention mechanism." | **Overcome 登场句**（回应 A2 缺口+命名+双机制枚举——In-response 家族的 overcome 变体，动词更强势） |
| A4 | "Experimental results on **19 industrial designs in 5−12nm technologies** demonstrate that RL-CCD achieves **up to 64%** TNS reduction and **66.5%** NVE improvement **over the native implementation of a commercial tool**." | **范围限定证据句**（19 设计+工艺范围作状语前置；双指标 up-to；基线点名 native implementation of a commercial tool——商业工具原生流程即基线） |

对比：4 句完成 stakes→flaw→overcome→证据。我们的摘要 7 句是诚实镜像变体（双实测对+边界句+反向计时句），多轮已判定保持。A4 的「范围状语前置+基线点名」我们等价（11-task RULER 32K、Qwen3-8B/32B、dense FullKV 点名）。

## Introduction

### P1（2 句 + 冒号引出 bullet）：定义 + 目标 + However-neglect 引子
- 定义句（interleave clock skewing and delay fixing——termed CCD——术语定义即开场）。
- **In-general-aims 句**（"CCD **aims to** find an optimal balance between 'clock' and 'logic' optimization **so as to** resolve violating timing endpoints in a flow-wise globally optimized manner"——目标平衡句带 so-as-to 目的）。
- **However + mainly-because 引子句带冒号**（"existing CCD algorithms fail to achieve this goal, **mainly because they neglect the following vital fact**: "**——归因引子+冒号悬停，为 bullet 论点蓄势）。

### Bullet（一行论点 + 机制平行 + However 复述）
- **一行论点句**（"**Not all violating endpoints are equal.**"——6 词论点作 bullet 开场，全文最短句承载全文核心洞察）。
- **distinct-sensitivity 展开 + better-to-be-fixed-more 平行机制句**（"some of them are better to be **'fixed more'** by clock-path optimization (i.e., a larger portion of their slack values should be resolved by clock fixing), **while others** are better to be 'fixed more' by data-path amendment"——引号口语化短语+i.e. 形式化的双轨表述；while-others 平行；两种端点各挂最优策略）。
- **However no-intelligence 复述句**（"existing CCD algorithms have **no intelligence on** weighing the balance between different strategies in endpoint level, leading to flow-wise sub-optimal results"——把 bullet 论点回收成前作缺陷——bullet 自身即微型 P1-P5 弧）。

### P2（2 句）：Overcome 登场 + Prior-to 范围句
- **Overcome 登场句第二变体**（"we overcome this critical issue by presenting RL-CCD, ... **built upon** a customized GNN named EP-GNN ..., and an attention-based encoder-decoder network using self-supervised learning [9]"——built-upon 枚举挂引用）。
- **Prior-to-CCD 范围句带 rather-than 排除**（"**Prior to** CCD optimization, RL-CCD selects a subset of violating endpoints that should be prioritized for clock-path optimization using useful skew (**rather than** data-path optimization using buffering, sizing, restructuring etc.), to achieve flow-wise globally optimal PPA metrics"——时序定位+rather-than 排除声明+目的——一句话同时说清做什么/不做什么/为什么）。

### P3（3 句）：The-goal 一行大目标 + Note-that 阶段收窄 + 参照工具点名
- **The-goal-of-this-work 一行大目标句**（"The goal of this work is to **unleash the true power of** CCD optimization in commercial tools using RL."——unleash 真力量——大目标单句成段，气势句）。
- **Note-that 阶段收窄句链**（"**Note that** modern PD tools perform CCD optimization throughout the entire PD flow. To demonstrate the effectiveness..., we **specifically focus on** improving the timing quality of CCD optimization **at the placement stage**, where the goal is to achieve better TNS values at the end of the entire placement optimization"——整流程→placement 阶段的两级收窄，Note that 先陈述全貌再切范围——范围诚实边界）。
- 参照工具点名句（industry-leading commercial PD tool, Synopsys IC Compiler II (ICC2), demonstrate significantly improves——点名基线工具+改进声明）。

### P4（3 句）：图 1 差异走查 + 公平比较声明
- **图差异载体句**（"Figure 1 **highlights the innovations of our framework and the key differences over** the native implementation"——图承担「创新点+差异」双功能）。
- **Given-X-unlike-Y 走查句**（"**Given** a globally placed netlist, **unlike** the default tool flow that has no intelligence on balancing CCD optimization techniques via endpoint prioritization, **our** RL agent selects a group of violating endpoints..."——同一输入下 unlike-默认/our-方法的对照走查）。
- **Note-that 公平比较声明句×2**（"**Note that** the total optimization steps between the left flow (default) and the right flow (ours) are **exactly the same**. **Except for** endpoint prioritization using margin (which is removed after useful skew), RL-CCD is **not taking any additional optimization step**."——步骤数恒等+无额外步骤双声明：改进来自机制而非加算力——把公平性本身写成卖点）。

### P5（1 句）+ 贡献 bullets ×3
- **The-outcome-of-our-effort 收口句带 i.e. 括注**（"The outcome of our effort is a **universal** (i.e., generalize to any design and technology) RL-based framework, which **drastically improves**..."——成果总括+括号内 gloss 展开论点词）。
- **贡献 1：问题发现即贡献**（"We **discover a new PD problem and demonstrate its importance**. **That is**, finding a balance between clock-path and data-path optimization through endpoint prioritization..."——发现新问题本身列第一贡献+That-is 展开）。
- **贡献 2：first-ever 首创声明+up-to/avg 双口径**（"the **first-ever** endpoint prioritization framework ... achieves **up to 64%** TNS improvements (**avg. 23%**), and **up to 66%** NVE reduction (**avg. 19%**)"——首创+每指标 up-to 与 avg 成对披露）。
- **贡献 3：可迁移性收益句**（"facilitates transfer learning and self-supervised learning, which makes it generalizable to any design or technology. A **pre-trained** RL-CCD agent can significantly improve ... with **only few iterations** of training"——预训练收益带 only-限定）。

## 对照我们手稿的差距结论（摘要 + Intro 遍历）
| RL-CCD 句式 | 我们的状态 |
|---|---|
| A1 well-adopted 信誉句 | 我们的 stakes 句（LongBench/RULER 引用型）等价；无商业工具背书可引，不加 |
| A2 However 直判 flaw + Particularly | 我们的缺口是问题固有难度（选择成本）非前作缺陷；诚实体裁=「我们的局限是我们的测量」（第 22 轮已判定结构不同不改） |
| **一行论点句**（Not all X are equal.） | Intro P3 首句 "A global TopK budget lets high-scoring candidates from one region occupy slots without guaranteeing minimum capacity for the other."（claim-first 论点句）等价，不改 |
| better-fixed-more while-others 平行机制 | Intro P2 "Mean pooling can dilute isolated large token scores, while extrema bound each page's largest shared-space token score." 等价，不改 |
| Overcome 登场（强势动词） | Intro P4 "These observations lead to a cascade with two explicit controls"（第 22 轮判定双回收等价）；诚实边界禁 overcome/flawed 级判词（我们的比较未隔离机制），不加 |
| Prior-to 范围句带 rather-than 排除 | §III "The cascade operates only during decoding; prefill remains dense" + Related Work "ordinary L2 still scores the full history before masking" 等价，不改 |
| The-goal unleash 一行大目标 | 诚实边界禁（无已证力量可 unleash），不加 |
| **Note-that 阶段收窄句链**（整流程→placement） | Setting 臂身份绑定+§III 解码范围声明覆盖，等价，不改 |
| 图差异载体句 | fig:observation 各面板走查句等价，不改 |
| **Note-that 公平比较声明**（步骤恒等+无额外步骤） | Setting "Controlled sparse comparisons must align final unique-token cap K (including protection), model, inputs and phase policy" 等价（比较对齐要求=公平控制），不加 |
| first-ever + up-to/avg 双口径 | first-ever 受 AsyncTLS already-combines 制约（诚实边界）；我们的指标本身是 AVG 单口径（无 up-to/avg 分层可报），不加 |
| 问题发现即贡献 | 无贡献 bullet（问句标题惯例，第 6 轮判定保持）；Intro P2/P3 观察段+P4 回收承担同功能，不改 |
| The-outcome-of-our-effort 收口 | Intro P5 承担（"We formulate... The evaluation examines..."），等价不改 |
| **本轮实缺 = Intro P4 主谓失配+指代歧义** | "The historical layer variation in panel (d) motivates this extension, **but uses** a different single-query statistic"——`uses` 的合法主语缺失：variation 不能「采用」统计（第 24 轮同类病灶），且若挂到 "this extension" 则语义反转（真义=panel (d) 的诊断统计不同于当前门控，caption 里 "It uses ... unlike the current gate" 已明示 It=诊断）。zh 版「但采用」同样悬空 |

### 本轮修改（第 26 轮）
- Intro P4：`The historical layer variation in panel (d) motivates this extension, but uses a different single-query statistic;` → `The historical layer variation in panel (d) motivates this extension, but the diagnostic uses a different single-query statistic;`（+2 词，补主语 the diagnostic——panel (d) 的诊断；消除歧义）。
- paper_zh.tex 直译同步：「但采用不同的单查询统计」→「但该诊断采用不同的单查询统计」。
- 版式核对：编译后核对 en 6 页。

---

## §2 Related Works（第 27 轮，2026-10-18）

### §2.A Predictive Useful Skew（1 段）
- **well-known 功劳句 + However as-pointed-out 引用缺口**（"Useful skew is a well-known technique that improves design timing ... **However, as pointed out in** [11], computed skew adjustments often require redo synthesis or placement to truly realize timing benefits"——缺口挂别人引用：as pointed out in——判词外包给文献，非本文自判）。
- **chicken-egg 习语锚定缺口句**（"To break the **chicken-egg problem** (i.e., iterative back annotation of skew), the authors of [1] proposed a 'predictive' useful skew technique ... where the TNS value can be improved by **up to 5%**"——习语命名问题+i.e. 形式化+引号命名+量化收益——问题命名的记忆点）。
- **In-this-paper-since 范围排除定位句**（"**In this paper, since** commercial PD tools have well-integrated useful skew techniques into CCD optimization and reach considerable success, **we are not focusing on** improving the native useful skew implementation in tools. **Our specific focus is to** balance clock-path and data-path optimization ... via endpoint prioritization, **which is an under-researched problem**"——三步定位：既然工具已做好 X→我们不做 X→我们专做 Y 且 Y 欠研究——边界声明即定位）。

### §2.B Learning-Driven Timing Prediction（1 段）
- 前作枚举句链（tree-based PBA 预测 [7] / GNN slack 预测 [2] / **Transformer gate sizing 带 1400x 量化**（"where a 1400x speed up is achieved in obtaining tool-accurate sizing moves on unseen netlists"）——每前作一句带量化或方法名）。
- **Nonetheless 不比较诚实声明句**（"**Nonetheless**, in this paper, **we are not comparing** RL-CCD with above methods **as** the focus of RL is to improve optimization quality **rather than** prediction accuracy"——主动声明不比较+理由——比较范围的诚实边界，防审稿人质疑漏比）。

### §2.C RL in EDA: Going Beyond Commercial Tool Quality（1 段）
- 小节标题即宣言（Going Beyond Commercial Tool Quality——标题带立场）。
- **时代让步代价句**（"**As the benefit of scaling saturates**, leading-edge commercial tools are seeking more powerful methods for PPA optimization **even at the cost of** runtime"——宏观趋势+even-at-the-cost-of 让步——为 RL 高运行时铺路）。
- **RL 双优点句**（"RL thus becomes a promising solution **as** it does not require any labeled data and has been demonstrated to achieve never-seen, high-quality optimization results in many fields [12]"——无需标注+never-seen 双卖点挂引用）。
- **superlative 前作枚举句**（floorplanning "generates **superhuman** floorplans that human labor is not able to achieve" [6]；RL-Sizer "can outperform the default sizing algorithms in a commercial tool **although it adopts a fundamentally different approach**" [5]——让步异曲同工的胜利证据）。
- **With-the-above-success-stories 收口宣言句**（"**With the above success stories**, in this paper, we decide to continue the research of RL in PD, **aiming at going beyond what state-of-the-art commercial tools are able to achieve**"——前作胜利回收→本文目标宣言——Related Work 收口即立场）。

### 对照我们手稿的差距结论（Related Work 遍历）
| RL-CCD §2 句式 | 我们的状态 |
|---|---|
| well-known 功劳 + as-pointed-out 引用缺口 | Quest/SparQ/HISA 功劳句（不带判词）+ AsyncTLS already-combines 等价；缺口挂引用形式不适用（我们对前作无 flaw 判词资格——诚实边界） |
| chicken-egg 习语锚定缺口 | 无习语可用（我们的缺口是设计约束非循环依赖），不加 |
| **In-this-paper-since 范围排除定位句** | P1 双定位句（"Our selector instead..."）+ AsyncTLS already-combines + "distinguishing page-level eligibility" 覆盖定位；「since 工具已做好→不做」形式无诚实对应物（我们的组件与 AsyncTLS 重叠，不能宣称工具已做好），不加 |
| **Nonetheless 不比较诚实声明句** | Setting 节 "We retain these nominal labels without treating the runs as controlled single-stage ablations" 等价（诚实不比较声明），不改 |
| 时代让步代价句 | 无宏观趋势可引，不加 |
| RL 双优点句 | 不适用（非 RL 方法），不加 |
| superlative 前作枚举 | 诚实边界禁（对前作无 superhuman 级判词），不加 |
| With-the-above-success-stories 收口宣言 | 诚实边界禁（无 going-beyond 立场可主张——我们的定位是「组合+区分约束」），不加 |
| 小节标题即宣言 | 我们的 Related Work 无小节（单层两段），结构不同，不加 |

### 本轮修改（第 27 轮）
- 本轮手稿修改来自**远端裁决**（784c1a4 Intro 重写四点裁决）：整体采纳（claim-first、We-contribute 枚举正是 best-paper 贡献列表句式），但**面板 (c)/(d) 主文引用被整句删除**——四面板图两面板无正文引用（图文一致性缺口，裁决点 1），且 P4 低频坐标主张失去动机诊断支撑链（裁决点 3）。恢复最小引用（caveat 细节 caption 已承载——"random32 does not preserve RoPE pairs"、"It uses ... unlike the current gate"——正文只补引用与最短限定）：
  - en P4 插入 (c) 句：`The coordinate diagnostic in Figure~4(c) motivates the compact view; its random baseline does not preserve RoPE pairs.`
  - en P4 门控句恢复 (d) 引用：`We also formulate a prefill-informed far gate motivated by the layer variation in Figure~4(d); it is disabled by default and requires separate evaluation.`
  - zh 直译同步两句：「图(c)的坐标诊断为紧凑视图提供动机；其随机基线未保留RoPE旋转对。」「本文还给出由图(d)中历史层间差异提供动机的预填充远区门控扩展；该门控默认关闭，需要单独评估。」
  - **版式溢出处置**：初版恢复（含完整 caveat，+46 词）使 en 正文溢至 6 页内容+references 第 7 页（零余量失败模式第三次）；两次压缩至 +12 词净增后回到 6 页总页数。caveat 的主文版（"cannot isolate frequency from pair structure"/"different single-query statistic"）以 caption 版为准——诚实信息未删，仅移位。
- 44d09d7（Design implications 压缩，Thus 收口句移除）裁决**采纳**：takeaway「quality--cost benefit remains unresolved→requires matched measurements」在结论完整存活，属零余量压缩非信息删除，不回插。
- 编译验证：en 6 / zh 6 页，0 未定义引用。

## §3 RL-CCD Framework（第 29 轮拆解，4 小节全）

### §3 开篇：形式化问题陈述（1 段）
- **Given-Goal-such-that 形式化开篇句**（"**Given** a globally placed netlist G=(V,E), the **ultimate goal** of RL-CCD is to select a group of violating endpoints V′∈V to be prioritized..., **such that** the TNS value after the proceeding placement optimization steps can be optimized"——数学对象→目标→使什么最优——设计节的标准开篇：先把问题写成可判定的形式）。
- **Note-that 空配置披露句**（"**Note that** V′ is an **empty set** in the native implementation of the reference commercial tool"——参照工具在该自由度上什么都不做——为贡献空间划界的诚实披露；我们无对应参照工具空配置，不加）。
- **In-this-paper 头牌宣言句**（"In this paper, we **demonstrate** that by selecting proper V′, the achieved TNS value can be **drastically improved**"—— drastically 属 superhuman 级判词，诚实边界禁；但「本节将证明 X 可改进 Y」的方向声明结构可学）。

### §3.A Overview and RL Formulation（2 段）
- **Figure-forward 概览句**（"Figure 2 shows a high-level overview of our RL endpoint selection process"——先指图再讲机制）。
- **Key-idea 机制链句**（"The **key idea** behind is to let the useful skew engine 'over-fix' the timing of the RL-selected endpoints **so that** the proceeding delay optimization techniques can spend less effort on them, **which together result in** flow-wise optimal solutions"——引号命名机制（over-fix）→ so-that 因果链 → which-together-result 收口——机制段的核心句式）。
- **替代路线承认+经验裁决句**（"**We understand another route may also work** (i.e., useful skew 'under-fix'), **however, we empirically observe that** the proposed method...works **significantly better**"——承认替代设计存在（We understand）→ 用实证观察裁决（empirically observe）——设计选择的诚实论证模板。**我们对 β 或评分视图无此级配对证据，不能套用判词 significantly better**；但「承认替代存在→实证裁决」结构记入范式库）。
- **问题分类句**（"Our endpoint prioritization problem **is a combinatorial optimization problem**. In this paper, **we choose to formulate it as** a Markov Decision Process"——先归类问题难度（组合优化=NP难暗示），再声明建模选择（we choose to=建模自由度的主动披露））。
- **MDP 五元组 bullet 定义列表**（States/Actions/State Transition/Reward/Trajectory——每项一两行，Reward 定义里嵌零奖励设计（"rewards are zero for intermediate actions except for the last"）——把实现细节写进形式化定义的紧凑做法）。
- **目标函数句**（"The ultimate goal of our RL agent...is to obtain an optimal policy π that **maximizes the expected return** J..., which can be denoted as: (1)"——文字目标→公式编号，where 从句随后定义 π_θ）。

### §3.B Detailed Architecture（3 子小节）
- **三组件编号枚举开篇**（"RL-CCD **consists of three main components**: (1) a GNN module..., (2) a LSTM network..., and (3) a self-supervised attention module"——编号+功能短语，与我们 §III 开篇「two explicit mechanisms」同构 ✓）。
- **耦合声明+图指针句**（"**Note that each of them is hardly independent of each other**. Figure 4 depicts the details of how they **function together** in one RL time step t, **forming a circular selection loop**"——组件不独立（防止读者当模块化消融解读）+「如何协同」的图职责声明——我们的对应物是「One 32-coordinate map supports both L1...and L2...」共享耦合句 ✓）。
- **B-1 GNN：先功劳后借用句**（"GNNs **have shown promising results in advancing** many traditional PD tasks **thanks to** their ability to perform effective graph representation learning [4]"——借技术前先给领域级功劳挂引用——我们 Related Work 的 Quest-construction 给出 credit 同构 ✓）。
- **命名抽象引入句**（"In this paper, **we present EP-GNN**, an endpoint-oriented GNN framework that **focuses on** generating node embeddings...through iterative neighborhood and fan-in cone aggregation"——命名+定向（endpoint-oriented）+一句功能——与我们「shared scoring view」命名同构）。
- **流水线顺序词句**（"**Prior to** the actual graph learning, we first construct...**Then**, for each node..., we hand-craft a comprehensive list of features as shown in Table I"——Prior-to/Then 顺序词+表指针——特征表 Table I 用「Note that the first attribute will be updated in each RL iteration」脚注披露动态特征——特征表的诚实注法）。
- **公式后逐符号定义句**（"where σ denotes..., N(v) denotes..., γ denotes **the trainable parameter that weighs** the importance between..."——where-从句逐符号；参数的语义角色（weighs the importance）而非仅类型——我们 U_b 公式后「where z+=max(z,0)」同构 ✓）。
- **因果重算披露句**（"**Since** the masking mechanism changes some node features...after each selection, the graph learning by EP-GNN **is conducted in every RL time step**"——Since-X-therefore-每步重算——把成本来源写成因果——我们「普通 L2 仍沿全历史执行点积」同构 ✓）。
- **B-2 LSTM：借用-定制句**（"Our encoder-decoder structure...is **inspired by the renowned** Transformer architecture proposed in [9]. In this paper, **we customize the renowned architecture** to solve our specific problem **by replacing** the encoder with a LSTM...**and by simplifying** the attention mechanism..."——inspired-by 挂引用→两个 by-动名词并行说明改动——「借名+改两处」的诚实定位）。
- **收益带规模锚句**（"Our effort **significantly reduces the number of parameters** required for training, **making the framework fully applicable to industrial designs with millions of instances**"——参数减少→工业规模可用（millions of instances 锚）——设计收益句必须带规模/倍数锚。**本轮对应修复：我们「Compact scoring narrows dot products」→「narrows each dot product fourfold」补上 4× 数字**）。
- **Rationale-first 组件引入句**（"The **rationale behind using** LSTM..., a renowned sequence encoding network, for past actions encoding **is that** the decision of each selection is made sequentially, and each of them should not be independent of each other"——为什么用 X：因为问题本质 Y——组件引入先给存在理由。**本轮对应修复：坐标选择句「Their slower positional rotation」指代悬垂（最近先行词是 z_q/z_i 而非 16 个旋转对）→「The retained pairs' slow positional rotation」**）。
- **数据流边披露句**（"**Note that** the hidden vector ht is passed to **both** the decoder network at the current time step t **and** the LSTM-encoder itself in the next time step t+1"——双消费者数据流边——防读者漏掉自回归边）。
- **B-3 Attention：目的先行句**（"**To efficiently consider all endpoints at once** for the selection, in this paper, we leverage a self-supervised attention mechanism"——To-目的 → leverage 手段）。
- **公式大白话复述句**（"**Basically, Equation 5 aims to find** the weight matrices that **jointly quantify the importance** of all endpoints"——公式后用 Basically 复述直觉——我们 U_b 后「Each signed coordinate contribution is bounded by its chosen extremum, so...」同构且更严谨 ✓）。
- **单调性解释句**（"**With a higher attention score** A(i)t, an endpoint i **will have a higher probability of being chosen**"——分数→概率的单调关系一句话——我们「Raising β cannot reduce the nominal near-page quota」单调性句同构 ✓）。
- **性质链收口句**（"**Note that** the entire...process is performed in a self-supervised manner. **That is**, we are not using any pre-defined label...**Hence**, RL-CCD is **generalizable to any design or technology** as..."——Note 性质→That-is 展开→Hence 泛化结论——三连性质论证。generalizable-to-any 判词诚实边界禁（我们的配置绑定 Qwen3-8B），不加）。

### §3.C Masking Rationale（1 段）
- **Two-fold 理由枚举句**（"The rationale behind is **two-fold**: (1) **from design knowledge**, successive endpoints are better not to be prioritized at the same time, **otherwise** it may cause ping-pong effect..., and (2) for different designs, our strategy **allows the RL agent to decide** the total number...**by either** aggressively selecting...**or vice versa** based on design characteristics"——理由一=领域知识+otherwise 反例（ping-pong effect 习语命名危害）；理由二=自适应性（允许智能体自己决定数量）——把设计选择的双重理由编号展开的标准模板）。

### §3.D Training Methodology（1 段+算法）
- **命名算法+同位语定义句**（"we leverage REINFORCE [12], **a renowned policy gradient algorithm**, to train..."——算法名+同位语一句话定义+引用）。
- **跨节目标回指句**（"The objective of our RL agent **is defined in Equation 1**, which is to maximize..."——回指 §3.A 的目标函数保持一致性——跨小节符号/目标复用必须回指）。
- **公式文字等价复述句**（"Equation 7 **denotes that** the gradient of the objective **is equivalent to** the expected sum of the gradients of the log probabilities of the taken actions, **weighted by** the achieved reward"——公式→文字等价陈述）。
- **算法逐行走读句**（"Algorithm 1 illustrates the end-to-end training process...**We first initialize**...in Line 3. **Then, in Lines 5–13**, we sequentially determine...**Note that** an overlap masking is performed in Line 11 to...**When the selection process completes**, in Line 14, we worsen...**Finally**, we run through...**With the reward**, all parameters are jointly updated in Line 18, and the whole process **repeats until** the reward (TNS) is optimized"——first/then/when/finally 阶段副词+行号锚+Note-that 插注+repeat-until 终止条件——算法走读的完整句式链。我们无伪代码，但 S7 整数规则段的顺序叙述可参照）。

### 对照我们手稿 §III 的差距结论与修复
| RL-CCD §3 句式 | 我们的状态 | 本轮处置 |
|---|---|---|
| Given-Goal-such-that 形式化开篇 | §III 开篇「To control both...the cascade uses two explicit mechanisms」目的先行+枚举 | 已达标，不改 |
| 组件编号枚举+耦合声明 | 「two explicit mechanisms」+「One 32-coordinate map supports both L1 and L2」 | 已达标，不改 |
| 公式后 where 逐符号定义 | U_b 后 where z± 定义 ✓；o_{h,g} 前置符号定义 ✓ | 已达标 |
| 公式大白话复述 | U_b 后「Each signed coordinate contribution is bounded...so」 | 已达标 |
| 收益带倍数/规模锚 | 「narrows dot products」缺倍数 | **修复：→ narrows each dot product fourfold（32/128=4×，算术事实，非判词）** |
| Rationale-first 组件引入 | 坐标选择动机句指代悬垂 | **修复：「Their slower positional rotation」→「The retained pairs' slow positional rotation」（先行词从 z_q/z_i 修正为 16 个旋转对）** |
| 替代路线承认+经验裁决 | 无配对证据支撑「significantly better」级判词 | 诚实边界禁，不加 |
| generalizable-to-any / drastically / superhuman 级判词 | — | 诚实边界禁，不加 |
| Note-that 空配置披露 | 无参照工具空配置对应物 | 不加 |
| define-before-use | **主文 $S=t+1$ 定义后全文未用（仅 S7 使用且 S7 未定义）——define-without-use 违例** | **修复：主文删 S 定义（-10 词），定义移入 S7「Integer boundaries」使用处（含排除填充语义）；「far is its complement there」→「far is the rest of $\mathcal M_t$」消解 there 指代** |

### 本轮修改（第 29 轮）
1. en §III-B：「$S=t+1$ counts valid cached positions, excluding padding」删除（主文 0 使用）；S7 补「with $S=t+1$ valid cached positions including the current token and excluding padding」（en+zh，定义移到唯一使用处——define-before-use 正向修复）。
2. en §III-A：「Their slower positional rotation」→「The retained pairs' slow positional rotation」（指代悬垂修复，RL-CCD rationale-first 句式校准）；zh「其较慢的位置旋转」→「所保留旋转对的位置旋转较慢」。
3. en §III-B：「far is its complement there」→「far is the rest of $\mathcal M_t$」；zh 已显式（「远区为它在M_t内的补集」）不动。
4. en §III-A：「Compact scoring narrows dot products」→「narrows each dot product fourfold」（RL-CCD 收益带倍数锚句式；32/128 算术事实）；zh「缩小评分点积」→「将每次评分点积缩窄至四分之一」。
5. 远端 1ecc63d 裁决**整体采纳**（先逐项核验 manifest：三臂 source_data_sha256/ids/answers_sha/lengths 全等 ✓、稀疏两臂 scorer+script 哈希一致而与锚不同 ✓、锚 legacy-partial+operator-declared YaRN ✓）：图注按长度区分 β 归属（64K 展示 8B 两 β）为真实纠错；S8 64K 段 manifest 级披露深化诚实边界；「are reported on one machine」归因严谨化；S7 attribution-controls 段重写更可执行；draw.py 相对字体路径+去重复 y 轴标签。
