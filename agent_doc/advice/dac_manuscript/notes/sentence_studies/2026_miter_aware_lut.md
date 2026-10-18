# 2026 DAC Best Paper 逐句拆解：Miter-Aware LUT Mapping（Zhu et al., CUHK）

> 第 7 篇（队列最后一篇）。来源：`agent_doc/advice/DAC_best_paper/2026_Miter_Aware_LUT_Mapping.pdf`，
> 全文提取 `/tmp/miter_lut.txt`（40314 字符，7 页）。本轮（第 31 轮）：Abstract + §1 Introduction。
> 论文主题：LEC（逻辑等价性检查）——把两设计映射到 LUT 抽象再做 SAT，取代平铺 miter 直连。
> 领域对应：我们的「共享评分视图 + 区域配额」同样是「求解前的表示建模」主张——同构度极高。

## Abstract（7 句）

1. "Logic Equivalence Checking (LEC), a fundamental hardware verification task, is often bottlenecked by synthesis-induced structural perturbations and XOR-dense regions that degrade SAT solver performance."
   - **任务定义内嵌 + 双痛点归因句**：一句完成 stake（fundamental 任务）+ 两个 bottleneck 命名（综合扰动 / XOR 密集区），who-cares 与 what-breaks 压缩在同句。"bottlenecked by" 把痛点定位为结构性而非算法性。
2. "We contend that the modeling of the miter is as critical as the SAT solver itself."
   - **论点宣言句（全文灵魂）**：contend + as-critical-as 对等句式，一句翻转领域默认假设（默认：solver 是主角）。六词核心洞察的抽象版——摘要第二句即立场，不先讲方法。
3. "To this end, we introduce a miter-aware mapping framework that strategically formulates the problem before solving."
   - 登场句 + **before-solving 时间边界**：框架的价值定位压缩在一个介词短语里（在求解之前把问题构造好）。"strategically" 属虚词（诚实边界内可容忍但不学）。
4. "By constructing a LUT-based miter—instead of a traditional, flat netlist—our approach preserves critical structural correspondence between the two designs while making high-level logic relations explicit."
   - **instead-of 双破折号机制对比句 + while 双收益**：新表示 vs 旧表示一短语内对撞；preserve X while making Y explicit——结构收益与显露收益并列。
5. "Our framework uniquely integrates three techniques: equivalence-preserving mapping to structurally align the two circuits, Gaussian-guided XOR modeling to algebraically simplify dense arithmetic, and solver-oriented LUT selection to generate a representation optimized for efficient SAT reasoning."
   - **三组件列举句：命名品牌词 + to-不定式目的**。每个组件一个三词复合名（equivalence-preserving / Gaussian-guided / solver-oriented），每个后接一个 to-从句说明它买来什么。命名密度即贡献密度。
6. "Evaluated on comprehensive datasets, our method achieves up to a 92.1% reduction across state-of-the-art SAT solvers."
   - **头牌数字句**：up-to + across-SOTA-solvers（收益上限 + 跨基线口径双重限定）。
7. "This demonstrates that a solver-aware modeling paradigm, which unifies structural mapping with SAT reasoning, can fundamentally enhance LEC efficiency."
   - **范式升格收口句**：demonstrates-that 把数字升格为范式结论；solver-aware modeling paradigm 是全文再命名。"fundamentally" 属判词（诚实边界排除，但 demonstrates-that + paradigm-命名 结构可学）。

## §1 Introduction

### P1（现状 + However 双挑战）

1. "LEC [18] proves that a golden design and its implementation compute identical Boolean functions."
   - 任务定义句带引用（与摘要句 1 分工：摘要讲痛点，正文讲定义）。
2. "The de facto LEC pipeline builds a miter by connecting the outputs of two designs through XOR gates and checks whether the miter can ever evaluate to 1."
   - **de facto 流水线描述句**：命名行业默认做法，一个 build+check 双动词走完整个 pipeline（为后文「换 pipeline 前半段」铺垫）。
3. "This turns equivalence checking into a single Boolean satisfiability (SAT) problem [6], and the approach has scaled thanks to modern SAT solvers [3,4,42] and solver heuristics [12,16,21,35]."
   - **先功劳句**：has-scaled-thanks-to——给现状记满功劳（7 个引用），为 However 蓄力。
4. "However, two challenges limit the robustness of such pipeline on complex designs."
   - **However 双挑战总起句**：预告数量（two），收窄范围（on complex designs）。
5. "First, logic synthesis (e.g. rewriting [27], resubstitution [26] and decomposition [23]) can significantly perturb logic cones and obscure matchable points between the miter's two sides."
   - 挑战一：**括注三算例点名**（rewriting/resubstitution/decomposition 各带引用）——痛点来源具体到操作名。
6. "The resulting lack of recognizable internal equivalences inhibits functional merging, a key simplification step shrinking the SAT instance within solvers [5,41]."
   - **后果链句 + 同位语机理层**：lack→inhibits 链条，同位语解释 functional merging 是什么、为何关键——痛点不止现象，还讲清它破坏了 solver 内部的哪个简化步骤。
7. "Second, many real-world designs contain XOR-dense modules, such as arithmetic units and cryptographic units."
   - 挑战二总起 + 两实例（arithmetic / cryptographic）。
8. "Unfortunately, modern SAT solvers relying on the search-backtracking algorithm [24] poorly handle constraints derived from XOR chains, deducing a literal typically requires almost all inputs fixed and terminating searching branch after verbose decisions [10,39,40]."
   - **Unfortunately 机理归因句**：把「为什么慢」讲成算法机理（deducing a literal requires almost all inputs fixed）而非现象描述——痛点的可解释性即合法性。

### P2（缺口 + 归因宣言 + 登场）

1. "Although prior efforts attempt to simplify the SAT instances before solving [14,33,37], operating solely on the flattened miters remains insufficient."
   - **Although-attempt-remains-insufficient 缺口句**：前作有功劳（attempt），但 remains-insufficient 点死穴（solely on flattened miters——层次不清才是问题）。
2. "We attribute that LEC efficiency is determined largely by how the miter is modeled before solving."
   - **归因宣言句**：We attribute + largely by how-clause——把领域归因重定向到自己的阵地。正文版 contends（摘要句 2），首尾呼应。
3. "Rather than simply constructing a conventional miter and hoping the solver compensates, we advocate a modeling strategy that preserves structural correspondence and exposes strong implications prior to SAT."
   - **Rather-than-hoping 对比句**：hoping-the-solver-compensates 轻贬现状路线（希望求解器补救），advocate 立论——对比句里嵌态度。
4. "In this work, we propose a novel mapping-based LEC formulation that constructs a simplified LUT-based miter (see Figure 1) instead of directly linking the two netlists at the bit level."
   - 登场句带图引用 + **instead-of 现状对比**（instead of directly linking at the bit level——把被替代做法写全）。
5. "By mapping both designs to LUT abstraction, our approach preserves more internal identical cones, makes XOR structure explicit, and reduces solving complexity."
   - **By-gerund 三收益连环句**：一个手段从句（By mapping）接三个动词收益（preserves / makes explicit / reduces）。
6. "We realize this agenda through three contributions:"
   - **agenda 承上启下句**：把前述主张命名为 agenda，引出 bullets。

### 贡献 bullets（3 条）

- "An equivalence-preserving LUT mapping algorithm that actively aligns corresponding subgraphs across the two designs, forming more internal equivalent logic..."
  - 品牌名 + that-从句机制 + **现在分词结果从句**（forming more...）。
- "A Gaussian-guided XOR modeling mechanism that detects XOR-dense area, implements with the LUTs representing XOR logic, and leverages Gaussian elimination to extract explicit affine relations."
  - 品牌名 + **三动词链**（detects / implements / leverages）——bullet 内部就是微流程。
- "A solver-oriented LUT selection criterion within the mapping algorithm that quantifies the solver-friendliness to minimize overall SAT solving difficulty."
  - 品牌名 + quantifies-X-to-minimize-Y（量化中间量压最终量）。

### P3（结果预览 + In-summary + 定位三连）

1. "We implement the proposed framework on top of a standard miter-based LEC flow and evaluate it using both structurally optimized and arithmetic-heavy benchmarks."
   - 实现句 + **双负载类型**（与 P1 双挑战一一对应：structurally optimized ↔ 挑战一，arithmetic-heavy ↔ 挑战二——评估设计呼应痛点设计）。
2. "Experimental results demonstrate that our complete framework achieves up to 92.1% reduction across solvers, with an 83.3% average improvement over the baseline under the PAR2 metric."
   - **up-to/avg 双口径头牌句**（RL-CCD 同款）：每指标成对（up to 92.1% + avg 83.3% PAR2）。
3. "Ablation results further show that the miter-aware mapping alone reduces runtime by 62.5%, while the Gaussian-guided XOR modeling mechanism alone yields a 36.8% reduction."
   - **消融预览句：alone×2 单组件归因数字**——摘要级消融预告，组件价值在 Intro 就有单独数字背书。
4. "Overall, our method can systematically reshape the problem representation—bridging LUT-mapping and SAT reasoning into a unified framework that delivers more compact CNFs and scalable equivalence checking performance."
   - Overall 收口 + 破折号 bridging 分词（判词 systematically/reshape 属诚实边界排除项）。
5. "In summary, this work introduces a solver-aware modeling paradigm for LEC that unifies miter-aware mapping, algebraic simplification, and solver-oriented LUT selection."
   - In-summary 范式复述句（三组件名第三次出现——命名贯穿）。
6. "Rather than introducing a new mapping framework, we embed SAT/LEC-specific insights into standard LUT mapping so the resulting abstraction is solver-friendly rather than merely structurally compact."
   - **Rather-than 边界句 + rather-than-merely 双层对比**：明确不做 X（新 mapping 框架）做 Y（把领域洞察嵌入标准流程）；Y 的价值再升一层（not merely compact but solver-friendly）。**诚实定位的教科书句**。
7. "This approach marks a significant departure from recent works [33,37] that apply LUT mapping to single-circuit SAT problems."
   - **marks-a-departure 定位句**：与最近同类工作的差异点一句话钉死。
8. "While their methods optimize an individual netlist, our core contribution is the joint mapping of both golden and implementation circuits."
   - **While-their-our 核心差异句**：先让步（their methods optimize...）后立论（our core contribution is...）——核心贡献以对比形式而非自夸形式给出。
9. "This strategy is designed to explicitly preserve and rediscover the internal equivalences often obscured by synthesis—a challenge unique to equivalence checking."
   - 破折号 **独有性收口**（a challenge unique to LEC——问题独占声明，把差异化上锁）。
10. "Consequently, the proposed framework not only improves practical verification efficiency but also forges a conceptual bridge between circuit optimization and SAT solving."
    - **Consequently not-only-but-also 双层收益收口**：实用层（效率）+ 概念层（bridge between 两个领域）。

## 入库范式（本轮新增）

- **contend-that + as-critical-as 论点宣言句**（摘要第二句即立场，不先讲方法）
- **de facto 流水线命名句**（build+check 双动词走完默认做法，为替换其前半段铺垫）
- **has-scaled-thanks-to 先功劳句**（现状记满功劳再 However）
- **However two-challenges 总起 + First/Second 括注算例点名**
- **后果链句 + 同位语机理层**（lack→inhibits + "a key simplification step shrinking..."——痛点讲清破坏了哪个内部步骤）
- **Unfortunately 机理归因句**（deducing-a-literal-requires——「为什么慢」讲到算法机理层）
- **Although-attempt-remains-insufficient 缺口句**（前作功劳 + solely-on-X 点死穴）
- **We-attribute + how-clause 归因宣言句**（领域归因重定向，与摘要 contends 首尾呼应）
- **Rather-than-hoping 对比句**（hoping-the-solver-compensates 轻贬现状 + advocate 立论）
- **登场句 instead-of 现状全写**（instead of directly linking...at the bit level）
- **By-gerund 三收益连环句**（一手段三动词收益）
- **bullet 品牌名 + 动词链/结果分词**（命名密度即贡献密度）
- **up-to/avg 双口径 + alone×2 消融预览**（Intro 组件级数字背书）
- **Rather-than-introducing 边界句 + rather-than-merely 价值升格**（不做 X 做嵌入 Y，Y 再升一层）
- **marks-a-departure + While-their-our 核心差异句**（差异化以对比形式给出）
- **a-challenge-unique-to 独有性收口**（问题上锁）
- **Consequently not-only-but-also 双层收口**（实用层+概念层）

诚实边界排除：fundamentally / significantly / novel / systematically reshape / marks-a-significant-departure（判词层）；我们只允许数字与结构事实做对比句的主干。

## 与我们手稿的对照

- 我们的 Related Work P1 收口句 "We study these existing components with complete low-frequency RoPE pairs shared across both stages and separate nominal near--far L1 page quotas, distinguishing page-level eligibility from final token capacity." 缺 **Rather-than 边界句** 前缀——本文最诚实的定位句式（不做新级联、组合已有组件）正对我们的论文事实。→ 本轮修改点。
- 摘要句 2 的「论点宣言句」对应我们 P1 尾句 "Both candidate selection and final attention-set size are therefore central design problems."——对等句式已在（selection 与 attention-set 双中心），保持。
- P3 双负载与双挑战一一对应：我们 Q1(LongBench 五任务)/Q2(RULER) 与引言观察的对应已建立，不重复。
- up-to/avg 双口径：我们无 improvement 数字可配对（诚实边界），不适用。

## 本轮修改（第 31 轮）

- en（Related Work P1 收口句加边界前缀）：
  `Rather than proposing a new cascade, we study these existing components with complete low-frequency RoPE pairs shared across both stages and separate nominal near--far L1 page quotas, distinguishing page-level eligibility from final token capacity.`
  （"Rather than introducing a new mapping framework, we embed..." 的诚实定位句式；我们的事实正是组合先例组件——AsyncTLS 级联 + Quest 极值，见同段上文，无新增主张。）
- zh 同步直译：「本文不提出新级联，而是研究将这些已有组成部分与……」
- 版式风险：+6 词，编译后须复验 en 仍 6 页（零余量已 6 次触发）。

## §2 Related Work（第 32 轮）

### §2.1 Logic Equivalence Checking（9 句）

1. "LEC is a cornerstone of formal verification in EDA, aiming to ensure that two circuits, typically before and after synthesis, implement the same Boolean function [20]."
   - **cornerstone 定性句 + aiming-to 任务定义**：与摘要分工（摘要讲痛点，正文讲地位）；括注 typical 场景（synthesis 前后）预埋后文的综合扰动。
2. "The standard approach is to build a miter circuit by XORing corresponding outputs and ORing all XOR results together."
   - 标准做法机制句：双动名词（XORing/ORing）一步写完构造。
3. "Unsatisfiability of the resulting miter indicates functional equivalence."
   - **判定语义句**：UNSAT ⟺ 等价，一句给 soundness——方法段的知识基线压缩成 5 词。
4. "Typically, when two designs share a high degree of structural correspondence, equivalence points (i.e., functionally identical node pairs) can often be detected, making the miter instance of LEC easy to solve."
   - **有利情形句**：when 条件 + i.e. 括注定义 + making-gerund 后果——先立「什么时候容易」再让 However 打碎它。
5. "However, aggressive logic synthesis disrupts the structural similarity [refs] but retains the hard-to-solve regions (i.e., XOR-dense regions) [refs]."
   - **However + disrupts-but-保留 双面伤害句**：摧毁有利项（结构相似）同时保留有害项（XOR 密集区）——一个 but 把两类痛点合并为单一因果源。修辞上比罗列两个缺点强一档。
6. "Recent efforts apply preprocessing to the flattened miter to simplify the SAT instance prior to solving [33,37], yet overlooking the inherent cross-design correlations between the golden and implementation sides."
   - **credit + yet-overlooking 盲区命名句**：前作动词完整给功劳，yet-gerund 点名盲区（cross-design correlations）——盲区即我们的战场。
7. "Another line of research combines multiple engines, such as simulation-guided engine [13,28] or non-SAT alternatives [30], to solve the miter instance."
   - 第二路线枚举句（another line + such-as 双例）。
8. "While effective on certain classes of designs, these methods still rely heavily on favorable problem features and initial decisions."
   - **While-effective 让步 + rely-heavily-on 依赖性限制**：夸一句再点「依赖运气/初始决策」的软肋。
9. "In contrast to operating solely on a flattened miter, our work targets the initial modeling stage of LEC, reconstructing similar structure and neutralizing hard-to-solve regions before SAT solving."
   - **In-contrast-to + targets + 双动名词定位收口**：对比轴显式命名（solely on a flattened miter）；reconstructing/neutralizing 与 Intro 双挑战 1:1 映射——收口即回到自己的主战场。

### §2.2 Look-up Table (LUT) Mapping（11 句）

1. "LUT mapping is a fundamental step in logic synthesis, where a Boolean network is decomposed into a network of k-input LUTs for FPGA [8]."
   - 第二社区背景定义句（给 LUT 社区读者的 anchor）。
2. "In a typical mapping procedure, the circuit is traversed in an output-to-input direction."
   - 流程定向句（遍历方向一短语交代）。
3. "For each node, multiple candidate LUTs are enumerated based on feasible input cuts and their corresponding Boolean functions, and the most suitable one is selected to optimize delay and latency."
   - **枚举-后-选择机制句**：enumerate candidates → select the most suitable——与我们两级筛选同构；For-each-node 逐节点视角。
4. "Recently, it has also attracted attention as a structure-preserving abstraction for SAT solving, where multiple logic gates are packed into solver-friendly LUTs to reduce CNF size and improve clause locality [33,37]."
   - **Recently 转用句**：经典用途→新用途一步转向；where-从句机制 + to-不定式双收益（reduce size / improve locality）。
5. "Each LUT encapsulates a small truth table, offering flexibility for customized encoding strategies and fine-grained control over structural granularity."
   - 性质-可供性句（encapsulates + offering flexibility）。
6. "The SAT-solving advantage of LUT-based abstraction arises from its ability to preserve local functional coherence [5,33]."
   - **段首机制断言句**：advantage-arises-from-ability-to——段落 topic sentence 直接是机理主张（不是背景复述）。
7. "By grouping strongly related logic signals into a single LUT, the circuit representation becomes more compact and semantically cohesive, which helps remove redundant variables and clauses in the resulting CNF."
   - By-gerund 因果链 + which-helps 后果（紧接到具体 CNF 层）。
8. "Moreover, each LUT inherently defines a deterministic mapping between its inputs and output, enabling stronger inferences once partial assignments are known."
   - 第二机制 + enabling-gerund + **once-条件从句**（部分赋值已知时推理更强——机制的时间触发点写明）。
9. "This leads to more concentrated propagation, shorter implication chains, and overall improvement in solver efficiency."
   - **This-leads-to 三联结果收口**：concentrated propagation / shorter chains / solver efficiency——从微观到宏观三层。
10. "However, existing approaches primarily target single-circuit optimization and overlook miter-specific structural patterns that arise in equivalence checking."
    - However 缺口句：target-X-and-overlook-Y 并列双动词。
11. "Our work extends this line of research by performing miter-aware, solver-friendly LUT mapping, jointly considering structural alignment, XOR reasoning, and CNF-level propagation properties."
    - **extends-this-line-by-performing 收口**：extends（非 replaces——给前作留地位）+ jointly-considering 三要素——Related Work 两个小节都以定位收口，且动词不同（§2.1 用 In-contrast-to、§2.2 用 extends）——**收口动词按与前作的关系亲疏选型**。

### 入库范式（本轮新增）

- **disrupts-but-retains 双面伤害句**（一个 but 合并两类痛点为单一因果源）
- **yet-overlooking 盲区命名句**（credit 完整给足 + 盲区即自己战场）
- **While-effective + rely-heavily-on 让步限制句**
- **In-contrast-to + 双动名词定位收口**（对比轴显式命名；动名词与 Intro 挑战 1:1）
- **Recently 转用句**（经典用途→新用途转向 + to-不定式双收益）
- **段首机制断言句**（advantage-arises-from-ability-to；topic sentence 即机理主张）
- **once-条件从句**（机制的时间触发点写明）
- **This-leads-to 三联结果收口**（微观→宏观三层）
- **收口动词按关系亲疏选型**（In-contrast-to 用于批判性距离 / extends 用于继承性延伸）

诚实边界排除：cornerstone（判词但属领域公认定性，我们可容忍同等地位的公认定性，不自创新判词）。

## 本轮修改（第 32 轮）

- en（Related Work P2 对比轴显式化，In-contrast-to 句式）：
  旧：`Our selector instead ranks candidates using a fixed coordinate subset under page and token budgets; original-dimensional KV remain available for final attention.`
  新：`In contrast to pruning token transfers from partial key bits, our selector ranks candidates with a fixed coordinate subset under page and token budgets, keeping original-dimensional KV for final attention.`
  （Miter §2.1 收口句式：对比轴显式命名——旧版 "instead" 的对比对象（按部分键比特剪枝）隐没在上文，新版点名；keeping-gerund 收尾替代分号并列。）
- zh 同步直译。
