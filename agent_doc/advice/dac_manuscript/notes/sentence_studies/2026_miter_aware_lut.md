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

## §3 Methodology（第 33 轮）

### §3.1 Overview（4 句）

1. "Given the golden design (G) and implementation design (I) in an LEC problem, our framework formulates it into an easy-to-solve SAT instance."
   - Given-输入-输出 形式化开场句（RL-CCD Given-Goal-such-that 同族，此处更短）。
2. "As shown in Figure 1, our framework targets three objectives:"
   - 图引用 + 三目标总起（编号列举引出）。
3-5. (1) Preserve structural correspondence..., (2) Gaussian-guided XOR modeling..., (3) Select solver-friendly LUTs...
   - **三目标编号句**：每条 = 品牌动词短语 + to-不定式/分词收益从句。三目标与图 1 三组件严格 1:1（图号锚）。

### §3.2 Equivalence-preserving LUT Mapping（关键 8 句）

1. "To mitigate the synthesis-induced structural perturbations, we introduce a miter-aware LUT mapping scheme (Figure 2) that explicitly reconstructs structural correspondence between the two sides of the miter."
   - **To-mitigate-X 使命开场句**：小节第一句 = 针对哪一个 Intro 痛点（痛点回指）+ that-explicitly-reconstructs 机制。小节标题即品牌名，首句即痛点绑定。
2. "In a typical mapping procedure, the circuit is traversed in an output-to-input direction; for each node, multiple candidate LUTs are enumerated from feasible input cuts..., and the most suitable one is selected to optimize delay."
   - 标准流程复述句（复用 §2.2 的流程句，为 Building-upon 铺垫——**Related Work 里的流程句在 Methodology 原样复用，跨节零漂移**）。
3. "Building upon this standard process, our mapping identifies XOR/XNOR structures near the primary outputs as natural comparison anchors between the golden and implementation circuits."
   - **Building-upon 继承句**：标准流程→我们的增量（identifies X as Y 命名发现物——"natural comparison anchors" 是给观察起的名字）。
4. "The two fan-in nodes of these XOR gates serve as the respective boundaries of the two sides, providing an approximate yet practical alignment even after aggressive synthesis."
   - 机制细化 + **approximate-yet-practical 诚实降格**：承认近似但给出适用条件（even after aggressive synthesis）——限制内声明价值。
5. "Starting from these boundary anchors, our mapper proceeds recursively to generate and match LUT candidates on both sides."
   - Starting-from 递归机制句（动词链 generate-and-match）。
6. "This similarity is measured using a heuristic that combines topological depth, fan-in/fan-out counts, and the functional hash of the immediate logic cone."
   - **三要素启发式披露句**：相似度度量三成分全点名——启发式不藏着，成分透明即可复现。
7. "While SAT-sweeping detects equivalences via repeated SAT queries, our alignment provides a lightweight, orthogonal structural matching that complements such solver-based techniques without their computational cost."
   - **While-前作-our-轻量互补句**：不贬前作（complements），差异维度三连（lightweight / orthogonal / without their cost）。
8. "It is worth noting that LUT mapping inherently absorbs a group of primitive gates into each LUT, leaving only the LUT's input and output variables exposed in the final CNF."
   - **It-is-worth-noting 机制显影句**：absorbs/leaving-exposed 一对动词把「映射为什么天然有利于 CNF」讲到变量可见性层。
9. "Consequently, variables corresponding to internal gates disappear once absorbed."
   - Consequently 单句后果（短句节奏）。
10. "When mapping is guided to preserve equivalence points as LUT input or output boundaries, these equivalence variables remain explicit in the CNF and can be more effectively exploited by the solver's redundancy elimination mechanisms."
    - When-引导条件收益句（机制→收益，条件触发点显式）。
11. "Moreover, equivalence points located closer to the primary outputs enable earlier clause subsumption and stronger pruning within the search space, amplifying solver efficiency."
    - Moreover 位置梯度收益（closer→earlier→stronger 三级递进）。
12. "If no direct correspondence can be found for a node pair—e.g., due to heavy resubstitution or decomposition—we independently select LUTs on each side according to the solver-friendliness metric introduced in Sec 3.4, maintaining comparable structural granularity across designs."
    - **失败分支句**：If-no-X—e.g. 归因—回退策略 + 前向引用（Sec 3.4）。方法的鲁棒性 = 把找不到对应的情形写全。
13. "The resulting LUT network thus achieves improved symmetry, enhanced clause reuse, and stronger implication propagation, effectively bridging structural matching and solver-oriented optimization."
    - Thus 三联收益收口 + bridging 收束语（对称/复用/传播——微观到宏观）。

### §3.3 Gaussian-guided XOR Modeling（关键 6 句）

1. "XOR-dominated circuits—such as arithmetic and cryptographic designs—pose major challenges to CDCL-based SAT solving, since their CNF encodings often require a large number of clauses and provide weak unit propagation."
   - **痛点-机理 since 句**：现象（major challenges）+ since 归因（clauses 多 + unit propagation 弱）——「为什么难」先于「怎么办」。
2. "To this end, we first detect XOR relations within the AIG network and perform Gaussian elimination over the finite field GF(2) within XOR-dense regions."
   - To-this-end 双动词机制句（detect + perform）。
3. "Each XOR gate represents an equation among variables."
   - 一句语义桥（电路→代数翻译规则，为公式铺路）。
4. "To construct equations with constant right-hand sides, we leverage simple transformations: [两条变换规则]"
   - **冒号引出变换规则 bullet**（数学写法：规则直接列，不写成散文）。
5. "For example, given the XOR relations: [4 个方程] performing elementary row transformations yields: [4 个化简后方程]"
   - **4×4 实例演算句**：For example + 具体方程组 + yields 结果——Gauss 消元用 8 行实际数字演示，不是抽象符号。
6. "Unlike conventional elimination that substitutes away variables, we selectively retain XOR-related LUTs whose variables participate in nontrivial affine relations discovered by Gaussian elimination."
   - **Unlike-conventional 差异机制句**：替代做法一句概括（substitutes away）+ 我们保留什么（retain X whose...）。
7. "These LUTs are explicitly reconstructed as standalone XORs rather than merged into larger logic cones, ensuring that their participating variables remain visible and semantically linked in the final representation."
   - **rather-than-merged + ensuring-可见性**：设计决策的判据是「变量可见性」——与 §3.2 的 absorbs/leaving-exposed 同一机理轴，跨小节呼应。
8. "For circuit-based solvers, this explicit construction enables early identification of XOR structures and reduces intermediate branching and propagation effort. For CNF-based solvers, the preserved XOR LUTs naturally translate into small XOR substructures during CNF conversion..."
   - **For-X-For-Y 双受众收益句**：同一机制对两类求解器各讲一遍收益（受众分治）。
9. "Overall, our approach integrates Gaussian-derived affine relations into the circuit as explicit XOR logic, ensuring their visibility across abstraction levels and improving both propagation and clause-level simplification."
   - Overall 收口句（visibility 主题第三次出现——小节主题词贯穿）。

### §3.4 Solver-oriented LUT Selection（关键 7 句）

1. "For regions not dominated by XOR structures, we perform LUT implementations guided by a solver-friendliness metric."
   - **范围互补开场句**：For-regions-not-X——与前一小节划清分工边界（哪里不用这个机制）。
2. "1) Branching efficiency: These metrics capture how easily the solver can explore and prune the search space around L."
   - **大类定义句**：编号大类 + These-metrics-capture-how（度量族的总括从句）。
3. "(a) Branch count: We measure how compactly L partitions the input space..."
   - 子度量定义句（We-measure-how 开头——操作定义先于公式）。
4. "Formally, a branch is a pattern p∈{0,1,x}^|I| (where x denotes 'don't care') such that every assignment covered by p produces the same output."
   - **Formally 定义句**：口语定义（measure how compactly）→Formally 精确化（pattern + such-that）——先直觉后形式的双层定义。
5. "Intuitively, a smaller branch count indicates...implying that decisions on input literals more rapidly constrain the search space. From the SAT-solving perspective, such functions yield simpler decision trees, shorter BCP chains, and fewer case splits during clause learning, thus improving solver convergence."
   - **Intuitively / From-the-X-perspective 双视角解释**：公式后必跟直觉句 + 求解器视角句——**每个度量都是三段式：定义公式→Intuitively 直觉→求解器收益**。
6. "For example, a 2-input AND can be covered by three branches (see Table 1), while XOR requires four fully specified minterms, hence AND yields a lower f_branch and is considered more solver-friendly."
   - **For-example + hence 判例句**：AND vs XOR 用真值表数字裁决（3 vs 4 分支）——度量立即落在一个具体对比上。
7. "Lower f_sensitivity indicates locally stable behavior and typically leads to a narrower effective branching space for CDCL-based solvers."
   - **方向句模板：Lower/Higher-X indicates-Y and leads-to-Z**（每个度量的解释句同构——度量族的可读性来自同构句式）。
8. "Complexity and reusability: Although metrics like DetCount require enumerating variable subsets and partial assignments, the cost is negligible since only 2/3/4-input LUTs are used. All scores are precomputed offline (in five seconds) and cached for reuse, introducing no overhead in synthesis or solving."
   - **复杂度诚实披露段**：Although-看似贵-since-实际小 + 具体数字（five seconds）+ no-overhead 收口——开销问题主动前置处理。
9. "Cost aggregation and weight learning: The overall LUT cost combines each metric with Bayesian-optimized weight, tuned on a small validation set to minimize solving time."
   - 聚合公式段（加权 + 调参目标一句话）。
10. "While the weight is tuned on a subset, the results are robust across all validation sets."
    - **While-调参范围-鲁棒性声明**：诚实披露调参在子集 + 鲁棒性声明（潜在过拟合质疑前置消解）。
11. "Their relative emphasis (higher f_det for SAT, higher f_branch for UNSAT) aligns with the distinct behaviors of satisfaction and proof-oriented solvers."
    - **权重结果→机理回读句**：学到的权重本身成为发现（SAT/UNSAT 求解器偏好不同度量）——副产品升格为 insight。

### 入库范式（本轮新增）

- **To-mitigate-X 使命开场句**（小节首句=痛点回指+机制，标题即品牌名首句即绑定）
- **Building-upon 继承句 + identifies-X-as-Y 发现物命名**（natural comparison anchors）
- **approximate-yet-practical 诚实降格句**（承认近似+适用条件）
- **三要素启发式披露句**（成分全点名，透明即复现）
- **While-前作-our-轻量互补句**（complements 不贬前作+三差异维度）
- **It-is-worth-noting 机制显影句**（absorbs/leaving-exposed 变量可见性动词对）
- **If-no-X 失败分支句**（e.g. 归因+回退策略+前向引用——鲁棒性=写全找不到的情形）
- **痛点-机理 since 句**（「为什么难」先于「怎么办」）
- **For-example + hence 判例句**（AND vs XOR 真值表数字裁决度量）
- **度量三段式模板**（定义公式→Intuitively 直觉→求解器收益；每度量同构）
- **方向句模板**（Lower/Higher-X indicates-Y and leads-to-Z）
- **For-X-For-Y 双受众收益句**（同一机制对两类受众各讲收益）
- **复杂度诚实披露段**（Although-since+具体秒数+no-overhead）
- **While-调参-鲁棒性声明**（过拟合质疑前置消解）
- **权重结果→机理回读句**（学到的权重升格为 insight）
- **跨节零漂移**（Related Work 流程句在 Methodology 原样复用）

诚实边界排除：easy-to-solve（判词——我们的对应表述是"降低求解难度"级别且须绑定数字）；amplifying/naturally 等弱判词 tolerate 但不自加。

## 与我们手稿的对照

- §3.2 句 1 的「To-mitigate-X 使命开场句」：我们 §III 小节首句多为机制陈述（"Both regions and both selection stages use the same coordinate map"），缺痛点回指——但该模式在 §III-B（regional budgets）已有等价物（"Regional page caps and token quotas constrain the cascade"），不硬套。
- §3.4 度量族三段式（定义→Intuitively→收益）：我们 §III-B 的 U_b 段结构（定义→bound 声明→loose 警告）是诚实版三段式，已有。
- **For-example + hence 判例句**是我们缺失且可用的：§III-B 的 β 整数取整规则（"Raising β cannot reduce the nominal near-page quota...rounding can leave both unchanged"）后可加数值判例句。→ 本轮修改点：β 规则句后补一句 16→8 页数值判例（β=.125→.0625 时 16→8 页、112→120 页——K1=128 已有下文数字，句内判例立即落地；无新数字，仅复用下文既有数值做即时例证）。
- While-调参-鲁棒性声明、权重回读：无对应实验，不适用。

## 本轮修改（第 33 轮）

- en（§III-B β 规则句后加判例句，For-example+hence 句式）：
  在 "For $K_1=128$ and $\beta=.125$, L1 reserves 16 near and 112 far pages." 后已有数字例——改为紧邻 Raising-β 句的判例形式：
  `For example, $K_1=128$ halves the near-page quota from 16 to 8 when $\beta$ drops from $.125$ to $.0625$, so rounding leaves far pages 112$\rightarrow$120; at $\beta=.0625$ the far-page quota rises despite the unchanged cap.`
  落位：替换原 "For $K_1=128$ and $\beta=.125$, L1 reserves 16 near and 112 far pages."（判例句把静态数字例升级为方向判例；16/8/112/120 全部为下文 Q2 已报数字与 round 规则的算术结果，无新数字。）
- zh 同步直译。

## §4 Experiments + §5 Conclusion（第 34 轮，第 7 篇收官）

### §4.1 Experiment Setting（5 句）

1. "We construct a comprehensive benchmark suite to evaluate the efficiency of our proposed LEC formulation framework."
   - 基准构造开场句（We construct——评测资产本身是贡献姿态）。
2. "This suite includes the circuit pairs derived from industrial LEC problem and the instances built from public sources: ForgeEDA, ITC99, EPFL, and Opencore."
   - **工业+公开双源枚举句**：来源分层（industrial / public 四套件点名）。
3. "To build the equivalent circuit pairs, each circuit is synthesized and conjunctive with the original one to form a miter. To build the inequivalent circuit pairs, we introduce small functional perturbations by randomly modifying a few nodes in the synthesized AIGERs before miter construction."
   - **成对构造披露句×2**：等价对与不等价对的构造法各一句——正负两类测试都交代生成方式（防「只测容易的」质疑）。
4. "All circuits are synthesized with ABC, and LUT networks are generated using a customized cost-driven mapper built on Mockturtle, restricted to 2-, 3-, and 4-input LUTs, since larger LUTs (i.e., ≥5 inputs) would produce exponentially more complex CNF encodings (2^n clauses) and thus hinder subsequent SAT solving."
   - **工具链披露 + 限制带原因句**：restricted-to-X **since**-Y-would-hinder-Z——设计限制不是拍脑袋，有机理（2^n 子句）。
5. "Each instance runs on a single Intel Xeon Platinum 8375C core with a 3,600 seconds timeout."
   - 硬件/超时单句披露（RL-CCD 环境四件套的极简版）。

### §4.2 Main Results（关键 7 句）

1. "To validate the effectiveness of our proposed framework, we compare the solving time of LEC instances formulated by our framework (Ours) and the baseline flow (Baseline)."
   - 验证目的开场句（To-validate + 双臂命名绑定 Ours/Baseline）。
2. "The performance is measured using average PAR2 score, where any instance failing the 3,600s timeout is penalized with 7,200s."
   - **指标定义带惩罚规则**：where-从句写明超时怎么罚（指标的可复现性在罚则里）。
3. "The PAR2 score for 'Ours' includes both solving time and LUT mapping runtime, where the formulation overhead accounts for < 10%."
   - **开销计入声明句（本文最诚实的一句）**：报告数字包含自己方法的 formulation 开销，且给出占比上限——「收益不是靠不计成本换来的」一句话锁死。对应我们「完整请求成本含准备维护选择」主张——他们有 <10% 数字，我们尚无，这是差距也是目标。
4. "Besides, we also record the number of solved instance within 3,600s time limit as #Solved."
   - 次指标补充句（#Solved 命名绑定）。
5. "After formulating SAT instance..., we employ two state-of-the-art CNF-based SAT solvers (Kissat and CaDiCaL), two circuit-based SAT solvers (X-SAT and CSAT) and an advanced LEC engine (command cec in ABC, denoted as CEC in Table 2), respectively."
   - **五引擎三分类枚举**：括注全名+表内简称绑定。
6. "Since CEC constructs the miter and translates it to CNF before calling solver, our approach is seamlessly inserted as a plugin-like pre-CNF optimization layer."
   - **Since-机制 + 插件式集成定位句**：与现有工具的关系一句话（plugin-like pre-CNF layer）——集成兼容性写成位置声明。
7. "The results are summarized in Table 2, from which three key observations emerge. First, ... This is most dramatic with the CaDiCaL solver, where our approach reduces the PAR2 score by nearly an order of magnitude (from 856.6 to 67.4). Second, ... We attribute this to our framework's ability to leverage circuit-based methods for effective elimination—optimizations that CNF-based solvers cannot perform on their own. Third, our framework still provides a 66.49% PAR2 reduction when used with the advanced CEC engine. Therefore, our approach is fully compatible with existing LEC tools, acting as a powerful pre-processing step that provides additional simplification, logic packing, and solver-friendly reformulation."
   - **「from which three key observations emerge」+ First/Second/Third 观察枚举**：表格解读结构化；每条观察 = 总括 + 最极端锚点（most dramatic + from 856.6 to 67.4 双数字）/ We-attribute-this-to 机理归因 / still-兼容性收口 + Therefore 三联收益收束。**Third 用 still**（连 LEC 引擎都能增益——弱场合仍成立是兼容性证据）。

### §4.3 Ablation Studies（关键 9 句）

1. "We further perform ablation studies to assess the effectiveness of the three key components in our framework using Kissat, the strongest solver among those tested."
   - 消融总起句：三组件预告 + **求解器选择带理由**（the strongest——最难场合做消融）。
2. "To assess the impact of our equivalence-preserving LUT mapping, we conduct an ablation study comparing the configurations with (Ours.) or without (Ours.w/o Equiv) the proposed method."
   - **with/without 命名约定句**：臂名直接编码消融变量（Ours.w/o Equiv）。
3. "Besides, we also evaluate the configurations with (w CCC) or without (w/o CCC) Clausal Congruence Closure."
   - 第二消融轴引入（外部技术 CCC 的有/无）——**双轴消融 = 2×2 四设置**，证明与后续 CNF 级优化互补。
4. "We include such comparison to demonstrate our LUT mapping provides benefits that are complementary to the subsequent CNF-based optimizations."
   - **比较目的前置声明句**：为什么加第二轴——给互补性铺路（先说目的再给四设置）。
5. "Here, Clausal Congruence Closure serves as a CNF-level circuit simplification technique... In contrast, our equivalence-preserving mapping operates at the circuit level during LUT decomposition..."
   - **对照技术定位句对**：CCC 在 CNF 级 / 我们在电路级——**层次正交性**是互补性的论据。
6. "We include both with and without CCC settings to demonstrate that our method is complementary to CNF-level simplification—our mapping enhances structural regularity before CNF translation, while CCC further eliminates redundancy after translation."
   - **before/after 阶段互补句**（重复强调——两处出现，跨段呼应）。
7. "Comparing Ours.w/o Equiv w/o CCC and Ours.w/o Equiv w CCC shows that CCC alone contributes a noticeable performance gain... However, when our equivalence-preserving mapping is enabled (Ours.w/o CCC), an even larger improvement is observed, indicating that preserving equivalence alignment early at the circuit level leads to substantially better clause reuse and stronger implication propagation."
   - **成对走读 + However + indicating-that 归因收口**：指定比较哪两臂→结果→However 转折（我们的更大）→indicating-that 机理。
8. "While we observe an average 5.3% PAR2 reduction across the entire dataset by using Gaussian-guided XOR modeling, the gain is especially significant on XOR-dense datapath circuits. Therefore, we perform a deeper analysis on 20 circuits constructed from 32×32 multipliers implemented with diverse architectures."
   - **While-平均-深钻转折句（本轮最可移植范式）**：全集平均 modest → 子集显著 → Therefore 深钻 20 例。平均掩盖异质性时，深钻就是第二层证据。
9. "These multipliers pose one of the most challenging verification tasks [19] due to their deep and XOR-dense structures."
   - **子集难度合法性句**：深钻对象为什么选它——最难+机理（due to）。
10. "First, a large improvement on solving efficiency is seen when our method is applied... With our method, all 20 XOR-dense instances are solved within 2,000 seconds. In contrast, the configuration without our Gaussian modeling solves only 7 instances by the 2,000s mark and 19 instances by 3,600s."
    - **all-20 vs only-7 时间检查点计数对比**：两臂在两个时刻的解决数——分辨率比均值高一级。
11. "Second, the two techniques are not conflicting, as they operate at different stages of the flow and can be applied simultaneously—Ours.w CCC achieves the best overall performance, yielding an average speedup of 36.8%."
    - **不冲突声明句**：as-原因（不同阶段）+ 破折号最优组合数字——「能同时用」本身是结果。
12. "To evaluate the effectiveness of our proposed solver-friendliness metric, we isolate it from the other two components and activate only the metric-guided LUT selection module (denoted as Ours.metric)."
    - **隔离声明句**：we-isolate-it-from——单组件评估的纯净性交代。
13. "Two baselines are included for comparison: Comp.1, from prior work [37], which employs a branching-based LUT cost function emphasizing structural complexity. Comp.2, from prior work [14], which uses area-oriented cost functions..."
    - **外部基线带出处与机制描述**：每个基线 = 出处引用 + 一句话机制（比什么、差异在哪）。
14. "As shown in Figure 4, Ours.metric consistently outperforms both Comp.1 and Comp.2 across the runtime range. Compared with Comp.1, it solves 5.5% more instances within 3,600 seconds and yields an overall 34.1% average speedup. These results demonstrate that our metric effectively captures solver-oriented behaviors beyond structural complexity, leading to a more solver-efficient formulation."
    - outperforms + 双数字 + **These-results-demonstrate-that 收口**（结果→能力声明→leading-to 最终收益）。

### §5 Conclusion（3 句）

1. "This paper presented a mapping-based modeling framework for Logic Equivalence Checking."
   - 一句总结句（presented 过去时，框架名）。
2. "By integrating equivalence-preserving LUT mapping, Gaussian-guided XOR modeling, and solver-oriented LUT selection into a unified flow, our method formulates the miter into a structurally symmetric, algebraically simplified, and solver-efficient form."
   - **By-integrating 三组件 + 三形容词形式收口**：三组件名第四次全文贯穿；形式三形容词与三组件一一对应（对称↔映射、代数↔高斯、高效↔选择）。
3. "Experiments on comprehensive benchmarks confirm substantial and consistent improvements across multiple solver paradigms, demonstrating that modeling formulation can markedly enhance LEC efficiency and bridge the gap between logic synthesis and SAT-based formal verification."
   - **experiments-confirm + demonstrating-that 论点回读 + bridge-the-gap 收口**：结论最后一句回到摘要 contends 的同一主张（建模与求解器同等重要）+ 呼应 Intro 尾句 "forges a conceptual bridge"——**首尾三重闭环**（摘要论点→Intro 概念桥→结论 bridge-the-gap）。"substantial/consistent/markedly" 属判词（诚实边界排除；我们的结论用数字复述替代）。

### 入库范式（本轮新增）

- **成对构造披露句**（等价对+不等文对各自的生成法——正负测试都交代）
- **限制带原因句**（restricted-to-X since-Y-would-hinder-Z，机理级理由）
- **开销计入声明句**（报告数字含自身 formulation 开销 + 占比上限 <10%——收益不靠隐藏成本）
- **指标定义带惩罚规则**（超时罚则写进 where-从句）
- **插件式集成定位句**（Since-机制 + plugin-like pre-CNF layer——兼容性写成位置声明）
- **「from which three key observations emerge」+ First/Second/Third 观察枚举**（每条=总括+最极端锚点/机理归因/still-弱场合）
- **most-dramatic + from-to 双数字锚**（856.6→67.4）
- **Third-still 兼容性证据句**（弱场合仍成立）
- **with/without 命名约定**（臂名编码消融变量）+ **双轴 2×2 消融设计**
- **比较目的前置声明句**（为什么加第二轴先说）
- **层次正交性定位句对**（CCC 在 CNF 级 / 我们在电路级）
- **成对走读 + However + indicating-that 归因收口**
- **While-平均-深钻转折句**（全集平均 modest→子集显著→Therefore 深钻——平均掩盖异质性时深钻是第二层证据）
- **子集难度合法性句**（深钻对象=最难+due-to 机理）
- **all-N vs only-M 时间检查点计数对比**（两个时刻两臂解决数，分辨率高于均值）
- **不冲突声明句**（as-不同阶段 + 最优组合数字——「能同时用」是结果）
- **隔离声明句**（we-isolate-it-from 单组件纯净性）
- **外部基线带出处+机制描述**（每基线=引用+一句话差异）
- **These-results-demonstrate-that 收口**（结果→能力→最终收益三连）
- **结论三句式**（一句总结 / By-integrating 三组件+三形容词一一对应 / experiments-confirm+bridge-the-gap 论点回读——首尾三重闭环）

诚实边界排除：substantial/consistent/markedly/dramatic（判词）；我们的结论以数字复述替代判词（已达成）；「开销计入声明」是我们的目标差距——需完整请求测量给出占比后才能写同款句。

## 与我们手稿的对照（收官总结）

- **开销计入声明句**是我们与本文最大的诚实姿态差距：他们写 "includes both solving time and LUT mapping runtime, where the formulation overhead accounts for <10%"；我们只能说 "still requires matched quality and complete-request cost"——目标一致，缺的是那个 <10% 级别的数字。这是投稿后第一优先实验（完整请求 E2E 计时含准备维护选择）。
- **While-平均-深钻转折句**：我们门控段有现成素材（AVG 降 .034/.218 vs HotpotQA 降 1.05/1.26）但写成三个平列句。→ 本轮修改点。
- 结论三句式：我们结论段已有数字复述版（β 0.99 分、58.96、1.58 分），比本文判词版更符合诚实边界；By-integrating 三组件收口句可对照检查（我们结论首句 "We studied shared low-frequency page/token scoring and regional allocation" 已是等价物）。不硬套。

## 本轮修改（第 34 轮）

- en（门控段 While-平均-深钻转折句，合并三句为两句）：
  旧：`...decreases of .034 and .218 points. AVG is the primary quality measure. Three tasks decline in each arm, including HotpotQA by 1.05 and 1.26 points; these differences describe heterogeneity, not a per-task zero-loss requirement.`
  新：`...decreases of .034 and .218 points. While the AVG decreases are small, three tasks decline in each arm, including HotpotQA by 1.05 and 1.26 points; AVG remains the primary quality measure, and these differences describe heterogeneity, not a per-task zero-loss requirement.`
  （Miter §4.3.2 "While we observe an average 5.3%...the gain is especially significant on XOR-dense datapath circuits" 句式：平均口径与逐任务口径在同一 While 从句里对撞，异质性不用额外解释句；"small" 由句内 .034/.218 vs 1.05/1.26 四个数字锚定。AVG 主判据声明保留。）
- zh 同步直译。
