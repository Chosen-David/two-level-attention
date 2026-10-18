# Gamora (DAC 2023 Best Paper) 逐句精读笔记 — Abstract + Introduction

> 2026-10-18 第 22 轮逐句学习，直接读 PDF 原文（`2023_Gamora.pdf`，DAC'23，UCSB + Utah + GT + NVIDIA + 阿里达摩院）。
> Gamora 是「学习方法替代传统算法」类 best paper（GNN+GPU 符号推理替代结构哈希/功能传播），观察其摘要的编号式双口径证据与前作三分法局限枚举。

## Abstract（6 句）

| # | 原文（缩写） | 功能 |
|---|---|---|
| A1 | "Reasoning high-level abstractions from bit-blasted Boolean networks ... can significantly benefit functional verification, logic minimization, datapath synthesis, malicious logic identification, etc." | **stakes 多域枚举句**（四个下游应用域枚举——benefit 是潜力式） |
| A2 | "Mostly, conventional reasoning approaches leverage structural hashing and functional propagation, **suffering from** limited scalability and inefficient usage of modern computing power." | **前作 suffering-from 双缺口句**（一类方法一个分词缺口：可扩展性+算力利用） |
| A3 | "In response, we propose a novel symbolic reasoning framework exploiting GNNs and GPU acceleration ..., namely GAMORA, which offers high reasoning performance w.r.t. exact reasoning algorithms, strong scalability ..., and generalization capability ..." | **In-response 登场句**（回应 A2 缺口+命名+三能力枚举——与 suffering 缺口逐点对应） |
| A4 | "To further demonstrate the capability of GAMORA, we also evaluate its reasoning performance **after various technology mapping options, since** technology-dependent optimizations are **known to make** functional reasoning much more challenging." | **压力测试预告句带 since 归因**（扩展评估+公开难度声明——主动把最难场景纳入评估范围） |
| A5 | "Experimental results show that (1) GAMORA reaches almost 100% and over 97% ... **with up to six orders of magnitude speedups compared to** the state-of-the-art implementation in the ABC framework; (2) GAMORA maintains high reasoning accuracy (>92%) ... **and we comprehensively analyze the impacts** ..." | **编号双口径证据句**（(1) 头牌质量+速度对+点名基线 ABC；(2) 抗压质量+分析贡献——质量/速度/压力三证据编号打包） |
| A6 | "GAMORA is available at https://github.com/Yu-Utah/Gamora." | 开源披露句 |

对比：A5 编号双口径（质量门先行+速度对+压力测试+分析承诺）是 Token-Picker/AdaptivFloat 头牌句的又一变体；我们的摘要第 4-5 句（58.30/59.29 + 50.64/52.22 + 边界句）是诚实镜像。A4 压力测试预告句是新模式：**预告将主动评估已知更难的场景**——我们的对应物：摘要「Separate historical timings show slower sparse decoding」与 Q2 128K 失败案例（把已知更差/更难的证据主动放进摘要与评估），功能等价。

## Introduction

### P1（3 句）：应用域引用链 + 量化背景句
- 应用句带引用簇（verification 效率 [7][16] + 恶意逻辑检测 [3][14]——双应用各挂引文）。
- **时代定位句带量化背景**（"In the era of globalization and democratization of IC development ..., such reasoning is expected to bring broader impacts on hardware security, which is at the heart of modern computing systems: **more than 40 percent** of FPGA/ASIC projects are working under safety-critical ... standards [9]"——宏观趋势句+冒号内嵌行业量化数字；引用支撑）。

### P2（4 句）：机制难点三连——优化破坏+结构复杂+规模爆炸
- **优化破坏句**（"Due to the optimization conducted by RTL synthesis tools, reasoning ... is extremely challenging, **since** hierarchy and module information is lost during multi-level logic minimization and technology mapping, **which is also complicated by** functional blocks overlapping and gate sharing"——三层嵌套归因：工具优化→信息丢失→结构复杂化）。
- **规模递进句**（"The problem goes further due to the explosion in runtime for large-scale BNs"——goes further 链式递进）。
- 前作两分类枚举句（structural: shape hashing/reference libraries；functional: cut enumeration [10][20]）→ 组合句（[15][20][26] more prevalent）。

### P3（2 句 + 编号三点）：Despite 功劳 + 编号局限枚举
- **Despite 功劳让步句**（"Despite the achieved success, the performance ... is restricted by limited scalability and inefficient utilization of modern computing power"——先承认成功再收回缺口，与摘要 A2 呼应）。
- **编号局限枚举**（"(1) structural hashing is very time/memory-consuming ...; (2) functional propagations ... are solver-ready but extremely expensive ...; (3) all these algorithms do not effectively utilize modern computing power **due to the difficulty of parallelism**"——三点各带机制归因；第 (3) 点是算法类共性缺口）。

### P4（3 句）：ML 转向句
- **Recently-趋势句带 alternative 定位**（"we have witnessed the emergence of ML applied for computer systems and EDA tasks [23], **as an alternative to** conventional design solutions"——趋势+替代关系声明）。
- **Since-自然适配句**（"Since circuit netlists or BNs can be easily represented as graphs, GNNs are **naturally suitable to** classify ... [2], analyze ... [27], and predict ... [12]"——结构适配性论证：领域对象天然是图→GNN 天然适用，三任务各挂引文）。

### P5（2 句）：Motivated-by 双回收登场句
- **双回收登场句**（"**Motivated by the limitations of conventional approaches and the potentials of GNNs** applied on circuit designs, we propose ... GAMORA"——把 P3 局限与 P4 潜力两个前段回收到一句登场——动机双源回收）。
- 能力复述句（三能力+多任务 GNN 机制句+Once-well-trained 时间状态句带 leveraging 分词）。

### 贡献 bullet ×4：**粗体短语标题式**
- 每条 bullet 以**粗体短语标题**开头（Novel multi-task GNN / Billion-node scalability and parallelism / Generalization capability / Evaluation）——条目自带命名，正文两句展开。
- bullet 1 的 **corresponding-to 句式**（message passing "corresponding to the symbolic propagation and structural hashing in conventional reasoning methods"——新机制与前作机制逐点对应声明，图 1 佐证）。
- bullet 3 的 **Unlike-反向泛化句**（"Unlike many ML-based approaches that are trained with complex designs and infer on simpler ones, GAMORA can easily generalize from simple to complex"——点破领域常见做法的反向性）。
- bullet 4 的双 Regarding 分口径（Regarding reasoning performance ...; Regarding runtime and scalability ...——质量/速度再编号，与摘要 A5 同构重复）。

### 对照我们手稿的差距结论（摘要 + Intro 遍历）
| Gamora 句式 | 我们的状态 |
|---|---|
| A4 压力测试预告句（since 已知更难+主动评估） | 摘要 "Separate historical timings show slower sparse decoding" + Q2 128K 失败案例保留——主动纳入不利证据，功能等价，不改 |
| A5/贡献 bullet 4 编号双口径 | 摘要三实测数字+边界句（诚实镜像），不改 |
| P1 量化背景句（40% + 引用） | LongBench/RULER 引用型 stakes 句已有（第 6 轮补回）；行业量化数字无可引来源，诚实边界禁造，不加 |
| P2 三层嵌套归因句 | Intro P1 机制链（KV 反复访问→选择自带开销→级联两决策）等价，不改 |
| P3 Despite+编号局限枚举 | Related Work credit-then-gap 结构（第 7/12/18 轮已建）等价；我们的「局限」是对我们的测量（非前作缺陷枚举），结构不同，不改 |
| P4 Since-自然适配句 | 「低频 RoPE 旋转较慢」动机句+坐标诊断等价（适配性论证=慢旋转动机），不改 |
| P5 双回收登场句 | Intro P4 "These observations lead to a cascade with two explicit controls"——双观察（P2 评分目标+P3 配额）回收，等价，不改 |
| 贡献 bullet 粗体短语标题 | 我们用 Q1/Q2/Q3 问句标题（两种 DAC 惯例，第 6 轮已判定保持现状），不改 |
| **corresponding-to 新旧机制对应句**（bullet 1：message passing ↔ symbolic propagation+structural hashing） | **本轮实缺候选**：我们 Related Work P1 有「已有组成部分+我们组合」定位句，但**无新旧机制逐点对应声明**——共享坐标对应什么前作机制、区域配额对应什么前作机制，正文从未逐点映射。诚实可对应项：区域配额的近远容量分离对应 SparQ 近期窗口保护（远区侧扩展）；共享坐标对应 SparQ 查询选定坐标（低频+配对结构侧差异）。但逐点对应需谨慎——AsyncTLS already-combines 句已覆盖组合先例，加 corresponding-to 有夸大新颖性风险。**判定：诚实边界下不加**（对应关系会暗示等价性/优越性主张，需实测支撑） |
| Unlike-反向泛化句 | 无诚实可用的反向性（我们的比较是配置内比较非训练泛化），不加 |

### 本轮修改（第 22 轮）
- 判定上述各候选后，摘要与 Intro 的 Gamora 句式均有诚实镜像或不适用。转向**摘要句长审计**发现实缺：摘要第 2 句 "We study a near--far page-to-token cascade sharing 32 low-frequency RoPE scoring coordinates and using original-dimensional attention on selected tokens."（方法句 27 词）与 Gamora A3 对照——Gamora 登场句带三能力枚举（suffering 缺口逐点对应），我们的方法句无能力/范围枚举，但**「近远分区」区域属性在方法句中缺失**（题名 Near--Far、级联核心属性「区域配额控制两区访问」未出现在摘要方法句——读者从摘要只知道 cascade+共享坐标+原维注意力，不知道 near--far 分区这一标题词）。加 "near--far" 修饰：`We study a near--far page-to-token cascade ...` 句首已有 near--far；**改判定为句 3 与标题词对齐检查**：句 3 "Regional coverage involves trade-offs" 已承载区域属性。**最终实缺 = 摘要第 3 句被动泛指**："Regional coverage involves trade-offs" 无主语施动者（谁产生取舍？设计约束），与 claim-first/主动语态规则弱冲突，且未连接到标题词 near--far。改为 "Near--far coverage involves trade-offs"（把标题关键词注入摘要首三句，同时保留原义；-1 词无溢出风险）。
- paper_zh.tex 直译同步（「近远覆盖存在取舍」——原句「区域覆盖存在取舍」）。
- 版式核对：词数不变（近远 2 字替换区域 2 字），编译后核对 en 6 页。

---

## §2 Preliminary and Motivation（第 23 轮，2026-10-18）

### §2.A Boolean Networks and AIGs（2 段）
- **成熟度声明句带多域枚举**（"BNs are well-studied discrete mathematical models with broad applications in chemistry, biology, circuit design, formal verification"——铺垫节 opener=对象地位声明）。
- **表示选择理由句带 since**（"a concise and uniform representation ..., known as AIGs, has found successful use in diverse EDA tasks, **since** AIGs allow rewriting, simulation, technology mapping, placement, and verification to **share the same data structure** [18]"——数据结构统一性=选择理由，五任务枚举）。
- **定义句组**（each node at most two incoming / PI / PO / internal AND——教科书式 but 每句一个性质）。
- **转换保证句带性质副词**（"Based on De Morgan's laws, any combinational BN can be converted into an AIG [4] **in a fast and scalable manner**"——定理引用+性质限定）。
- **cut 定义段**：可行 cut 定义句（transitive fan-in cone / truth value assignments completely determine）→ K-feasible 定义句 → **图例走查句链**（Figure 2(a) 3-feasible cut of node 9 → truth table computation → **Thus 推导收口**（node 6 is XOR2, node 9 is XOR3）→ **By-pairing 提取机制句**（pairing XOR3 with MAJ3 with identical inputs → full adder bitslice → word-level aggregation——机制→提取→聚合三层目的链）。

### §2.B Word-Level Abstraction（1 段）
- **收益+机制+普及三合一开场句**（"Word-level abstraction **significantly reduces** the complexity ... **by grouping** wires ... **and keeping** useful information ..., which is widely applied in ... [15],[20],[26]"——分词双机制+引用簇）。
- 双方法定义句（shape hashing 定义含 constructed-by 被动结构 / bitslice aggregation 定义）。
- **However 三连缺口句**（"for large-scale BNs, structural hashing is memory-consuming; functional bitslice aggregation is not efficient due to the requirement of bit-blasting; the computation of symbolic evaluation is also expensive"——三方法逐个缺口，各带机制归因——Intro P3 编号局限的正文展开版）。
- **Motivated-by 收口句**（"Motivated by the limited scalability and the difficulty of parallelism, we propose to exploit graph learning and GPU acceleration for highly scalable reasoning"——小节级动机回收，双缺口→提案）。

### §2.C Graph Neural Network（2 段）
- **Since-自然适配句四任务版**（"Since BNs ... are naturally represented as graphs, GNNs can be leveraged to classify ..., predict ..., analyze ..., and predict ..."——适配句+四前作枚举）。
- **Promising-as-they-are 让步缺口句**（"**Promising as they are**, these approaches focus on graphs with tens of thousands of nodes, and conduct training on complex designs and inference on relatively simpler ones, in which the generalization capability ... is not well examined"——先功劳后双缺口：规模上限+泛化方向反——为贡献 bullet 3 的 Unlike 句铺垫）。
- **GNN 机制对应段**（operate-by 句 / 初始化句 / 更新句）→ **双 corresponding/analogous 句**（"The propagation along edges ... **corresponding to** structural shape hashing; ... neighborhood aggregation **is analogous to** functional aggregation"——把 §2.A/§2.B 铺垫的两机制逐点映射到 GNN——Intro bullet 1 corresponding-to 句的正文完整版）。
- **Thus 机制收口句**（"Thus, the inherent message-passing mechanism in GNNs enables simultaneously handling structural and functional information"）。
- **Motivated-by-the-analogy 登场句**（"Motivated by the analogy between GNN computation and conventional reasoning, we propose a multi-task GNN ..."——类比回收→提案；与 §2.B 末句形成同构双 Motivated-by 收口）。

### 对照我们手稿的差距结论（§III 方法节遍历）
| Gamora §2 句式 | 我们的状态 |
|---|---|
| 表示选择理由句带 since（统一数据结构=理由） | §III-A 坐标选择动机句 "Their slower positional rotation motivates this selection, although it does not guarantee ..."——动机+诚实限定版，等价，不改 |
| 转换保证句带性质副词（fast and scalable） | eq:bound 段 "$z_q^\top z_i\leq U_b(z_q)$ for every unquantized shared-space key"——定理级保证+范围限定（我们的性质副词=精确限定域），等价，不改 |
| **Promising-as-they-are 让步缺口句**（ML 前作也双缺口：规模+泛化方向） | 我们 Related Work 无 ML 前作类（领域内无「学习型稀疏注意力」先例可诚实枚举四篇），不适用 |
| **双 corresponding/analogous 机制映射句**（正文版） | 第 22 轮已判定诚实边界下不加（等价性主张需实测）；但注意 Gamora 的用法是**类比**（analogous）非等价（equivalent）——类比声明弱于功能等价声明。重审：我们的诚实对应物是 Intro P4 "supplies the scoring coordinates for both stages"（共享事实）而非机制类比；类比仍会暗示消息传递级等价，维持不加判定 |
| However 三连缺口句（正文版编号局限） | credit-then-gap 已建（第 7/12/18 轮），改 |
| Motivated-by 小节收口×2（双缺口→提案 / 类比→提案） | **本轮实缺候选**：我们 §III 各小节无 Motivated-by 收口——但 §III 开头段已有需求承接句（"To control both selection cost and candidate access, the cascade uses two explicit mechanisms"）+ gate 小节承接句（第 9 轮加）；小节级 Motivated-by 是 Gamora 背景节的结构（背景铺垫→动机收口→方法），我们的结构是方法直述（Intro 动机→方法），两种组织均常见，判定不加 |
| By-pairing 提取机制句（机制→提取→聚合目的链） | §III-C→§III-D 走查链等价，不改 |
| 图例走查句链（Thus 推导收口） | eq:bound 段 Each-signed-coordinate 句→so 收口（第 9 轮已确认达标），不改 |
| **By-pairing 式「组合结构识别」句** | 重审无对应物：我们的页/词元两级组合（配额+共享坐标）在 §III 开头以 "two explicit mechanisms" 列出，但**两机制如何相互作用**（配额决定候选资格、坐标决定排序——两级各自的角色分工句）从未一句话点明。诚实可写：L1 配额决定哪些页可提供候选（eligibility），L2 共享坐标决定哪些词元被选（ranking）——这是定义级角色分工，§III 开头段 "L1 assigns separate page quotas ... In the ordinary L2 path, full-history scoring precedes ..." 已分句覆盖。判定：已覆盖，不加 |

### 本轮修改（第 23 轮）
- 遍历 §2 各句式后均有诚实镜像或判定不加（维持第 22 轮 corresponding-to 判定）。转向**§III-A 首句的对应物检查**：Gamora §2.A 的「表示选择理由句带 since」（AIG 统一数据结构）——我们 §III-A 首句 "Both regions and both selection stages use the same coordinate map within each layer and KV head, while each query head is scored separately." 是**共享事实句但无 since 理由**；共享的理由（同一评分视图服务两级→选择成本与视图一致性）在 §III 开头段（"To control both selection cost and candidate access"）而 §III-A 小节内无。实缺确认：**§III-A 首句缺 since 理由从句**——共享坐标图的「复用」性质（两个区域两个阶段共用→无需两套评分表示）是定义级事实。加理由：`Both regions and both selection stages use the same coordinate map within each layer and KV head, so the two stages share one scoring view, while each query head is scored separately.`——检查后判定更简形式：句内已有 "use the same coordinate map"，加 "so ..." 会同义重复。**改为另一实缺**：§III-A 第二句 "For Qwen3-8B, we select post-RoPE coordinates $J=...$ from each 128-dimensional query and key." 的 "For Qwen3-8B" 限定语——Qwen3-8B 是 8B 模型、128 维 RoPE；「每层每头」的层域已在首句，此句的 For 限定是模型限定（对非 Qwen3 模型坐标需重选）——**该限定语已充分**。最终定论：§III-A 句式对齐度已高（第 9/19 轮累积修复），**本轮实缺 = §III-A 末段 L1/L2 分工句的「用于粗粒度排序/精筛」目的从句缺失的 zh 版**——zh 版 "L1将选定的键坐标聚合为页摘要，用于粗粒度排序" 已有「用于」；en 版 "L1 aggregates selected key coordinates into page summaries"（第 19 轮远端精化删去了 for coarse ranking）——**en 版目的从句被删**。恢复定义级目的从句：`L1 aggregates selected key coordinates into page summaries for coarse ranking; L2 refines candidates using quantize--dequantize token keys.`（+3 词，恢复远端 55b9a17 精化前的功能等价信息；zh 保持不变）。
- 版式核对：+3 词，编译后核对 en 6 页。

---

## §3 Proposed Approach（第 24 轮，2026-10-18）

### §3.A Overview（2 段）
- **粗体段落标签句**（"**Problem Formulation.** Figure 3(a) illustrates the overview of GAMORA."——run-in 粗体标签自带命名，段首即功能声明）。
- **输入定义句带 without 排除声明**（"The inputs are flattened gate-level netlists in AIG format, **without any** micro-architectural or RTL information"——输入界定同时声明排除项，范围诚实边界）。
- **目标句带 by-机制与 after-which 下游后果**（"The goal is to exploit a multi-task GNN to reason ... **by performing** node-level classification ..., **after which** functional blocks (e.g., adders) can be extracted"——目标→机制→下游后果三段链）。
- **Case Study on Multipliers 段**（第二个粗体标签）：stakes 句（indispensable+双应用枚举挂引用）→ 趋势句（Recent years also witness+同态加密引用）→ **难度句带 especially-子类聚焦**（"formal multiplier verification is challenging, **especially for** structurally complex designs such as Booth multipliers"）→ SCA 功劳句（successfully employed+引用簇+relies heavily on 依赖声明）→ **前作 SOTA 点名句带量化上限与代价**（ABC fast algebraic rewriting "can handle large bitwidth multipliers (**up to 2048-bit**) **but with** extremely long runtime"——credit-then-but 正文版，上限量化+代价直报）→ **Thus 收口登场句带双收益 which 从句**（"**Thus**, targeting integer multipliers, we leverage GNNs to ..., **which improves** the efficiency of word-level abstraction ... **and has** strong scalability enabled by GPU acceleration"——定位词+机制+双收益收口）。

### §3.B Multi-Task Learning（开头段 + 三小节）
- **机制对应声明句**（"Boolean reasoning **requires** gathering structural and functional information from neighbor nodes, **a process that can be imitated by** the message-passing mechanism in GNNs"——需求→GNN 可模仿，同位语从句；对应 §2.C 铺垫）。
- **两步程序引用句**（"involves a two-step procedure [15][20][26]: (1) detecting XOR/MAJ functions to construct adders, and then (2) identifying their boundaries"——编号流程挂引用，程序结构归前作）。
- **Therefore 提案句带本质贴合论证**（"we propose to apply MTL for Boolean reasoning **to approach its nature**, in which knowledge sharing across sub-tasks provides higher reasoning precision"——贴合问题本质=设计正当性）。
- **编号路线图句**（"This section details (1) ... (2) ... (3) ..."——与小节 1)2)3) 一一对应）。

#### 1) Fusing Structural and Functional Information
- We-leverage 目的句（generate node embeddings that **simultaneously fuse**——目的+双信息融合）。
- First/Second 双机制分述句。
- **not-only-but-also 双收益句**（"These compressed node features **not only** encapsulate Boolean functionality **but also** enable high compute and memory efficiency"——设计决策双收益）。
- **图例走查句链**（"node 1 is a PI with the feature vector [0, 0, 0]; node 7 ... [1, 0, 0]; node 17 ... [1, 1, 1]"——具体节点特征向量实例枚举）。
- **模型选择句带强调理由回收**（"**With the emphasis on** generalization from simple to complex designs, the specific model employed is GraphSAGE"——回收 Intro Unlike 句伏笔：强调什么→选什么）。
- 公式后性质收口句（"After stacking K layers, the ... within K-hop search depth is fused"——公式→性质）。

#### 2) Multi-Task Classification
- **We-identify 本质贴合句带冒号展开**（"to approach the nature of the problem: there are two steps involved"）。
- **标注规则句带 Since 归因**（"**Since** each XOR/MAJ function consists of multiple nodes, only the root nodes are labeled ..., with other nodes marked as plain nodes"——结构性原因→标注设计决策）。
- 等价扩展披露句（negation-permutation-negation equivalent functions also labeled——鲁棒性扩展边界）。
- **MTL 双收益论证句**（"MTL **not only** follows the intuition of this two-step reasoning **but also** exploits divide and conquer, **since** it is extremely hard for GNNs to reach high accuracy with a single-task multi-label classification"——not-only-but-also + since 单任务难度机制归因——设计选择双重理由）。
- **The-employment-of enables 句**（"enables knowledge sharing ... and improves sample efficiency ..., **which guarantees** high reasoning performance"）。
- Specifically 解耦句（two-step decoupled into three simpler tasks——数量转换明确）。
- **超参具体值披露句**（"In our implementation, α = 0.8 and β = γ = 1"——具体数值进正文）。

#### 3) Adder Tree Extraction
- 匹配规则句（XOR and MAJ pairs with identical inputs are matched）。
- **完整算法图内走查长链**（集合枚举 X={12,17,...} M={10,22,...} → 移除非根 → 逐对匹配 node 12 is XOR3(8,9,0) ... "matching continues until all six pairs"——算法全流程用图实例走查）。
- **诚实失败案例句（方法节内！）**（"In Figure 3(e), one HA **cannot** be automatically extracted **due to the misprediction of** node 10"——方法节直接展示失败案例图）。
- **失败定界句三级止损**（"**only several** nodes near the least significant bit are **always** mispredicted **due to** their shallow neighborhood structure, **which has a subtle impact** on ..."——失败范围 only-限定+机制归因+影响量化 subtle——把失败钉死在小范围）。
- **期望收口句带诚实动词**（"GAMORA **is expected to** reach **as close as possible to** the exact reasoning precision"——By-doing 收口+expected/as-close-as-possible 不做等价声明）。

### 对照我们手稿的差距结论（§III 方法节遍历）
| Gamora §3 句式 | 我们的状态 |
|---|---|
| 粗体段落标签句 | subsection 标题等价，不改 |
| 输入 without 排除声明 | §III 开头 "operates only during decoding; prefill remains dense" 等价（范围排除），不改 |
| 目标句 by-机制 after-which | §III 开头 "To control both selection cost and candidate access, the cascade uses two explicit mechanisms" 等价，不改 |
| SOTA 点名带量化上限与代价 | Related Work credit-then-gap 已建（第 7/12/18 轮），不改 |
| Thus 收口登场句带双收益 which | 诚实边界禁（我们的论点即「无既定联合优势」，方法节不能挂未证双收益），不加 |
| 编号路线图句 (1)(2)(3) | §III 开头段已分句预览 A（共享视图）/B（配额）/C（合并+原维注意力），D 有自带承接句（"Beyond scoring and budgeting"）；零版式余量下全节路线图与既有预览重复，判定不加 |
| not-only-but-also 双收益 | 我们是诚实反称版 "narrows dot products but adds work"（收益+反收益对），等价不改 |
| 图例走查句链（具体实例枚举） | §III-B 已有数值走查（$K_1=128$/$\beta=.125$→16/112 页；892 槽→640/252），等价不改 |
| 模型选择句带理由回收 | 无模型选择自由度（Qwen3-8B 固定）；坐标选择已有慢旋转动机句，等价不改 |
| 标注规则 Since 归因句 | §III-C "Because softmax precedes group averaging..." + §III-B "The partial prefix page can reselect up to four protected tokens, so 252 is nominal" 等价，不改 |
| 超参具体值披露句 | 表 1 + 行内数值（$K_1=128$、$\beta=.125$、$K=1024$）已覆盖，不改 |
| 算法走查长链 | eq:bound 段+配置走查等价，不改 |
| **诚实失败案例句（方法节内）** | 我们失败案例在评估节（观察图漏页）；方法节已有诚实界限句（"does not guarantee preservation"、"loose enough to misrank"、"cannot recover omitted tokens"）。方法节新增失败实例需新测量支撑（观察图单查询数字不可泛化为方法级失败定界），诚实边界下不加 |
| 失败定界句三级止损 | "Extrema may come from different tokens, making the bound loose enough to misrank pages and miss important tokens"——失败机制+后果等价；范围量化需新数据，不加 |
| 期望收口句 | §III-D 末句诚实界限收口等价（"The prefill threshold does not bound decoding error"），不加 |
| **本轮实缺 = §III-B 首句主谓失配** | "zero-based position $t$ includes prefill"——位置不能「包含」预填充（包含的是计数范围），主语-谓语语义悬空；zh 版括注清晰（「在完整序列中从0计数的位置（包含预填充词元）」），en 版未达到同等清晰度。属表述精度问题 |

### 本轮修改（第 24 轮）
- §III-B 首句：`In decoding, zero-based position $t$ includes prefill, and $S=t+1$ counts valid cached positions through the current token, excluding padding.` → `In decoding, $t$ indexes the current token from zero across the full sequence including prefill, and $S=t+1$ counts valid cached positions, excluding padding.`（主谓修复：t 作索引主动计数；「through the current token」由 t=当前位置定义蕴含；+4 词；zh 已清晰无需改）。
- 版式核对：编译后核对 en 6 页。

---

## §4 Experiment + §5 Conclusion（第 25 轮，2026-10-18）

### §4.A Experiment Setup（1 段）
- 工具与真值来源句（ABC 生成 + ground truth by extraction command——工具链完整披露）。
- **编号双库枚举句**（two technology libraries: (1) mcnc.genlib (gate input size <=3); (2) ASAP 7nm——每库带括注量化）。
- **双模型分述句**（shallow 4-layer/32-channel for simple netlists; deep 8-layer/80-channel for Booth+complex mapping——模型按设计复杂度分化，括注用途）。
- 环境/框架句（PyTorch Geometric + EPYC 7742 + A100 40GB）。
- **Setup 收尾泛化范围句**（"In general, GAMORA is trained on small bitwidth multipliers (**typically less than 32-bit**) and evaluated on large bitwidth multipliers (**up to 2048-bit**)"——训练-评估不对称范围声明，为全部后续评估定性）。

### §4.B Evaluation on Reasoning Performance（开头 + 三小节）
- **编号三方面评估路线图句**（"from three aspects: (1) how ...; (2) how ...; (3) how ... and what domain insights can be derived"——评估维度编号+how 从句，与 §3.B 路线图同构）。

#### 1) Reasoning Precision Analysis
- 图题句带三因子枚举（Figure 4 illustrates how ... affected by bitwidth/single-multi-task/functional info）。
- **First/Second/Third 三发现枚举**（每条带机制归因）：
  - **the-larger-the-higher 单调句带收敛点**（"the larger bitwidth multiplier is adopted for training, the higher precision ..., which typically **converges after** training with 8-bit"）+ **The-main-reason-is 归因句带充分性论证**（8-bit provides a **sufficient variety** of structural properties, learned and **well generalized**）。
  - **conspicuously-outperforms 比较句带 indicating 归因**（multi-task vs single-task, indicating knowledge sharing greatly benefits every single task）。
  - **there-is-always-a-boost 句带 since 归因**（functional info always helps, since role identification relies on not-only-structure-but-also-function——not-only-but-also 的归因从句版）。
- **实验发现→前作机制类比句**（"The synergy of structural and functional information in GAMORA **is analogous to** the combination of structural hashing and functional propagation in conventional symbolic reasoning"——评估发现回收 §2 对应铺垫，corresponding-to 的评估节回声）。
- **头牌句**（achieves almost 100% prediction accuracy for CSA）。
- **诚实失败句带可纠正性**（"several nodes near LSB are always mispredicted ... The HA at LSB **cannot** be automatically extracted, **but can be easily corrected during post-processing**"——失败+简单补救收尾——方法节失败定界的评估节版本，but 转折化解）。

#### 2) The Impact of Design Complexity
- **From-X-aspect 句式 ×2**（From the model selection aspect, ...; From the training aspect, ...——双视角分述同一实验）。
- thus-guaranteeing 分词收口；**adequate-variety 训练多样性归因句**（larger multipliers required **such that** adequate variety and representativeness ... are exposed to and well captured）。

#### 3) The Impact of Technology Mapping
- **known-challenge 引用句 + Thus 评估动机句**（"It is a **known challenge** that technology mapping can increase the complexity ... [15][20][25]. **Thus**, we evaluate GAMORA with respect to different technology mapping options"——难度归前作引用→评估正当性）。
- **库规模量化句带复杂度归因**（"ASAP 7nm library contains **161** standard-cell gates, **including** multi-output cells such as the full adder cell, **which significantly increases** the complexity and irregularity"）。
- **好情形双数字句**（still reaching **over 99% and 92%** for CSA and Booth——simple mapping 泛化）。
- **The-scenario-is-fairly-different 反转句 + first/second/third 反转三点枚举**（7nm：泛化受限/精度略降即使重训/需大训练乘法器——三点递进恶化）。
- **These-observations-imply-several-takeaways 收口句 + First/Second takeaway 枚举**（观察→教训编号收口——takeaway synthesis 的编号版）：
  - the-more-the-more 单调归因句带 since（more complex library → more difficult, since more complexity both in structures and functionality）。
  - **comes-at-the-expense-of 代价句**（"the capability to cope with intricate AIG netlists **comes at the expense of** more comprehensive training data"——能力-代价对偶）。
  - 理论引用支撑句（ERM+Vapnik [22]——训练数据增大→泛化的理论背书——takeaway 挂理论引用）。

### §4.C Runtime and Scalability Analysis
- **In-addition-to-quality 递进登场句**（"**In addition to** the high reasoning performance, we demonstrate the superiority of GAMORA **by analyzing** its runtime and scalability"——质量→速度双口径递进）。
- **粗体 run-in 段落标签**（Runtime complexity analysis. / Batched reasoning with single GPU.）。
- **规模量化锚定句**（"a 2048-bit CSA multiplier with **around 34 million nodes and 67 million edges**"——把 bitwidth 翻译成 |V|/|E| 物理规模）。
- **头牌速度句**（speedup of **up to six orders of magnitude**）。
- **This-shows-not-only-but-also 收口句**（not only great efficiency ... but also the scalability to extremely large designs——图证→双收益收口）。
- **当前限制披露句带 future work**（"we focus on single GPU ..., and **leave multi-GPU implementation as our future work**"——范围限制+出路）。
- **Even-with 让步正面收口**（"**Even with** a single GPU, there already reveal promising results and positive trends"）。

### §5 Conclusion
- 四度同构登场句（benefited 枚举=摘要 A1）+ 提案句（exploits GNNs **to imitate** structural hashing and functional aggregation——一句话点机制）。
- **编号三贡献复述句**（"(1) ... almost 100% and over 97% ..., still over 92% after complex technology mapping; (2) ... over 33 million nodes, up to six orders of magnitude speedups ...; (3) ... generalization from simple to complex ..., from before to after technology mapping"——与摘要 A5 同构，质量/速度/泛化三贡献各一句，数字全带）。
- 潜力收口句（reveals the great potential of applying GNNs and GPU acceleration）。

### 对照我们手稿的差距结论（Setting/Evaluation/Conclusion 遍历）
| Gamora §4/§5 句式 | 我们的状态 |
|---|---|
| 编号三方面评估路线图句 | Setting opener Q1/Q2/Q3 标签+小节问句标题（第 17/20 轮已建）等价，不改 |
| Setup 收尾泛化范围句（train-small/test-large） | 我们的 compact/historical 臂身份绑定句（远端 7d33672）+ Workloads 段枚举覆盖范围声明，等价，不改 |
| First/Second/Third 三发现枚举 | Q2 各证据块是异质记录非平行轴，编号会暗示不存在的平行结构，不加 |
| the-larger-the-higher 单调句 | Q3「Although the relative penalty decreases with length, absolute sparse TPOT increases」等价（带方向的让步单调句），不改 |
| 失败句带可纠正性（cannot ..., but easily corrected） | 诚实边界禁：22.45 分损失无可纠正性主张，不加 |
| known-challenge 引用句+Thus 动机 | gate 小节负范围 opener 等价（"does not establish that ... can be safely skipped"），不改 |
| The-scenario-is-fairly-different 反转句 | Q2 顺序证据链承担对比功能，反转句式无对应物需求，不加 |
| takeaways 编号收口 | 「Design implications and limitations」段（两约束枚举+Likewise 连接）等价，不改 |
| comes-at-the-expense-of 代价句 | 「narrows dot products but adds work」诚实代价对偶等价，不改 |
| In-addition-to-quality 递进登场 | 我们无已确立的质量优势可递进（诚实边界），Q3 问句标题承担动机，不加 |
| 粗体 run-in 段落标签 | Model-level decoding. / Local selector cost. 等价，不改 |
| 规模量化锚定句 | TPOT 三长度带毫秒数+比值等价，不改 |
| 当前限制披露带 future work | 「remains missing/ unresolved」需求句式等价（出路=条件而非承诺），不改 |
| 编号三贡献复述句 | 结论顺序描述性结果+诚实边界收口（既有判定：诚实镜像），不改 |
| **本轮实缺 = Q2 128K 句裸形容词缺中心词+比较对象缺失** | "Historical 128K sparse scores 23.98 versus 46.43"——"sparse" 悬空（Q2 既有术语是 sparse arm/configuration），46.43 的参照（稠密 FullKV）未点名；zh「历史128K稀疏得分23.98，对照46.43」同病 |

### 本轮修改（第 25 轮）
- Q2 128K 句：`Historical 128K sparse scores 23.98 versus 46.43.` → `The historical 128K sparse arm scores 23.98 against dense FullKV's 46.43.`（补中心词 arm 对齐 Q2 术语、点名比较对象；+4 词，远端压缩后版式有余量）。
- paper_zh.tex 直译同步：「历史128K稀疏得分23.98，对照46.43」→「历史128K稀疏臂得分23.98，对照稠密FullKV的46.43」。
- 版式核对：编译后核对 en 6 页。
