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
