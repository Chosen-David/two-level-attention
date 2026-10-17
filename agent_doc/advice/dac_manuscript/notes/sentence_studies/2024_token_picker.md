# Token-Picker (DAC 2024) 逐句精读笔记 — Abstract + Introduction

> 2026-10-16 第 1 轮逐句学习，直接读 PDF 原文（`2024_Token_Picker.pdf`，arXiv:2407.15131v1）。
> 用途：对照 `source/paper_en.tex` 同段落的句子功能，逐句修我们的表述。
> 拆解格式：原文句 → 该句在做什么（功能标签）→ 对我们的启示。

## Abstract（6 句）

| # | 原文（缩写） | 功能 | 对我们的启示 |
|---|---|---|---|
| A1 | "The attention mechanism in text generation is memory-bounded due to its sequential characteristics." | 痛点 + 归因一句话（慢在哪、为什么） | 我们摘要第 1 句是抽象定位（"must decide which history remains accessible"），没有归因。应有"成本在哪、为什么"一句：decode 注意力成本随缓存历史增长，选择本身要扫全史。 |
| A2 | "Therefore, off-chip memory accesses should be minimized for faster execution." | 需求推导（Thus/Therefore 收口） | 我们缺"因此……必须……"的需求句：选择机制本身必须便宜，否则省下的注意力成本被检索吃掉。 |
| A3 | "Although previous methods addressed this by pruning unimportant tokens, they fall short in selectively removing tokens with near-zero attention probabilities in each instance." | 前作缺口（Although…they fall short in + 具体机制缺口） | 我们摘要完全没有前作定位句。应加：Although query-aware selectors reduce the attended set, they score in full width and rank one global pool…（机制级缺口，不虚黑） |
| A4 | "Our method estimates the probability before the softmax function, effectively removing low probability tokens and achieving an 12.1x pruning ratio without fine-tuning." | 方法一句 = 机制动词 + 效果 + 头牌数字 + 免微调卖点 | 我们方法句可合并成一句带机制动词的：cascade 用共享 32 维评分图 + 区域配额，注意力仍用原始维 Q/K/V。 |
| A5 | "Additionally, we present a hardware design supporting seamless on-demand off-chip access." | 第二贡献一句 | 对应我们的 regional quota 第二机制。 |
| A6 | "Our approach shows 2.6x reduced memory accesses, leading to an average 2.3x speedup and a 2.4x energy efficiency." | 结果三连数字，"leading to" 因果链 | 我们给三个裸分并列（47.588/47.910/47.666）无方向无解读。应给"within 0.33 points"式的方向性收口。 |

诚实边界（我们的 6-9 句）是项目硬约束，保留但压缩：Token-Picker 用一个介词短语（"without fine-tuning"）解决一个卖点，我们用 3 整句重复"未建立优势"，压成 2 句。

## Introduction

### P1（2 句）：stakes 段
- "Text generation using LLMs have been instrumental in advancing applications such as chatbot systems and virtual assistants." → 谁受益。
- "With their growing importance, several companies are striving to integrate these applications into their hosted services, highlighting the importance of efficient inference." → 商业/部署压力收口。
**我们缺整段 stakes**：我们的 Intro P1 直接进入机制问题，没有一句"长上下文服务/成本/部署"的世界事实句。

### P2（~8 句）：机制铺垫 → 归因 → 部分解 → 残余问题 → Thus 收口
句链：transformer 组成 → attention 角色 → "this sequential property makes the workload memory-bound, not fully utilizing GPUs"（归因句，主动语态）→ 动态 batching 部分解决 → "Still, the attention mechanism faces memory challenges because each user's sequence remains separate"（Still 让步转折）→ batch 越大 access 越多 → "Thus, minimizing the memory transfer of attention can improve throughput and energy efficiency"（Thus 需求收口）。
**启示**：每个动机段落必须以 Thus/Therefore 需求句收口；归因句用主动语态（"X makes the workload memory-bound"），不用"Y is made memory-bound by X"。

### P3（~5 句）：前作 + However 缺口 + Furthermore 第二缺口
- "Previous works [1,8,16] could achieve this by leveraging the inherent redundancy by softmax…" → 前作功劳先承认。
- "However, these approaches overlooked the varying number of unimportant tokens across instances and could not selectively spare the important ones only." → However + 具体机制缺口（实例间变异）。
- "Furthermore, retraining is often required for high pruning rate, as…" → 第二缺口 + 原因从句。
**启示**：先承认前作功劳（"could achieve this by…"），再 However。我们的 Related Work 段是平铺列举，没有"先功劳后缺口"的让步结构。

### P4（~4 句 + 3 编号机制点）：方法登场
- "To tackle the aforementioned problems, we introduce an adaptive token pruning method that aligns with each instance." → 方法名 + 一句定位。
- "Before completing all correlation calculations, our method estimates token probabilities and eliminates those below a set threshold." → 机制核心一句。
- "Memory accesses during the estimation is minimized by: 1)… 2)… 3)…" → 编号机制列举。
- "Notably, our method achieves a higher pruning ratio compared to previous techniques, without the need for retraining." → Notably + 对照卖点收口。

### P5：贡献 bullet ×3
- "We propose / We present / We design + what + 数字"（12.1×、1.39×、2.28×/2.41×）。
**启示**：我们的 Intro 用 Q1/Q2/Q3 问题列表替代贡献 bullet。DAC 惯例是动词开头的贡献 bullet；至少应在 Q 列表前加一句贡献声明，或改成"贡献 bullet + 每条绑 Q"。

## 本轮修改（paper_en.tex）
1. Abstract 重写：加 A1 归因句、A2 需求句、A3 Although 缺口句；方法合并；数字加方向性（within 0.33 points）；诚实边界 3 句压 2 句。
2. Intro P1：补 stakes 段（2 句世界事实）+ 原机制句合并改主动。
3. paper_zh.tex 同步直译。

## 待后续轮次
- Intro P2（观察段）对照 Token-Picker §2.2 每小节"一图一数字一结论一收口"。
- Intro P4 的 Q1/Q2/Q3 → 贡献 bullet 化评估。
- Related Work 段"先功劳后 However 缺口"结构。

---

## §2 Background & Motivation（第 2 轮，2026-10-16 20:5x）

### §2.1 Autoregressive transformer（背景铺垫，2 小节）
句链：transformer 组成 → 三步公式（q/k/v→scores→softmax→output）逐方程用 "(1)(2)(3)" 编号讲解 → KV caching 定义 → generation 阶段 GEMV → "Therefore, latency and energy consumption are dominated by off-chip access"。
**功能**：背景节也以 Therefore 收口；每方程配编号，为动机节引用（"near-zero p_i in Eq. (3)"）建立锚点——我们的正文公式 (1)(eq:bound) 同样被后文引用，做法一致。

### §2.2.1 Memory transfer overhead（6 句，一图一数字双收口）
| # | 原文（缩写） | 功能 |
|---|---|---|
| 1 | "Efficient batching technique was developed to improve the generation process." | 先功劳（承认前解） |
| 2 | "It spreads out the cost of loading weights…, thereby increasing efficiency." | 前作机理 + thereby 因果 |
| 3 | "Yet, the unique KV cache … cannot be shared, resulting in increased overheads." | Yet 让步转折 + 归因 |
| 4 | "Fig. 2 shows the breakdown of off-chip accesses … for different batch sizes, all set to maximum context length." | 图设置句（条件披露） |
| 5 | "While KV caching transfer is 7.8% at batch 1, it becomes 84.3% for batch 64, leading to prolonged latency as demonstrated in [9]." | While 数字对比 + leading-to 后果 + 引文背书 |
| 6 | "Thus, minimizing latency in self-attention is essential…. This emphasizes the need to reduce off-chip accesses…." | Thus + This-emphasizes 双重需求收口 |

### §2.2.2 Distribution-aligned pruning（12 句，含归因句与双缺口）
句链：softmax 机理 → "Thus … can be pruned with minimal loss"（Thus 剪枝机会）→ "Using this property, several works…"(先功劳) → "Among those, SpAtten … which retains … at a pre-defined ratio"（点名前作+机理）→ "While the method is effective …, it often overlooked variations across instances"（While 让步缺口）→ "Fig. 3 demonstrates the variability by comparing … in identical setups"（**图句自带结论 "demonstrates the variability"**）→ "only 4.6% …, while 23.5%"（量化对比）→ "This discrepancy arises from the relative nature of the softmax; wider distribution … leads to fewer dominant tokens"（**归因句：为什么会这样**）→ "Therefore, fixed-ratio pruning could either remove dominant or keep unimportant tokens, undermining…"（Therefore 否定固定策略 + 后果）→ "Furthermore, it necessitates additional fine-tuning …"（第二成本）→ "Other studies [8,16] … showing decent gains"（再功劳）→ "However, they are not optimized … as they require loading all KV pairs. Thus, previous methods become suboptimal … as they do not … or …"（However + Thus 双缺口总收口）。

**对我们 Intro 观察段的差距结论**（当前远端 539342c 版已含量化锚点）：
1. (a) 段已达标：claim 句 + 机理 + 图量化（13/32、59.65% vs 17/32、26.46%）+ 恢复不可能 + Therefore 收口。
2. (b) 段缺 **Tail 落点披露**：T-P 用 While 对比同时给两端数字（4.6%/23.5%），且记忆口径「配额=防挤出保底非均值增益」——arm minima 0.00→0.27（非配对）已画进图 (b) 图例（O_observation_meta 登记为 "arm minima 0.00→0.27 (unpaired)" 标注），但正文从未引用。补一句 claim-first 的尾部句，三口径（均值微升/最小值保底/43 条下降）齐报 = Token-Picker 双向诚实传统。
3. (b) 段归因句可由配额构造直接支撑（固定区域切分按构造保住容量），无需新机理。

### 本轮修改
- paper_en.tex Intro (b) 段：在均值句后补 "Quotas act on the tail, not the mean: the single pool drives far retention to zero for some queries, while the quota arm never falls below .27 (arm minima, unpaired)."（数字来自图 (b) 已登记标注，非新造）。
- paper_zh.tex 直译同步。
- （后续远端 9d82537 已采纳此句进 origin/main，并做 caption 措辞修正 "stored selections" / "a 2048-token cap"。）

---

## §3 Proposed Work（第 3 轮，2026-10-17）

### §3 开篇段（6 句：需求承接 → 障碍 → 登场 → 机制 → 双分支 → 挑战预告）
| # | 原文（缩写） | 功能 |
|---|---|---|
| 1 | "To counteract the described variability, removing only the tokens with low probability is practical instead of removing them under fixed ratio." | **需求承接句**：To [上节缺口目标], [方法主张] instead of [被否定替代]——方法节第一句必须回收动机节建立的缺口，不是直接进机制 |
| 2 | "However, since the probability is determined by the difference between scores, it requires all correlation scores, which hinders the reduction of K transfers." | However + **内在障碍**（朴素解的隐藏代价），构成第二层需求 |
| 3 | "To identify negligible tokens and reduce K transfers simultaneously, we propose a method that estimates the probabilities prior to completing the correlation score calculations." | **we propose 登场句**：双目标状语 + 机制动词（estimates ... prior to ...） |
| 4 | "Specifically, this method estimates the probability using the segmented bits (bit chunks) of K, and dynamically prunes tokens with the estimated value falling below a set threshold, thr (Sec. 3.1)." | Specifically + 机制一句 + **交叉引用小节**（为下文分节铺路） |
| 5 | "If the pruning occurs before requesting the final chunk, ... can be avoided; otherwise, the token is considered pivotal and the subsequent chunk is requested for more precise pruning decisions." | **if/otherwise 双分支执行句**：一次说清两种运行时行为 |
| 6 | "In this procedure, there are two main challenges: 1) Errors by estimation ... (Sec. 3.1). 2) On-demand DRAM requests ... (Sec. 3.2)." | **双挑战编号预告**：每个挑战各自指向一个小节 = 方法节的路线图 |

### §3.1 Probability Estimation（数学小节的三段式）
**段 1（子集估计，4 句）**：性质句 claim-first（"The nature of the exponential allows estimating..."）→ Since 推理（指数恒正 → 不等式成立）→ **Therefore 安全性收口**（估计值低于 thr 则真值必低于，irrespective of 任何后续分数）。
**段 2（局部性利用，3 句）**：目标句（"To find negligible tokens as early as possible, it is advantageous to..."）→ "To this end, we exploit the locality..."（we exploit + 已观察性质）→ 图句自带结论（"As shown in Fig. 4(a), recently generated tokens and the first token often carry more weights"）→ **Therefore 策略收口**（倒序计算 enhances 剪枝率）。
**段 3（bit-chunk margin，~11 句）**：目标句（"To further reduce K transfers, we introduce..."）→ 前作扩展定位（"This method extends the conservative margin concept [16]"——引用即功劳）→ 定义 → 公式 (4) + 性质句 → **核心能力句**（"we can predict possible range of the result even if only a portion of bits is given"）→ 图句（"Fig. 4(b) details this concept"）→ 正例 yields s_max / **Conversely** 反例 determines s_min（对称机制两句）→ **Note that 边界披露**（margin pairs 仅由 Q 决定）→ 公式 (5) → **Thus 推断收口** → 保证句（"This method ensures the safe elimination of KV transfers"）。
**小节收口（2 句）**："In conclusion, our approach provides a conservative probability estimate. It identifies redundant tokens based on ..., thus ensuring that all necessary tokens are retained in each instance."——**每个机制小节以「能力定性 + thus 保证」双句收口**。

### §3.2 Out-of-order Score Calculation
句链：承上能力句（"Our estimation method permits pruning decisions that rely on chunks of K"）→ 机制句 → **However 障碍**（等待导致 under-utilization）→ "To mitigate this, we introduce ... which processes as follows:" + **编号 (1)(2)(3)(4) 过程列举** → enables 收益句（"enables calculating ... as soon as any chunk becomes available"）→ **For example 时间线走查**（请求与加载间隔内做什么，具体到单 token k_t 的第二个 chunk）→ **Therefore 三收益收口**（keep PE active + optimize DRAM access + faster execution with fewer bit accesses）→ 衔接句（"In the end, only the tokens that ... participate in subsequent softmax and ×V operations"——与全局注意力衔接）。

**对我们方法节（§III Near–Far Selection in a Shared Subspace）的差距结论**：
1. **§III 开头缺需求承接句**（T-P 句 1 功能）：我们第一句直接进机制（"The cascade separates..."），没有 "To control [评分成本与候选访问], the cascade uses..." 的缺口回收句。Intro P4 已建立 "two explicit controls"，方法节开头应显式呼应。
2. 双挑战预告（句 6）我们已有对等物：Intro 的 Q1/Q2/Q3 问题列表 + claim 式小节标题，无需重复。
3. if/otherwise 双分支句我们已有对等物：gate 小节 "a mean far-region attention proxy below ε sets the gate OFF; otherwise far selection stays active"。
4. 数学小节的 Since→Therefore 安全性收口我们已达标：§III-B "Each signed coordinate contribution is bounded by its chosen extremum, so z_q^T z_i ≤ U_b" + 边界披露句（"This does not bound original-dimensional logits"）。
5. T-P 小节末 "In conclusion" 正面收口 vs 我们小节末诚实边界披露：边界不可删（硬约束），且 §III-A 加正面收口会与首句（"Both regions and both selection stages use the same coordinate map"）重复（non-duplication），不强加。

### 本轮修改（第 3 轮）
- paper_en.tex §III 开头段：拆原第一句为「需求承接句 + 机制句」——新增 "To control both selection cost and candidate access, the cascade uses two explicit mechanisms: a shared scoring view and regional page quotas."（selection cost/candidate access 均为全文既有术语，two explicit controls 呼应 Intro P4）。
- paper_zh.tex 同段直译同步。

---

## §4 TOPICK ARCHITECTURE（第 4 轮，2026-10-17）

### 开篇段（3 句）
| # | 原文（缩写） | 功能 |
|---|---|---|
| 1 | "This section presents the ToPick architecture (illustrated in Fig. 6), which implements self-attention mechanism with dedicated modules: the DAG and the Margin Generator, both supporting probability estimation and out-of-order score calculation." | **段落级 claim 开头**：架构 = 命名模块列表，且每个模块回指 §3 的机制（both supporting...）——§3 机制到 §4 模块的 1:1 映射在第一句完成 |
| 2 | "We design a lane-based processing element (PE Lane) for ToPick, where K/V vectors from the DRAM are partitioned to each PE Lane." | We design + 设计决策 + where 从句给数据通路 |
| 3 | "The operand precision for self-attention is set to 12 bits, segmented into three 4-bit chunks for each K/V vector." | 配置数字前置（架构节即报关键位宽） |

### 两操作 + 相位对比段（~7 句）
句链："ToPick performs two main operations: q·k (step 0) and Σpv (step 1)"（**数据流框架句**，两步编号）→ MUX 网络句（enabling...）→ **相位对比句对**："During the prompt phase, all K/V vectors are preloaded into the on-chip buffer to be reused across queries. Conversely, in the generation phase, q_t resides in the operand buffer and each KV chunk is streamed from DRAM"——两相各自的数据驻留与流动，**Conversely 显式对照**→ "Following are the details about hardware modules utilized for probability estimation in the generation phase:"（显式路线图句，细节限定在 generation phase）。

### 编号模块列举 (1)(2)(3)
Margin Generator / PE Lanes / DAG——每条 = 模块名 + 角色动词（produces / perform / determines）+ 何时（Before starting step 0 / During each cycle）+ 去向（utilized through a LUT / aggregated in DAG）；条目 (2) 内嵌前向引用（"Further details ... will be provided in the subsequent subsection"）。

### §4.1 微架构（图先行 + 组件编号定义 + 时序走查 + 回环收口）
- "Fig. 7 illustrates the PE Lane microarchitecture. It consists of four components, including..." → 图句 + 组件计数。
- 编号 (1)(2)(3) 每条角色动词开头："Scoreboard in each lane **acts as** temporary storage..."、"(2) RPDU **makes** a prune decision and decides which chunk to request"、"(3) PEC **makes** the partial exp to aggregate..."。
- 联作总结句："These modules jointly operates to make prune decision and creating the partial exp for aggregating the denominator in step 0."（先总后分再总；原句有语法瑕疵但结构可学）。
- **时序走查段**用 At first / At the same time / After that / If true, ... Otherwise, ... 显式时序连接词逐步走一个周期。
- 段末 Probability Generator 以 o_t 形成回环收口（呼应开篇两操作）。

### 对我们 §III 的差距结论
1. **相位对比句对（prompt vs generation, Conversely）**：我们方法节开头未说明级联的作用阶段；"dense prefill, sparse decoding" 只出现在实验设置节与 tab:settings，方法节读者要到 §III-B 深处（"In decoding, ..."）才得知。应在 §III 开头补一句阶段声明。
2. §3 机制→§4 模块 1:1 映射句：第 3 轮已建立（two explicit mechanisms 呼应 Intro two controls），达标。
3. 时序走查（At first/After that/If/Otherwise）：我们 L2 路径句已用 precedes/then 压缩表达，两栏页面预算下无需展开成整段，不改。
4. 组件编号定义（角色动词开头）：gate 小节已达标（"The optional gate tests whether..."）。
5. 配置数字前置：我们刻意集中到 tab:settings（Experimental Setting），结构选择不同，不改。

### 本轮修改（第 4 轮）
- paper_en.tex §III 开头段：在图句后补阶段声明句 "The cascade operates only during decoding; prefill remains dense."（dense prefill / sparse decoding 为 tab:settings 与实验设置节既有口径，无新事实）。
- paper_zh.tex 直译同步。

---

## §5 Experiments + §6 Conclusion（第 5 轮，2026-10-17，Token-Picker 最后一节）

### §5.1 Setup 三小节（claim 式标题 + 动作动词开头）
5.1.1 Algorithm / 5.1.2 Hardware / 5.1.3 Design Configurations——每小节以 "We evaluate / We set / We implement" 动作动词开头；**指标带方向定义**（"lower perplexity values indicate better performance"）；工具链一句一工具（RTL 综合 / CACTI / DRAMsim3 各带用途）；对照定义用 **lack-listing**（baseline "lacks five hardware modules: ..."——用缺失模块清单定义对照）；两配置名 + 各自 PPL 代价（+0.05 / +0.3）当场报出。

### §5.2 Result 三小节范式（对照我们 Q1-Q3 最直接）
**§5.2.1 Memory access reduction（标准结果段公式）**：
1. 图句带结论动词："Fig.8 **illustrates the impact of** our method on the reduction of off-chip access"；
2. **归因句**："Our probability estimation scheme selectively prunes only the unimportant tokens in each instance, **achieving** a 12.1× and 22.2× reduction"——结果=机制+achieving 数字；
3. **Moreover 第二机制句**：out-of-order → "resulting in a reduction of K accesses by 1.45× and 1.51×"；
4. **As a result 总收口**："our method achieves a total of 2.57× and 2.79×"。
5. 公平比较披露："For a fair comparison, we set the precision ... for both configurations."
6. **诚实例外句**："generally shows a better reduction ratio than SpAtten, **except in** the case of longer prompt setting where ..." → **However 重构句**："However, when evaluating both designs without fine-tuning, our scheme shows a 1.64× higher reduction"——承认例外后换条件重报，不是删掉例外。

**§5.2.2 Speedup**：承上机理句（"As outlined in Sec. 2.1.2, the workload is memory-bounded"）→ 归因句 + 数字（"yielding a speedup of 1.73×"）→ **"These results are due to the precise prediction"**（结果归因于机制）→ Integrating 第二机制 "leads to further benefits—a 1.32× increase"（破折号数字）→ **Moreover** 机理句（"hides the latency ... thus accelerating"）→ **trade-off 收口**："by accepting a minor algorithmic degradation, ToPick-0.3 achieves 2.48×/2.63×"→ "These results underscore the effectiveness of..."（小节收口句）。

**§5.2.3 Area & Power**：表句 → 归因句（modules "are designed to minimize V access ..., resulting in an area overhead of 1.0%"）→ Nevertheless 让步 → **Therefore 战略权衡收口**："demonstrates a strategic compromise, exchanging a moderate rise in power for a substantial improvement in energy efficiency."

### §6 Conclusion（Firstly/Secondly + 头牌数字回收）
"In this work, we address [痛点]。 Firstly, we propose [方法1]， achieving [数字]， while also creating [第二收益]。 Secondly, we design [方法2]。 In conclusion, our proposed method reduces 2.57×..., realizing a speedup of 2.28× and 2.41× energy efficiency."——结论=痛点回收+双贡献+头牌数字三连，无新信息。

### 对我们 Experimental Setting + Q1-Q3 的差距结论（诚实边界内）
1. **Q1 开头非 claim-first**：第一句是纯数据罗列（"The five-task means ... are 47.588, 47.910, 47.666"），T-P 结果小节第一句必带结论动词/方向。摘要已有 "a descriptive spread within 0.33 points" 的表述可复用——**本轮修改点**。
2. Q2/Q3 开头已达标（"Regional reservation changes which distant tokens remain reachable." / "The reported sparse configuration is slower than dense FullKV at every measured length."）。
3. 归因句（"These results are due to..."）：我们反向使用（"Aggregate medians cannot locate the cause"）——诚实边界决定不能加归因，不改。
4. 诚实例外→However 重构句式：我们 Q3 已有（"Although the relative penalty decreases with length, absolute sparse TPOT increases"），达标。
5. 小节收口句（"These results underscore..."）：我们用 "Thus these measurements show no decode-time benefit" 等对等句，达标。
6. Setup 三小节：我们用 \textbf{Workloads and comparisons.} / \textbf{Metrics.} 粗体段对等；指标带方向（"lower ... indicates better"）我们 Metrics 段隐含（unweighted mean），不必改。
7. 结论节：我们已含痛点回收+机制回收+诚实边界，头牌数字（22.45 点损失、3.38 比值）已入 Conclusion，达标。

### 本轮修改（第 5 轮）
- paper_en.tex Q1 开头：数据罗列句前加 claim-first 句 "The five-task means sit within a descriptive spread of 0.33 points across the three nominal scoring configurations."（0.33 来自既有 0.322，摘要同款表述），原句改列任务名。
- paper_zh.tex 直译同步。
