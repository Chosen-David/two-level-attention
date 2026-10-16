# DAC Best Paper 写作范式提炼（2019–2026 七篇全读）

> 2026-10-16 by Claude。材料：`agent_doc/advice/DAC_best_paper/` 七篇获奖 PDF（全文提取核对）。
> 用途：TLI（近远分区稀疏注意力）DAC 2027 投稿的表述参照；observation 段落轮询优化的评判基准。
> 用户指令："关注远端的 best paper，可以作为重要参考"。

## 0. 同领域标杆：Token-Picker（DAC 2024）——注意力剪枝加速，离我们最近

结构：Abstract → Intro（痛点→前作缺口→方法→量化贡献 bullet）→ §2 Background & **Motivation**（2.1 transformer/KV cache 铺垫；2.2 动机两个编号小节，**每节绑定一张图**）→ §3 Proposed Work → §4 Architecture → §5 Experiments → §6 Conclusion。

动机节范式（§2.2，两小节各一图一数字一结论）：
- Fig.2 内存分解："KV caching transfer is 7.8% at batch 1, it becomes **84.3%** at batch 64" → "Thus, minimizing latency in self-attention is essential..."
- Fig.3 实例间分布差异："only **4.6%** of tokens in instance A exceed 10⁻³, while **23.5%** in instance B" → "Therefore, fixed-ratio pruning could either remove dominant or keep unimportant tokens, undermining the overall performance and efficiency."
- **对我们的映射**：其"实例间差异否定固定比率"与我们 (b) 面板"查询间 far 分布差异否定单池"是同构论证（instance A/B ↔ 我们的排序捕获率曲线+配额保底）。DAC 评委接受并奖励这种"变异→否定固定策略→自适应/分区"的动机链。

公平性声明原文模板："For a fair comparison, we set the precision of Q, K, V to 12 bits and allow a +0.5 PPL in the Wikitext-2 dataset for both configurations."

工作点范式：ToPick（+0.05 PPL 保质量）/ ToPick-0.3（+0.3 PPL 换收益）双配置——**一个保守点一个激进点**，对应我们 4/128 紧凑候选与历史 128/128 配置的呈现方式。

## 1. 观察/Motivation 呈现范式（七篇横向）

四种被奖励的体例（按 DAC 出现频率排序）：

1. **折叠进 Intro 的编号痛点**（Miter 2026 最新惯例）："However, two challenges limit... First, ... Second, ..."，各 3–4 句、引文背书，不独立成节。观察→设计一步完成："We attribute that ... Rather than ..., we advocate ..."。
2. **Background & Motivation 合并节、每小节末尾落过渡句**（Gamora 2023）："Motivated by the analogy between X and conventional reasoning, we propose ..."；Token-Picker 同体例（"Thus/Therefore"收尾）。
3. **Intro 内加粗观察 bullet**（RL-CCD 2023）："• **Not all violating endpoints are equal.**"——先立"existing algorithms are flawed"再给 vital fact。
4. **图绑定观察、就地转设计**（DREAMPlace 2019）："We observe that ... (21.1% in Figure 3)" → 直接简化流程。

共性铁律：
- **每条观察绑定一个量化数字**（84.3%、4.6%/23.5%、90%、<0.04%）——我们的 0.84/0.51/0.24、0.00→0.27、24/36 已符合。
- **观察后必跟归因或设计蕴含句**（"We attribute this to..." / "Motivated by this observation, ..."）。
- **机理观察用小图讲透**（Miter 的 Table 1 AND/XOR 微型对照），大版面留给方法与消融。
- 回收句式：评估段用 "These observations imply several takeaways. First, ... Second, ..."（Gamora）或 "from which three key observations emerge"（Miter）。

## 2. 公平性与诚实披露模板句（可直接改写）

- AdaptivFloat："The training setup and the hyper-parameter recipe are kept the same for all five data types under evaluation **in order to impose a fair comparison**."
- AdaptivFloat 自证清白："**unlike several works [3] that intentionally skip** quantization on the sensitive first and last layers ..." → 我们版：unlike works that keep sink/attention-sink layers dense。
- RL-CCD："To perform fair and apple-to-apple comparison, we use the **same seed** in each run to completely remove non-deterministic run-to-run variation. Also, ... does not leverage any additional optimization step ... (**exact same recipe**)."
- Miter 2026 开销披露："The PAR2 score for 'Ours' includes both solving time and LUT mapping runtime, where the formulation overhead accounts for **< 10%**." → 我们应报选择器准备开销占 decode 时间比。
- DREAMPlace 失败记账：baseline 在 10M-cell OOM crash → 估算下界 9100s 写入表格并说明，不悄悄丢行。
- **摘要报反向 tradeoff**（AdaptivFloat）："0.9× and 1.14×"（能耗更好、面积更差并列）——我们摘要已报"稀疏 TPOT 高于稠密"，与此诚实传统一致，是加分项而非弱点。

## 3. 评估协议范式

- **双层结构是 DAC 标配**：DREAMPlace §4.1 e2e + §4.2 算子 microbench（我们的 kernel microbench + e2e 双口径铁律完全同构）。
- **质量门先行**：DREAMPlace 表格里 HPWL ratio 列在 speedup 前；先声明 "almost the same solution quality (within 0.2%)" 再报 30×。→ 我们应先立质量非劣（等 K、FullKV 参照）再报时延收益。
- **贡献-方法-消融 1:1:1 映射**（Miter 2026）：每贡献在 Methodology 有专属小节、在 4.3 有专属消融，消融小节标题复制贡献名。我们的分区/共享子空间/配额三组件应同样映射。
- **正交强基线证互补**（Miter 2026 引入 CCC）："our method is *complementary* to CNF-level simplification" → 我们可与 FlashAttention/量化叠加证明互补。
- Gemmini baseline 等配置声明："with the same number of PEs" → 我们的等 K 对齐即此传统。

## 4. 摘要与贡献句式

摘要流水线（Miter 2026 八句式）：痛点归因 → 观点主张（"We contend that X is as critical as Y"）→ 方法名 → 核心机制（对比破折号 "X—instead of Y—"）→ 技术列举 → 头牌数字 → 范式升华收尾。

贡献 bullet 惯例：
- 加粗名词短语开头（Miter："*An equivalence-preserving LUT mapping algorithm* that..."；Gamora："**Billion-node scalability and parallelism.**"）
- RL-CCD 把"发现新问题"列为第一贡献："We discover a new PD problem and demonstrate its importance."
- 数字双口径并列：max 与 avg 同报（"up to 64% TNS improvements (avg. 23%)"）。
- 头牌数字摘要-Intro-结果三处复现、口径一致（Miter 的 92.1%）。
- 预防性澄清句（DREAMPlace）："**To clarify,** the casting ... is orthogonal to ..."——抢答审稿人最可能的误解。
- RL-CCD 相关工作划界句："we are not comparing X with above methods as the focus of X is to improve ... rather than ..."

## 5. 对 TLI 论文的具体映射（当前状态对照）

| DAC 惯例 | 我们的现状 | 差距/行动 |
| --- | --- | --- |
| 观察绑定数字+归因句 | O 图 4 面板均有数字，intro 段已有 | 达标 |
| "变异否定固定策略"动机链 | (b) 面板单池 vs 配额正是此结构 | 达标（Token-Picker 同构） |
| 质量门先行再报速度 | 摘要/intro 已声明"未建立联合优势" | 等 E146b 匹配评估补齐后按"先质量非劣、后时延"重排 |
| 头牌数字三处复现 | 暂无（优势未建立） | 匹配评估出数后固定一个头牌数字 |
| 贡献-方法-消融 1:1:1 | 设计节两核心+一可选扩展 | 消融补齐后检查映射 |
| 开销占比披露 | TPOT 诊断已报、准备成本口径已声明 | 补"准备开销占 decode X%"具体数字 |
| 摘要报反向证据 | 已报稀疏 TPOT 高于稠密 | 达标（与 AdaptivFloat 诚实传统一致） |
| 双工作点（保守/激进） | 128/128 历史 vs 4/128 紧凑 | 等紧凑候选验证后按 ToPick 双配置体例呈现 |

## 6. 注意事项

- Token-Picker/Gamora 在图内放结论文字（Fig.3 内嵌 "only 48 tokens have attn prob over 0.001"）是 DAC 常见做法，但用户已明确要求我们图内不堆文字——用户偏好优先，此条不抄。
- 七篇均为 6–7 页短文，动机压缩、版面让给方法+消融；DAC 2027 若仍 6 页，O 图四面板占整页需评估压缩（可参考 Miter 的微型机理对照表思路）。
