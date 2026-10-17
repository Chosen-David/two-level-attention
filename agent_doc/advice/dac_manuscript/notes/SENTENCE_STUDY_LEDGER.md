# DAC Best Paper 逐句学习轮询账本（SENTENCE_STUDY_LEDGER）

> 用户 2026-10-16 指令建立的迭代协议。每小时一轮，每轮直接精读一篇 DAC best paper PDF 原文
> （不用提炼笔记），逐段逐句拆解（每句讲什么、什么功能），然后逐段遍历我们的投稿稿
> `source/paper_en.tex`（英文优先，中文 `paper_zh.tex` 直译同步），反思表述差距并 in place 修复。
> 循环直到学完七篇的最后一句话。诚实边界（未建立联合优势的披露）是项目硬约束，只优化句法与组织，不许删诚实内容。

## 轮询协议（每轮执行）
1. 读本账本，取「当前进度」的论文与章节。
2. 用 `PYTHONPATH=/tmp/pylibs python3 -c "from pypdf import PdfReader; ..."` 提取该 PDF 文本到 /tmp/<name>.txt（pypdf 已装在 /tmp/pylibs；若丢失则 `python3 -m pip install --target /tmp/pylibs pypdf`）。
3. 逐句拆解当前章节：原文句 → 功能标签（痛点归因/需求收口/前作功劳/However缺口/方法机制/编号列举/头牌数字/归因句/Thus收口）→ 对我们对应段落的启示。写入 `notes/sentence_studies/<论文>.md`。
4. 对照修改 `source/paper_en.tex` 对应段落（claim-first、主动语态、数字带方向、段落 Thus/Therefore 收口、先功劳后缺口、≤40词/句、无 hedging 虚词）。修改必须保守：只动表述，不动数字、不动诚实边界、不动 GPT 定域块（若与 observation 相关，遵守定域替换铁律）。
5. `source/paper_zh.tex` 同段落直译同步。
6. 编译验证：`bash build.sh paper_en paper_zh`（本环境缺 xelatex 时跳过，改为检查引用键存在 + LaTeX 语法目检，并在下轮补验）。
7. 更新本账本进度与修改记录。每轮至少完成一个论文章节 + 至少一处手稿修改，不许空转。
8. 每轮开始前 `git fetch` 检查远端新提交（GPT 发布频繁）；若有，先按上述合并流程择优合并再继续（远端为基底，本地独有内容叠加回去）。

## 论文队列与进度（7 篇）

| 顺序 | 论文 | 章节拆解进度 | 手稿对应段落已修 |
|---|---|---|---|
| 1 | 2024_Token_Picker.pdf | **全部完成**（Abstract/Intro/§2/§3/§4/§5+§6） | Abstract ✓ Intro-P1 ✓ Intro-(b)尾句 ✓ §III开头段 ✓（承接句+阶段声明句） Q1开头句 ✓ |
| 2 | 2019_DREAMPlace.pdf | **全部完成**（Abstract/Intro/§2/§3/§4） | Intro-P1 引用型 stakes 句 ✓ Related-Work-P1 However 缺口句 ✓ §III-D 承接句 ✓ Q3选择器段 claim-first ✓ |
| 3 | 2020_AdaptivFloat.pdf | **全部完成**（Abstract/Intro/§2/§3/§4/§5/§6+§7） | Intro-P2 机制归因句 ✓（后被远端 1010b06 收敛为诚实版，裁决采纳） Related-Work-P1 动词异质化 ✓ Related-Work-P2 appeal 句 ✓ §III-B β 参数-性质方向句 ✓（后被远端 c99315c 收敛为定理级方向，裁决采纳） Workloads 类型跨度句 ✓ 4/128 记号绑定句 ✓ 摘要尾句悬垂分词修复 ✓ |
| 4 | 2021_Gemmini.pdf | Abstract ✓ Intro ✓ §2 ✓；待：§3 生成器 | Intro-P5 Q1/Q2/Q3 前向指针 ✓ Related-Work-P1 定位句 enabling 从句 ✓ |
| 5 | 2023_Gamora.pdf | 未开始 | — |
| 6 | 2023_RL_CCD.pdf | 未开始 | — |
| 7 | 2026_Miter_Aware_LUT_Mapping.pdf | 未开始 | — |

## 修改日志

### 第 18 轮（2026-10-18，Gemmini §2 Background and Motivation）
- 远端检查：fetch 无新提交，本地 = origin/main（afbcde3 已合并含我第 17 轮提交；e1d584e 门控段 AVG 口径对齐、c84b5e9 Q2 β/K1=256 比较均已提交，工作区干净，无需裁决——内容与第 16 轮已采纳方向同源且边界完整）。
- 拆解笔记：§2 部分（开篇：路线图句带结论（roadmap 自带 motivating-the-need 结论）+ **Table I 特性对比矩阵**（8 生成器×三层特性打勾矩阵+表指针句）；§2.A：双层 to 手段-目的链+**分类句带双 exemplar**（TPU systolic / Brainwave+NVDLA vector）+Based-on-these-templates 前沿承接句；§2.B：**类别对比句**（实例 vs 生成器+improving 分词收益）+Here-we-discuss 三轴预告句+**三段同构需求模板**（must-provide 需求句→Most-today 局限句→making-it-challenging 评价困难后果→**In-contrast 编号能力句每项挂 enabling 从句**）+**双 persona 句对各挂点名前作**+first-infrastructure 声明句+**77% 行业证据句带 even-in-presence-of 让步与双归因**+Motivated-by-this-observation 回收句）。
- 对照结论：路线图句（claim-first opener 等价）、Table I（版式零余量+前作特性不可诚实补全，不可行）、类别对比句（无诚实类别对立）、persona/首创声明/行业数字（诚实边界不可用——AsyncTLS-already-combines 是首创声明的诚实镜像）、评价困难后果（nominal labels 句已有）、Motivated-by 回收句（Intro P4 已有）均不改。**唯一实缺 = Related Work 定位句缺 enabling 从句**（Gemmini 编号能力句每项带 to-handle/enabling 目的；且 Stage-4 集成检查项「页/词元关键抽象传播到 Related Work positioning」断裂——"page and token access are separate constraints" 只在 §III-B/Q2/结论出现）。
- paper_en.tex Related Work P1 定位句：句尾加 `, which keep page and token access separate.`（定义级事实＝§III-B 定理句，无新数字；为紧随其后的配额不保证句提供铺垫）。
- paper_zh.tex 直译同步（「……结合，使页访问与词元访问保持相互独立。」）。
- 版式核对：en 6 / zh 6 页保持，0 未定义引用。
- 待后续轮次：Gemmini §3 生成器（两级阵列微架构/数据流句式）、§4 评估、§5 案例研究。

### 第 17 轮（2026-10-18，Gemmini Abstract + Introduction —— 第 4 篇开始）
- 远端检查：无新提交；工作区干净（结论节 β 比较版已由 6490873/ba930db 等提交纳入，无需裁决——内容与第 16 轮已采纳的 242c821/e4c74f7 同源）。
- 拆解笔记：新建 `notes/sentence_studies/2021_gemmini.md`（Abstract 6 句表：**评价方法论缺口句**（"evaluated in isolation"——stakes=评价能力而非性能）/**This 后果句**（makes it difficult to appreciate）/To-address 登场句/能力枚举句（每项 that-clause 回扣缺口）/流片+头牌句（up-to 上界限定+点名基线）/开源披露句；Intro P1：However 双重目标句（嵌套关系从句 X 难造）/demand 归因句带引用簇/**While 让步缺口句**（一句内先功劳后范围限定）/**双 little-support 平行缺口句带 making 后果分词**；P2：登场句与摘要同构/**Unlike 对比句**（also 而非 only）/双工艺流片句/**诚实头牌句**（comparable 同级 claim+匹配条件限定语）/愿景句/**Specifically 案例预告句**；贡献 bullet：动词开头+**每条 Section 前向指针**+贡献 2 内嵌 verdict）。
- 对照结论：评价缺口链（摘要 "neither isolate nor establish" + Metrics 段）、While 让步句（Intro P1 but 句）、Unlike 定位（Related Work instead 句）、诚实头牌（Q2 0.09 分句+条件披露）、贡献内嵌 verdict（Intro P5 两 verdict 句）均有对等物；流片证据/愿景句/案例预告/设计空间跨度句诚实边界不适用。**唯一实缺 = Intro P5 评估预告句无 Q1/Q2/Q3 前向指针**（Q 标签只在评估节小节标题出现，Intro↔评估节导航断裂；Gemmini per-bullet Section 指针模式）。
- paper_en.tex Intro P5：`The evaluation examines task quality, regional coverage and execution cost.` → `The evaluation examines task quality (Q1), regional coverage (Q2) and execution cost (Q3).`
- paper_zh.tex 直译同步（「评估考察任务质量（Q1）、区域覆盖（Q2）与执行成本（Q3）。」）。
- 版式核对：en 6 / zh 6 页保持，0 未定义引用。
- 待后续轮次：Gemmini §2 背景动机（Table I 生成器对比矩阵/77% CPU 时间动机数字句/Motivated-by 观察回收句/In-contrast 编号枚举）、§3 生成器、§4 评估、§5 案例研究。

### 第 16 轮（2026-10-18，AdaptivFloat §6 Hardware Evaluation + §7 Conclusion —— 第 3 篇拆解完毕）
- **远端裁决**（fetch 发现 242c821/e4c74f7，本地已同步）：(1) 242c821 把 8B β 分配比较（β.125→.0625，AVG 58.30→59.29，页/词元配额 16/112→8/120、640/252→320/572 同报）整合进 Q2——**采纳**，完整披露「比较不能隔离页与词元分配作用」+ 非劣性边界；(2) e4c74f7 摘要重写为三实测数字证据版（58.30/59.29 + 50.64/52.22）——**采纳**，Token-Picker 头牌数字句范式的诚实镜像。
- 拆解笔记：§6+§7 部分（setup：负载-架构匹配理由句闭环 §1 Figure 1/公平比较声明句（"the two designs employ the same evaluation methodology"）；结果：**精确区间句**（"0.97× to 0.90×" 双端点）/**stems from the fact that 机制归因句**/On-the-other-hand 竞争者胜出句/**证实性交叉验证收口句**；结论：**tangible 节制词收口**——量化收益用 tangible 限定而非形容词堆砌）。
- 对照结论：公平比较声明（Metrics 段"受控比较须匹配……"已对等）、精确区间（TPOT 比值 3.38/2.46/1.78 序列同构）、交叉验证收口（诚实边界禁造新验证）。**唯一实缺 = 摘要尾句悬垂分词**："matched quality and complete-request cost, including preparation" 中 "including preparation" 语法上可挂到 "quality" 而非预期的 "cost"。
- paper_en.tex 摘要尾句：改 "matched quality and preparation-inclusive complete-request cost"（消除歧义，词数不变）。
- paper_zh.tex 直译同步（「含准备开销的完整请求成本」）。
- 版式核对：en 6 / zh 6 页保持，0 未定义引用。
- **AdaptivFloat 七节全部拆解完毕**（第 11-16 轮），累计手稿修改 6 处（机制归因→远端收敛、动词异质化、appeal 句、β 方向句→远端定理级、类型跨度、记号绑定、悬垂分词），en/zh 同步。下轮开始第 4 篇 **2021_Gemmini.pdf**（Abstract+Intro）。

### 第 15 轮（2026-10-17 23:0x，AdaptivFloat §5 PE Architecture）
- **远端裁决**（fetch 发现 2ff5457/e50d061/ff5ea05，本地已同步）：(1) e50d061 把第 14 轮登记待议的 E144a 32B 比较整合进 Q2（AVG 50.64 vs 52.22 / -1.58 分 / 逐任务方向 +11、-8.80、-7.33 / 双臂零分披露 / 跨机器与未重放披露 / neither-isolates-nor-establishes 收口）——**采纳**，第 14 轮登记的内容级决策已由远端完成且边界完整；`\newpage` 改自然续排；(2) 2ff5457 Metrics 段重写为 AVG 主指标口径（"Per-task gains and losses explain AVG; zero loss on every task is not required"）——**采纳**，与 E 系列实验口径一致。
- 拆解笔记：§5 部分（开篇：**能力→架构桥接句**（paves the way——§4 结果铺路 §5 设计）/**混合权衡声明句**（compromise between float accuracy and fixed-point density）/**基线代表性句**（NVDLA-like "observed in the plurality of commercial ML accelerators"）；§5.1：公平基线全描述+**记号绑定句**（"will be referred to as INT8/24/40 to designate..."）；§5.2：数据流走查+**静态/动态分岔句**（while weights post-training, activations offline stats）+**A-key-contrast 句带 in-turn 因果链**+前向引用句）。
- 对照结论：桥接句（Intro P5）、静态/动态分岔（gate 冻结句）、A-key-contrast（Because softmax precedes...）均有对等物。**实缺 = 记号绑定句**：全文反复使用 "compact 4/128-protection implementation/design/candidate"（摘要/Q1/Q2/Q3/结论），但 4/128 从未与 four prefix + 128 recent 显式绑定。
- paper_en.tex Experimental Setting 首句：加 ", referred to throughout as the 4/128-protection design"（AdaptivFloat INT8/24/40 句式压缩形，无新数字）。
- paper_zh.tex 直译同步（「，下文统称4/128保护设计。」）。
- 版式核对：en 6 / zh 6 页保持（记号绑定为句尾短语未触发溢出）。
- 编译验证：双语通过，0 未定义引用。
- 待后续轮次：AdaptivFloat §6 硬件评估（合成流量/能耗面积表）+ §7 结论。

### 第 14 轮（2026-10-17 21:0x，AdaptivFloat §4 Experimental Results）
- **远端裁决**（fetch 发现 c99315c/10a3cd7 等，本地已同步）：(1) c99315c 删除我第 7 轮 However 句（"without guaranteed minimum capacity for either region"），改为 SparQ 近期窗口/HISA 首尾块细粒度功劳+「nominal near--far L1 page quotas... do not by themselves guarantee final far-token capacity」——**采纳**：我的原句是过度概括（SparQ 近期窗口即近区保底），远端版 credit-then-gap 更准确；(2) β 方向句我的「enlarges/shrinks」→远端「cannot reduce/increase; rounding can leave both unchanged」——**采纳**：round 非严格单调，我的版本是数学错误，第 11 轮判例第二次验证；(3) 10a3cd7 Token-Picker 句机制精确化——**采纳**。
- 拆解笔记：§4 部分（setup：**选择理由内嵌句**（distinct types + narrow-to-wide spread 当场声明）/实现披露句/**in-the-sense-that 类别定义句**/公平性句（in order to impose a fair comparison）/**反取巧披露句**（quantize all layers unlike [3] skip sensitive layers——把自己协议与前作取巧协议划界）；§4.1：指标递进开场/图结论句/**例外当场限定**（BFP spread thinnest — although higher mean）/**例外→假设句**（suggests BFP would fare best）/小节桥接句；§4.2：**要点蒸馏句**（The key observation we can distill）/头牌+due-to 归因/**反推设计原则句**（it is critical to have...）/算法→硬件收益链/**惊喜上报句带机制引用**（exceeding FP32 by up to +0.3）；§4.3：失败 as 从句归因）。
- 对照结论：反取巧披露、失败 as 归因（我们显式不可归因）、公平性需求句均有对等物；例外-限定句式需逐任务数据（S4 仅三配置均值）禁造不加；**正向惊喜上报**：远端 E144a/E145/E146/E147 系列实验已产生配对正向证据（TLI vs FullKV RULER 32K +0.79、两级 vs 单池 LBv2 +2.59/RULER64K +2.77）——**是否整合进手稿（替换"lacks matched evaluation"口径）是用户/GPT 级内容决策，本轮仅登记待议**。实缺 = Workloads 段五任务无类型跨度声明。
- paper_en.tex Workloads 段：加括号形类型跨度 "(question answering, summarization and code completion)"。
- paper_zh.tex 直译同步（「（问答、摘要与代码补全）」）。
- **版式教训（重要）**：初版 "spanning multi-hop and single-document..." 长 10 词使 en 6→7 页——`\newpage` 平衡布局（远端 90edbf2 验证版）余量为零，**任何加句必须编译后核对页数**，超页即压缩为括号形或删词。压缩后 6 页恢复。
- 编译验证：双语通过（en 6 / zh 6 页），0 未定义引用。
- 待后续轮次：AdaptivFloat §5 PE 架构、§6 硬件评估、§7 结论；E144a-E147 正向证据整合待用户决策。

### 第 13 轮（2026-10-17 19:0x，AdaptivFloat §3 Methodology）
- **远端裁决**（fetch 发现 107bfb6/58d8457 等，本地已同步）：107bfb6 把 Design implications 从「Three constraints」清单改为「two distinct constraints + These are properties of the defined rules」+第三条挪入「A useful operating point additionally requires...」运行点要求——**采纳**：前两条是定义规则的固有性质（可从公式推出），第三条是未满足的实测要求，混排是层级混淆，对标 best paper 证据分层收口。58d8457：SparQ「query-selected key coordinates」术语精确化（改进我的第 12 轮句）+ Conclusion 区分 Qwen3 整箱诊断与历史配置两类证据来源——**采纳**。
- 拆解笔记：§3 部分（§3.1：格式依托句→设计决策句→**问题句**（决策代价当场命名）→**牺牲换位句**（sacrifice ±min for ±0 + 图 2 佐证）→if/otherwise 规则句→公式+where→**类比锚定句**（similar to integer quantization scale）→**对比收益句带前向引用**（"A benefit of exp_bias is... compared to... Discussed in Section 5"）→**参数-性质方向句**（narrower datapoints → more negative exp_bias）；§3.2：问题重定义句（格式定了→问题降维为映射）→**analogous-but 句式**（类比+一句话点差异）→算法走查句链（first/then/before/finally 串联 Algorithm 1））。
- 对照结论：牺牲换位句（无对应决策场景）、类比锚定（第 7 轮已确认达标）、analogous-but（已有对比句）、算法走查（gate 小节时序句达标）。**「A benefit of」句式判定不加**：AdaptivFloat 有实测支撑，我们的 32 维 vs 128 维填零是历史配置差异且收益未实测，诚实边界禁。**实缺 = 参数-性质方向句**：α/β/γ 段纯定义无方向性——β↑→P_n↑→P_f↓ 是纯代数事实（第 12 轮判例的定理级门槛达标），且为 Q2 的 β=.375→k_f=0 后果句提供方法节铺垫。
- paper_en.tex §III-B：ties-to-even 句后加 "Raising $\beta$ enlarges the near page reservation and shrinks the far one, since $P_n$ grows with $\beta K_1$ while $P_f$ exhausts the same cap."
- paper_zh.tex 直译同步（「提高$\beta$会扩大近区页预留并压缩远区……」）。**修复自身失误**：zh 编辑曾误删 k_n/k_f 定义句，当场发现并恢复（编译前 grep 双语定义句各 1 处确认）。
- 编译验证：双语通过（en 6 / zh 6 页），0 未定义引用。
- 待后续轮次：AdaptivFloat §4 实验表（精度-任务矩阵行=比特宽度、列=数据格式）、§5 PE 架构、§6 硬件评估、§7 结论。

### 第 12 轮（2026-10-17 17:0x，AdaptivFloat §2 Related Work）
- **远端裁决**（fetch 发现 90edbf2/1ba9faf/7021966/b93f0dd/7f3b182/1010b06，本地已同步）：1010b06 把第 11 轮我的 Intro P2 归因句 "The asymmetry is mechanical: ... misses more pages while losing less total mass" 收敛为 "Mean pooling **can** dilute... **Neither property determines the observed ordering**"。**采纳远端**——对标 AdaptivFloat P1 机制归因句的前提是定理级机制（LayerNorm 不重参数化→权重绝对值大可证），我们的稀释是启发式级趋势，"mechanical" 属过度断言，远端版正是诚实边界要求。判例教训：**机制归因句只适用于可严格推导的性质，启发式性质只能 can-动词+不决定次序**。Q3 逐轮差值（0.0237/0.0046 ms）与 gate 段代理/质量区分重写为诚实增强，保留。
- 拆解笔记：§2 部分（类别一**前作动词异质化**：[24] noticed+proposed+成本披露三连/[26] observed+can be used to+yielding/[22] performs in search of/Stripes performs to enable——每篇动词携带其贡献性质；类别收尾 **Rather-than 定位句**（他们改宽度、我们在固定宽度内调表示）；类别二 **appeal 句→However 失败机制句**（appeal stems from...→by collapsing the exponent... smaller magnitudes more prone to data loss）+ 近距离对比句；类别三互补性声明句划分比较域）。
- 对照结论：互补性声明句已有对等物。**实缺两处**：(1) P1 三选择器统一动词 "use query-aware approximate selection" 一刀切（平铺列举病的动词层残留）→ 按 Quest 构造保证/SparQ 近似/HISA 分层差异化；(2) P2 Token-Picker 句缺 appeal 先行（limits reads 先功劳再 instead retains 分歧）。
- paper_en.tex Related Work P1：改 "Quest supplies sign-aware page-extremum bounds for far-page scoring, SparQ approximates attention from partial key bits, and HISA prunes the attention set hierarchically"（原 "We adopt Quest's sign-aware page extrema" 信息保留于第一分句）。
- paper_en.tex Related Work P2：改 "Token-Picker **limits selector reads by** estimating... and scheduling... on demand; our coordinate-based scorer instead retains..."。
- paper_zh.tex 直译同步（「Quest以符号感知页极值为远区页评分提供构造保证，SparQ从部分键比特近似注意力，HISA则按层级剪枝注意力集合」「Token-Picker通过……来限制选择器读取量」）。
- 编译验证：双语通过（en 6 / zh 6 页），0 未定义引用。
- 待后续轮次：AdaptivFloat §3 方法（指数偏置机制句式）、§4 实验表、§5 PE 架构、§6 硬件评估、§7 结论。

### 第 11 轮（2026-10-17 15:0x，AdaptivFloat Abstract + Introduction —— 第 3 篇开始）
- 远端检查：无新提交（本地 = origin/main 95587a0）。
- 拆解笔记：新建 `notes/sentence_studies/2020_adaptivfloat.md`（Abstract 5 句表：A1 痛点机制归因句（as 从句）/A2 方法句带粒度限定语/A3 consistently+四点名基线+精度范围限定/**A4 例外反转句**（only 2.1 degradation vs total accuracy loss——同句完成悲观/乐观双口径）/**A5 诚实双数字句**（0.9× 能耗与 1.14× 面积并排报）；Intro P1 五句链：目标语境→前作聚类→**评价范围批评句**→However+图证据→**The reason for this phenomenon is that 机制归因句**（batch norm vs layer norm 双机制）；P3 Unlike 对比带对方失败机制+However 背离 IEEE 754 声明带收益；P4 Rather-than 反范围句；P6 良性性质注记句+self-supervised as 从句）。
- 对照结论：A5 好坏双数字并排（Q3 比值序列）、评价范围批评、Unlike 对比（第 7 轮 However 句）均有对等物；例外反转与 generalized blueprint 主张诚实边界不适用。**唯一实缺 = Intro P2 漏页-质量不对称无机制归因句**（avg 漏 13 页载 59.65% 质量 vs minmax 漏 17 页载 26.46%，数字已报但缺 The-reason-is-that 式解释）。
- paper_en.tex Intro P2：加 "The asymmetry is mechanical: mean pooling dilutes pages whose mass concentrates on few tokens, whereas the extrema bound upper-bounds each page's best token score and so misses more pages while losing less total mass."（机制由 §III-B eq:bound 性质与均值池化稀释算术推出，无新数字，AdaptivFloat P1 机制归因句式）。
- paper_zh.tex 直译同步（「这种不对称有其机制根源：……」）。
- 编译验证：双语通过（en 6 / zh 6 页），0 未定义引用。
- 待后续轮次：AdaptivFloat §2 相关工作（三类聚类：Adaptive Precision / DL on Quantization / posit）、§3 方法、§4 实验表。

### 第 10 轮（2026-10-17 13:0x，DREAMPlace §4 Experimental Results —— 第 2 篇拆解完毕）
- 远端检查：无新提交（本地 = origin/main 9837d79）。
- 拆解笔记：`notes/sentence_studies/2019_dreamplace.md` §4 部分（setup 六组件：工具链披露/己方硬件/**让分披露句**（单线程约束使加速更强）/基线环境/负载/**局限→上行句**；§4.1：**质量门先行头牌句**（within 0.2% 前置 + 35×/43× + 点名基线 40 线程）/崩溃+potential reason 谨慎归因/**下界估算句带内联算术**（119+1000∗9≈9100s）/分解定位句（76% 在 DP）/量化未来句（15×=除法式当场展示）/占比回扣设计句（67.8% v.s. 32.2%）；§4.2：**下沉声明句+臂命名 denotation 句**——microbench 用 1.9×/1.4× **回收 §3.1 Although 分歧句闭环**+诚实例外→still 句；表底 ratio 行）。
- 对照结论：质量门先行/谨慎归因/估算算术/Although→still 均有诚实镜像对等物；分解百分比与未来潜力数字禁造。**唯一实缺 = Q3 "Local selector cost" 段数据罗列开场**（与第 5 轮 Q1 同款病）。
- paper_en.tex Q3 选择器段：结论句前置 "The shared-subspace selector has no lower reported time than the nominal full-dimensional page-scoring variant in the synthetic study."，原 thus 句删除，数据句随后。
- paper_zh.tex 直译同步（「在合成研究中，共享子空间选择器的所报时间未低于名义全维页评分变体。」前置）。
- 编译验证：双语通过（en 6 / zh 6 页），0 未定义引用。注意：本会话 shell PATH 为空，编译前需 `export PATH="/usr/bin:/bin:/usr/local/bin:~/texlive/2026/bin/x86_64-linux:$PATH"`。
- **DREAMPlace 四节全部拆解完毕**（Abstract/Intro/§2/§3/§4），累计手稿修改：Intro-P1 stakes 句、Related-Work However 缺口句、§III-D 承接句、Q3 选择器段 claim-first，en/zh 同步。下轮开始第 3 篇 **2020_AdaptivFloat.pdf**（Abstract+Intro）。

### 第 9 轮（2026-10-17 12:4x，DREAMPlace §3 The DREAMPlace Algorithms）
- 远端检查：无新提交。
- 拆解笔记：`notes/sentence_studies/2019_dreamplace.md` §3 部分（开篇：差异披露句+设计决策双数字当场论证（<0.04% 质量差+21.1% 省时）+范围声明句；§3.1：As 依托句+For brevity 限定句+**数值陷阱句**（naive overflow→等价变换→"which is an equivalent transformation"）+朴素方案否定句（双原因）+**Noting 规模归并句**（|P|≫|E|→pin-level）+**Although 前作建议分歧句**（[15] 建议 SpMV 但 atomic add 实测更快+as 归因+前向引用）；§3.2：**another 承接句**+四步列举+热点步 since 归因）。
- 对照结论：数值陷阱句（我们的量化披露是类型披露非陷阱披露，不适用）、Although 分歧句（Related Work P2 "instead retains full-dimensional KV" 已对等）、规模归并句（eq:bound 性质→保证句已对等）。**唯一实缺 = 小节间并列衔接句**——§III-C→§III-D 无衔接，gate 小节冷启动开场。
- paper_en.tex §III-D 开头：加承接句 "Beyond scoring and budgeting, layer variation offers a second savings opportunity."（DREAMPlace "another bulk computation" 式；呼应 Intro P4 "historical layer variation in panel (d) motivates this extension"）。
- paper_zh.tex 直译同步（「在评分与预算之外，层间差异提供第二类节省机会。」）。
- 编译验证：双语通过，0 未定义引用。

### 第 8 轮（2026-10-17 10:3x，用户指令：文法机械审计 + 图美化）
- **文法审计**（gate_mechanical 式，165 句全文扫描）：>40 词句仅 1 处（摘要尾句 44 词）——拆分为动作先行句 "Establishing a beneficial operating point therefore requires evaluating the compact 4/128-protection implementation with matched quality and preparation-inclusive cost."（en/zh 同步）。hedging 命中（may/rather）与被动命中 14 处逐条目检均为合法用法（许可义 may、对比义 rather than、状态性系表 is disabled by default 等），不虚改。
- **图美化**（保守：不动数据/数字/结论）：(1) build_evidence_figures.py rcParams 字号 base 8→8.5、legend/tick 7→7.5（3.35in 列宽图缩印可读性）；(2) ruler128k 标题两行限定语压一行（"execution unverified" 警示仍在 caption 保留）；(3) decode_tpot y 上限 140→130（数据最大 119.9，去顶部松留白）；(4) build_evidence_figures.py 与 build_remote_observation.py 的 inkscape 调用加 cairosvg 兜底（本地无 inkscape，PyPI 可达）。四张 evidence 图 + o_observation 主图双语全部重生成（10:33/10:34）。
- 编译验证：双语重编译通过，0 未定义引用，页数不变（en 6 / zh 6）。
- 环境：matplotlib 3.11.2 装入 .venv-paper；cairosvg 装入（SVG→PDF 兜底，验证输出尺寸与 inkscape 一致 241x155pt 等）。

### 第 7 轮（2026-10-17 08:5x，DREAMPlace §2 Preliminaries；本轮前完成远端对齐）
- **远端对齐（用户指令）**：fetch 发现 471a83b/7a7ebc8/3628501/57b3039 四新提交。裁决 471a83b（Metrics 段把 retention 拆成 (b)/(c) 两图各自分母定义）：**采纳**——精确定义符合 define-before-use 铁律。本地第 3-6 轮全部修改先提交（ab95268），merge origin/main（44df413），PDF 二进制冲突重编译解决，五处标记句全部存活，双语编译 0 未定义引用。
- 拆解笔记：`notes/sentence_studies/2019_dreamplace.md` §2 部分（§2.1 术语铺垫+在哪耗时定位句；§2.2 **类比小节十五句链**：As 归并句→类比声明→图句→网络侧定义→换元声明（"we combine cell locations into w for brevity"）→替换句→推导句→带理由对应句→one-to-one mapping 收口+价值主张；尾段 toolkit 三层栈+only-need 工作量最小化句+Thus 痛点收口；§2.3 范式句+引用功劳+**参数-性质-代价三元组**（γ 句式）+公式→实现需求衔接句（"requires DCT/IDCT routines"））。
- 对照结论：换元声明句、公式→实现衔接句、As 归并句、公式 Thus 收口均已有对等物（第 3-6 轮累积修复）；参数-性质三元组诚实边界下不加。**转向第 1 轮遗留项：Related Work P1 缺 However 缺口句**（平铺列举无先功劳后缺口结构）。
- paper_en.tex Related Work P1：AsyncTLS 功劳句后插入 "However, these selectors rank candidates in a single pool, without guaranteed minimum capacity for either region."（缺口事实来自 Intro (b) 已确立结论，与"配额是我们的组合部分"既有定位一致）。
- paper_zh.tex 直译同步。
- 编译验证：双语通过，0 未定义引用。
- 待后续轮次：DREAMPlace §3 实现节（自定义算子节组织）、§4 实验节。

### 第 6 轮补充（2026-10-17，用户指令：不服从远端，批判性裁决）
- **配额尾部句恢复裁决**：远端 7d78c71 删除 "Quotas act on the tail rather than the mean..."（en/zh），经对标 Token-Picker §2.2.2 判定**本地版正确，已恢复**。四点理由：(1) 0.00→.27（arm minima, unpaired）登记于 O_observation_meta 且画进图 (b) 图例——正文删句造成图文脱节，删句者未同步改图；(2) Token-Picker 双端量化传统（"only 4.6%, while 23.5%"）要求均值/尾部/下降三口径齐报，删句丢失尾部口径；(3) 远端版自身漏洞：收口句 "This **controllable** coverage trade-off" 的 "controllable" 失去量化支撑——.27 下界恰是可控性的证据；(4) 非配对担忧已由句内 "(arm minima, unpaired)" 披露消解，删信息是过度保守。远端对该句的处置（删除）降级为不采纳；远端同批的其他重写（(a) 段、caption 防过度推断句等）未经逐句对标，保留现状待后续轮次裁决。
- 协议更新：远端合并原则从「尊重远端审阅版」改为「**批判性择优**——对标 best paper 逐点裁决，敢于采纳本地版」。
- 编译验证：恢复后双语编译通过，0 未定义引用。

### 第 6 轮（2026-10-17 06:5x，DREAMPlace Abstract + Introduction —— 第 2 篇开始）
- 拆解笔记：新建 `notes/sentence_studies/2019_dreamplace.md`（Abstract 4 句表：stakes/方法核心变换动词 casting X equivalently to Y/头牌数字+免损失限定+点名基线/We believe 愿景收口；Intro P1 双层 stakes 链+Therefore 收口；P2 范式点名句+Although 缺口；P3 前作段每作自带加速度数字+质量代价双口径、三层递进缺口（线程饱和→GPU 先例→生态缺失）；P4 摘要-登场句同构+orthogonal to 范围澄清句+开源披露）。
- 远端检查：无新提交（HEAD 仍 d1489fc）。**重要发现：远端 7d78c71（经 ca4ca8f 合并）的审阅版重写已移除第 2 轮的配额尾部句**（"Quotas act on the tail..."，摘要亦被进一步压缩）。（后续按用户指令批判性裁决：恢复本地句，见上方第 6 轮补充。）
- 差距结论：摘要愿景句不加（与诚实边界相邻，省略更稳）；范式点名句/前作数字段（诚实边界禁补造数字）/orthogonal 澄清句均有对等物或不适用；**唯一实缺 = Intro P1 缺带引用的世界事实句**（第 1 轮曾加、被远端重写移除）。
- paper_en.tex Intro P1 开头：补 `Long-context benchmarks such as LongBench and RULER evaluate models on tasks with long input histories~\cite{longbench,ruler}.`（无新数字新事实，两引用键既有）。
- paper_zh.tex 直译同步。
- 编译验证：compile_local.sh 双语通过，0 未定义引用。
- 待后续轮次：DREAMPlace §2 Preliminaries（背景节句式）、§3 实现节、§4 实验节。

### 第 5 轮（2026-10-17 04:5x，Token-Picker §5 Experiments + §6 Conclusion —— 第 1 篇拆解完毕）
- 拆解笔记：`notes/sentence_studies/2024_token_picker.md` §5+§6 部分（setup 三小节 claim 式标题+动作动词开头+指标带方向定义；**结果段标准公式**：图句带结论动词→归因句（"achieving 12.1×..."）→Moreover 第二机制→As a result 总收口；公平比较披露句；**诚实例外句→However 换条件重构**（"generally better, except in longer prompt... However, without fine-tuning, 1.64×"）；trade-off 收口 "strategic compromise, exchanging a moderate rise for a substantial improvement"；结论=Firstly/Secondly+头牌数字回收无新信息）。
- 远端检查：无新提交（HEAD 仍 d1489fc）。
- 差距结论（诚实边界内）：Q2/Q3 开头已 claim-first；归因句我们反向使用是诚实边界决定的（"cannot locate the cause"），不加；例外→However 重构、小节收口句、结论节均达标。**唯一实缺：Q1 开头是纯数据罗列句**（T-P 结果小节第一句必带结论动词/方向）。
- paper_en.tex Q1 开头：新增 claim-first 句 "The five-task means sit within a descriptive spread of 0.33 points across the three nominal scoring configurations."（0.33=既有 0.322 四舍五入，摘要同款表述），数据句重排为 "On [任务名], shared-subspace scoring scores 47.588, ..."。
- paper_zh.tex 直译同步（「五项任务均值在三种名义评分配置间的描述性差距不超过0.33分。」）。
- 编译验证：compile_local.sh 双语通过，0 未定义引用，PDF 04:57 产出。
- **Token-Picker 七节全部拆解完毕**（Abstract/Intro/§2/§3/§4/§5/§6），累计手稿修改：Abstract 重写、Intro-P1 stakes、Intro-(b) 尾部句、§III 承接句+阶段声明句、Q1 开头句，en/zh 同步。下轮开始第 2 篇 **2019_DREAMPlace.pdf**（Abstract+Intro）。

### 第 4 轮（2026-10-17 03:0x，Token-Picker §4 TOPICK ARCHITECTURE）
- 拆解笔记：`notes/sentence_studies/2024_token_picker.md` §4 部分（开篇 3 句表：段落级 claim + 模块回指 §3 机制 1:1 映射 + We design 句 + 配置数字前置；两操作数据流框架 + **Conversely 相位对比句对**（prompt 预载 vs generation 流式）+ 路线图句；编号模块列举（角色动词+何时+去向）；§4.1 图先行 + 组件编号定义 + 联作总结句 + At first/After that/If/Otherwise 时序走查 + o_t 回环收口）。同时修正笔记内 §3/§4 顺序颠倒问题。
- 远端检查：fetch 发现 d1489fc、ca4ca8f（均未改 tex），本地与 origin/main 的差异仅为第 3 轮我的修改，无需合并。
- 差距结论：机制→模块映射句（第 3 轮已建）、时序走查（已用 precedes/then 压缩）、组件角色动词句（gate 小节已达标）均无需改；**唯一实缺 = 相位声明**——方法节开头未说明级联作用阶段，"dense prefill, sparse decoding" 只在实验设置节与 tab:settings，读者要到 §III-B 深处才得知。
- paper_en.tex §III 开头段：图句后补 "The cascade operates only during decoding; prefill remains dense."（T-P 相位对比句对的压缩版；既有口径无新事实）。
- paper_zh.tex 直译同步（「级联仅在解码阶段执行；预填充保持稠密。」）。
- 编译验证：compile_local.sh 双语通过，0 未定义引用，PDF 03:00 产出。
- 待后续轮次：Token-Picker §5 Experiments（评估节组织：setup 三小节拆分 / 图句结论式 / 归因句），对照我们 Experimental Setting + Q1-Q3 节。

### 第 3 轮（2026-10-17 01:0x，Token-Picker §3 Proposed Work）
- 拆解笔记：`notes/sentence_studies/2024_token_picker.md` §3 部分（开篇 6 句表：需求承接/However 内在障碍/we propose 登场/if-otherwise 双分支/双挑战编号预告；§3.1 三段式含 Since→Therefore 安全性收口与 In conclusion 小节双句收口；§3.2 编号过程 + For example 时间线走查 + Therefore 三收益收口）。
- 远端检查：fetch 发现 9d82537（caption 措辞修正，**已采纳我第 2 轮的配额尾部句进 origin/main**：\"stored selections\"、\"a 2048-token cap\"）与 3fd1d38（仅 advice 文档）；本地 tex 与 origin/main 一致，无需合并。注意 zh 版方法节被远端重写过（§3 开头结构变了），修改按当前文本同步。
- 差距结论：我们已有对等物——双挑战预告=Intro Q1/Q2/Q3+claim 式小节标题；if/otherwise 双分支=gate 小节句；数学安全收口=§III-B eq:bound 段。唯一实缺：**§III 开头缺需求承接句**（T-P \"To counteract the described variability...\" 式缺口回收）。
- paper_en.tex §III 开头段：原第一句拆为两句——新增承接句 `To control both selection cost and candidate access, the cascade uses two explicit mechanisms: a shared scoring view and regional page quotas.`（selection cost/candidate access 均为全文既有术语；two explicit mechanisms 呼应 Intro P4 的 two explicit controls）+ 机制句保留原信息（32 坐标映射 → separating 分词结构）。
- paper_zh.tex 直译同步（「为同时控制选择成本与候选访问，级联采用两项显式机制：共享评分视图与区域页配额。……」）。
- 编译验证：compile_local.sh 双语通过，0 未定义引用，PDF 01:03/01:04 产出。
- 待后续轮次（Token-Picker 剩余）：§4 Architecture（架构节组织，对照我们图/caption 组织）、§5 Experiments（评估收口范式，供 Q1-Q3 节对照）。

### 第 2 轮（2026-10-16 20:2x，Token-Picker §2 Background & Motivation）
- 拆解笔记：`notes/sentence_studies/2024_token_picker.md` §2 部分（2.2.1 六句表 / 2.2.2 十二句链，含归因句与双缺口总收口范式）。
- 远端检查：539342c 已被并行会话同步进工作区（tex 与 origin/main 一致，含 (a) 段量化锚点 13/32、59.65%、17/32、26.46%——与 O_observation_meta 一致），本轮无合并需求。
- paper_en.tex Intro (b) 段新增一句：`Quotas act on the tail rather than the mean: the single pool drives far retention to zero for some queries, while the quota arm never falls below .27 (arm minima, unpaired).`——数字来自图 (b) 已登记的 "arm minima 0.00→0.27 (unpaired)" 标注（O_observation_meta），非新造；补齐 Token-Picker 式双向量化（均值微升 / 尾部保底 / 43 条下降三口径齐报），同时落实「配额=防挤出保底非均值增益」的项目口径。
- paper_zh.tex 直译同步（「配额作用于尾部而非均值…（两臂各自最小值，非配对）」）。
- 编译验证：compile_local.sh 双语通过，第三遍 0 未定义引用，PDF 20:26 产出。
- 待后续轮次（Token-Picker 剩余）：§3 Proposed Work（方法句式）、§5 Experiments（评估收口范式，供 Q1-Q3 节对照）。

### 2026-10-16 18:4x 远端合并（用户授权自行评定择优）
- 远端 GPT 发布 be04309（scholarly revision）：AsyncTLS 已有组件的诚实定位、预算公式 P_n/P_f/k_n/k_f、Q 小节 claim 式标题（"Does reduced scoring width preserve task quality?"等）、Related Work 加 Token-Picker 对比。**整体优于本地旧版，采纳为基底**（21 个文件 checkout 自 origin/main）。
- 本地独有内容**全部保留**并重新叠加：(1) 稀疏基线表 tab:baseline-comp + 段落（远端从未见过，插回 Q1 节）；(2) Token-Picker 式摘要三句——A1 成本归因（"selection cost therefore competes with the attention work it removes"）、A3 Although 缺口句（"they score candidates in the full coordinate width and rank pages in a single pool"——一句同时论证共享坐标与区域配额两个机制的必要性）；(3) Intro P1 stakes 句（长上下文负载 + 引 longbench/ruler/quest/sparq）。
- 合并后摘要 = 远端骨架 + 远端诊断数字（.6759→.7207、43/128、预算配额零）+ 本地 Token-Picker 三句式。en/zh 均已同步。
- 合并前补丁备份：`notes/sentence_studies/local_premerge_20261016.patch`。
- git 状态：远端文件已 checkout 进 index（staged），paper_en/zh.tex 为合并版（unstaged）；未提交，等用户指示。
- 轮询频率用户改为 **2h**（cron a6b3ea62，:47 触发，session-only）。

### 第 1 轮（2026-10-16 17:5x，Token-Picker Abstract+Intro）
- 拆解笔记：`notes/sentence_studies/2024_token_picker.md`
- paper_en.tex Abstract 重写（9句→8句）：新增 A1 成本归因句（decode 注意力随历史增长）、A2 "Selection must therefore remain cheap" 需求收口句、A3 "Although query-aware selectors… they score in full width and rank in a single pool" 前作缺口句；方法两句合一；结果句加方向性收口 "a descriptive spread within 0.33 points"；诚实边界 3 句压成 1 句枚举 + 状态句。
- paper_en.tex Intro P1：补 stakes 两句（长上下文负载+注意力成本一阶约束，引 longbench/ruler/quest/sparq）；原第 1-2 句合并改主动。
- paper_zh.tex：摘要与 Intro-P1 直译同步。
- 编译：本环境无 xelatex（build.sh 报 command not found；17:26 的旧 PDF 系远端构建），已验证 4 个引用键 en/zh 均存在；下轮有 TeX 环境时补编译。
- 停止条件：七篇全部拆解到最后一句话、且手稿各段对照修改完成。

## 硬约束备忘
- 图/数字口径一律以 `figs/O_observation_meta.md` 与正文现有数字为准，不新造数字。
- 修改若触碰 observation 相关段，遵守 dac_manuscript 定域替换铁律（只动目标块）。
- GUIDE.md 禁改。
- **基线表不回插**（2026-10-16 并行会话决定：GPT 重写 evaluation 节后锚点 "None is dense FullKV." 已不存在，WIP 备份 /tmp/wip_baseline_table_20261016.patch）。

## 本地编译环境（2026-10-17 升级：真字体版）
- **Python venv**：`/home/wangyuanshuo02/two-level-attention/.venv-paper`（pypdf 6.21 + pymupdf 1.28 + pillow 12.3 + fonttools 4.67）。PDF 逐句学习用它提取文本/图片：`.venv-paper/bin/python -c "import pymupdf; ..."`。
- **TeX Live 2026**：`/home/wangyuanshuo02/texlive/2026`（scheme-basic + xetex/fontspec/unicode-math/lm-math/sttools(flushend)/tex-gyre；tlmgr 仓库=USTC）。
- **真字体包（2026-10-17 远端推送）**：`agent_doc/advice/dac_manuscript/fonts/`（Nimbus Roman/Sans 8 个真 OTF + Noto Serif/Sans CJK SC 含真 Bold + Liberation + licenses + check_fonts.py 校验）。**旧符号链接/ttc 合成兜底已废弃**，本地构建与远端排版一致。
- **编译命令（必须带 TeX PATH）**：`cd source && PATH="/home/wangyuanshuo02/texlive/2026/bin/x86_64-linux:$PATH" bash compile_local.sh [paper_en|paper_zh|supplement_en|supplement_zh]`。compile_local.sh 已改为 portale 版（自带 source ../fonts/env.sh；不再固定 HOME）。注意：命令必须从 source 目录发起（cd source && ...），从仓库根直接跑会因相对路径失败。
- 验证状态（2026-10-17 10:05）：四文档全编译，0 未定义引用；zh PDF 嵌入字体确认真 NotoSerifCJKsc-Bold / NotoSansCJKsc-Bold；en PDF 嵌入真 NimbusRoman Regular/Bold/Italic/BoldItalic + NimbusSans。
