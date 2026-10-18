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
| 4 | 2021_Gemmini.pdf | **全部完成**（Abstract/Intro/§2/§3/§4/§5+§6） | Intro-P5 Q1/Q2/Q3 前向指针 ✓ Related-Work-P1 定位句 enabling 从句 ✓（后被远端 1c3bcd8 精化为 eligibility/capacity 术语对，裁决采纳） §III-A 末段 trade-off 句 ✓（后被远端 55b9a17 精化主语为 compact scoring，裁决采纳） Setting opener Q 标签 ✓ 结论 β 句平行形 ✓ |
| 5 | 2023_Gamora.pdf | **全部完成**（Abstract/Intro/§2/§3/§4/§5） | 摘要近远覆盖句（标题词注入） ✓ §III-A L1 目的从句恢复 ✓ §III-B 首句主谓修复（t indexes the current token） ✓ Q2 128K 句中心词+比较对象补全（sparse arm / dense FullKV's 46.43） ✓ |
| 6 | 2023_RL_CCD.pdf | Abstract ✓ Intro ✓ §2 ✓ §3 ✓；待：§4 实验 | Intro P4 门控句主语补全（the diagnostic uses） ✓ Intro P4 (c)/(d) 面板引用恢复（远端 784c1a4 裁决回插，压缩版） ✓ §III-A 坐标动机句指代悬垂修复（retained pairs'） ✓ §III-B S 定义移入 S7（define-before-use）+ fourfold 倍数锚 ✓ |
| 7 | 2026_Miter_Aware_LUT_Mapping.pdf | 未开始 | — |

## 修改日志

### 第 29 轮（2026-10-18，RL-CCD §3 框架拆解 + 远端 1ecc63d 裁决）

- **远端 1ecc63d 裁决：整体采纳**（先逐项核验 manifest 事实：三臂 source_data_sha256/ids/answers_sha/lengths 全等、稀疏两臂 scorer+script 哈希一致而与锚不同、锚 legacy-partial+operator-declared YaRN——全部与本地数据一致）。采纳点：图注按长度区分 β 归属（64K 展示 8B 两 β，真实纠错）；S8 64K 段 manifest 级披露深化；「are reported on one machine」归因严谨化；S7 attribution-controls 重写；draw.py 相对字体路径+去重复 y 轴标签。无回插项。
- **RL-CCD §3 全拆解**（4 小节，写入 sentence_studies/2023_rl_ccd.md）：新增范式入库——Given-Goal-such-that 形式化开篇、替代路线承认+经验裁决（We understand another route may also work...empirically observe significantly better）、问题分类句（we choose to formulate as）、借用-定制句（inspired by the renowned...customize by replacing and by simplifying）、收益带规模锚句（millions of instances）、Rationale-first 组件引入（The rationale behind using X is that Y）、公式大白话复述（Basically, Equation aims to）、单调性解释句、性质链收口（Note→That-is→Hence）、Two-fold 理由枚举（otherwise 反例+自适应性）、算法逐行走读（first/then/when/finally+行号锚+repeat-until）。诚实边界排除：drastically/generalizable-to-any/空配置披露不适用。
- **手稿修复 4 处（en+zh）**：① §III-B 主文删 S 定义（主文 0 使用），S=t+1 定义移入 S7 Integer boundaries 使用处——define-before-use 正向修复，主文 -10 词；② §III-A「Their slower positional rotation」→「The retained pairs' slow positional rotation」（指代悬垂：最近先行词是 z_q/z_i 而非 16 个旋转对）；③「far is its complement there」→「far is the rest of $\mathcal M_t$」；④「Compact scoring narrows dot products」→「narrows each dot product fourfold」（RL-CCD 收益带倍数锚；32/128=4× 算术事实非判词）。
- **编译验证**：四文档 6/6/9/9 页，0 未定义引用（log 中 undefined 全为字形替换警告，既有）。

### 第 28 轮（2026-10-18，用户改指令：Intro 逐句 in place 修复 + E147h 64K 图美化 + 推送远端）

- **指令变化**：不写新拆解笔记，直接对照 best paper 范式逐句审计引言并 in place 修复；同步中文；**推送远端**；新数据（E147h RULER 64K）可视化美化。
- **Intro 逐句审计修复（en+zh）**：P2 删 hedging 虚词 "potentially"（"can be loose" 保留方向性）；P5 β 比较平行化（"0.99 points higher for $\beta=.0625$ than for $\beta=.125$"）。
- **E147h RULER 64K 新数据接入图目录**：`inputs/` 增 e147h（β.0625，AVG 48.85）/e146c（β.125，48.05）/e119（FullKV 锚，48.54）三份 JSON，`provenance.json` 补 SHA256 与 commit 51b4679；`draw.py` 全图与紧凑图各加第三面板（64K 三臂：锚白圆、β.125 白菱淡边、β.0625 实心深蓝菱；紧凑图为三横条），`scores.csv` 扩至 48 行，`arithmetic_check.json` 加 8B-64K 断言（Δβ 两臂 +0.80、对锚 +0.31/−0.49，1e-6 容差通过）。
- **zh 图渲染 bug 根因修复**：`ValueError: bytes must be in range(0,256)` —— matplotlib ≥3.11 的 PDF 字形子集 tracker 从 **`ps.fonttype`**（而非 `pdf.fonttype`）取子集宽度；draw.py 只切 pdf.fonttype 致 CJK 码点原样进单字节 chunk。修复：两 rcParams 同步赋值（en 42 / zh 3）。README 已记坑。
- **主文与补充材料 64K 接入（en+zh）**：fig:ruler32k-quality 图注改 32K/64K 并披露"64K 稀疏两臂同机、锚另一次运行"；Q2 段 Against-dense 句改为三口径并列（−0.09 / −1.58 / +0.31，β.125 −0.49，同机披露句内）；Workloads 段证据清单补 64K；S8 补充材料标题改 32K/64K、新增 64K 三臂披露段（MK3 三臂全零、CWE 均 <1——64K 低水平为稠密稀疏共有，非稀疏特有损失）、来源段补 51b4679 修订号。
- **版式零余量第五次触发**：64K 新增 +45 词致 en 溢至 7 页（AI 声明块被挤到第 6 页、references 落第 7 页）；三波压缩回收（图注三句合并、Workloads/实验设置/Q1/Q2/Q3/设计启示/结论逐句收紧共约 40 词）恢复 6 页。诚实信息全部保留（"8B scorers differ; 32B manifests match, predictions unreplayed" 并入图注单句）。
- **编译验证**：四文档 6/6/9/9 页、0 未定义引用、zh 字体全嵌入、48.85 上图（en 第 5 页）。
- **远端 cb0c8f4 Intro 重写裁决**（合并时执行）：P1/P3/P5 采纳（否定先行句式删除、(b) 图口径精化 whole-bins/retained-fraction、β 定义为近区页分配参数+配额同变披露、4/128 展开）；P2 首句 claim-first 重写采纳但拒绝恢复 "potentially"；**P4 面板 (c)/(d) 引用再删——按第 27 轮判例恢复压缩版**。

### 第 27 轮（2026-10-18，远端 Intro 裁决 + RL-CCD §2 Related Works）
- **远端检查与裁决**：fetch 发现 784c1a4（Intro 重写「sanitized introduction mapping」）、b9988a0（qwen 静态掩码测试，不动 tex）、44d09d7（Design implications 措辞精化）。逐点裁决：
  - **784c1a4 Intro 重写：整体采纳，但面板 (c)/(d) 主文引用被整句删除——恢复**。重写质量高（P1 需求链更紧、P2 "so fewer misses need not mean less missed mass" 反直觉收口、P3 设计问句化、P5 新增 We-contribute 枚举——正式采纳 best-paper 贡献列表句式；第 26 轮 the-diagnostic 主语修复随之删除）。但四面板图两面板（(c)(d)）无任何正文引用——图文一致性缺口（裁决点 1）；P4「低频坐标对」主张失去动机诊断支撑链（裁决点 3）。恢复最小引用形：en P4 插入 `The coordinate diagnostic in Figure~4(c) motivates the compact view; its random baseline does not preserve RoPE pairs.` + 门控句改 `...far gate motivated by the layer variation in Figure~4(d); it is disabled by default and requires separate evaluation.`（zh 直译同步）。**版式溢出处置**：初版含完整 caveat +46 词致 en 正文溢至 6 页+references 第 7 页（零余量失败模式第三次）；压缩至 +12 词净增恢复 6 页总页数；caveat 主文版移位至 caption（"random32 does not preserve RoPE pairs"/"It uses ... unlike the current gate" 已在 caption 承载，诚实信息未删）。第 24 轮 §III-B（t indexes）、第 25 轮 Q2（sparse arm）修复核验存活。
  - **44d09d7 Design implications 压缩：采纳**。Thus 收口句删除但 takeaway 在结论完整存活（"Establishing benefit ... requires matched quality and complete-request cost"）；「Likewise, reserving far pages does not guarantee...」→「Far-page quotas admit candidates to refinement without guaranteeing final far-token capacity」为术语精确化。属零余量压缩非信息删除，不回插。
- 拆解笔记（追加至 `notes/sentence_studies/2023_rl_ccd.md`）：§2.A：**well-known 功劳句 + However as-pointed-out 引用缺口**（判词外包给文献）/**chicken-egg 习语锚定缺口句**（问题命名的记忆点+i.e. 形式化+引号命名）/**In-this-paper-since 范围排除定位句**（既然工具已做好 X→不做 X→专做 Y 且 Y 欠研究——三步边界即定位）；§2.B：前作枚举句链带 1400x 量化/**Nonetheless 不比较诚实声明句**（主动声明不比较+理由——防审稿人质疑漏比）；§2.C：小节标题即宣言/**时代让步代价句**（scaling saturates→even at the cost of runtime）/**RL 双优点句**/**superlative 前作枚举**（superhuman floorplans）/**With-the-above-success-stories 收口宣言句**（Related Work 收口即立场）。
- 对照结论：9 项句式均有诚实镜像/判定不加（flaw 判词资格、going-beyond 立场均受诚实边界约束；Nonetheless 不比较声明——Setting 节 nominal-labels 句等价；under-researched 定位——AsyncTLS 重叠禁用）。本轮手稿修改=远端裁决恢复（上文）。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：RL-CCD §3 框架（MDP 形式化+三模块）、§4 实验。

### 第 26 轮（2026-10-18，RL-CCD Abstract + Introduction —— 第 6 篇开始）
- 远端检查：fetch 无新提交（HEAD = origin/main），无裁决需求。
- 拆解笔记：新建 `notes/sentence_studies/2023_rl_ccd.md`（Abstract 4 句表：**well-adopted 信誉句**/**However 直判 flaw 句 + Particularly 收窄**（are flawed 一个判词极 blunt）/**Overcome 登场句**（In-response 家族强势动词变体）/**范围限定证据句**（19 industrial designs in 5-12nm 前置状语+双指标 up-to+基线点名 native implementation）；Intro P1：In-general-aims 平衡句+**However mainly-because 归因引子句带冒号**；Bullet：**一行论点句**（Not all violating endpoints are equal.——6 词承载全文核心洞察）/**better-fixed-more while-others 平行机制句**（引号口语+i.e. 形式化双轨）/However no-intelligence 复述——bullet 自身即微型 P1-P5 弧；P2：Overcome 第二变体（built-upon 挂引用）/**Prior-to 范围句带 rather-than 排除**（一句话说清做/不做/为什么）；P3：**The-goal unleash 一行大目标**/**Note-that 阶段收窄句链**（整流程→placement 两级收窄）/参照工具点名 ICC2；P4：**图差异载体句**（highlights innovations and key differences）/**Given-X-unlike-Y 走查句**/**Note-that 公平比较声明×2**（步骤数恒等+无额外步骤——把公平性写成卖点）；P5：The-outcome-of-our-effort 带 i.e. 括注；贡献 bullets：**问题发现即贡献**（We discover a new PD problem+That-is）/**first-ever+up-to/avg 双口径**（up to 64% (avg. 23%)、up to 66% (avg. 19%)——每指标成对）/可迁移性收益句带 pre-trained+only-few-iterations）。
- 对照结论：14 项句式均有诚实镜像/等价物/判定不加（flaw 判词与 overcome 受诚实边界约束——我们的比较未隔离机制，无资格判前作 flawed 也无已证问题可 overcome；first-ever 受 AsyncTLS already-combines 制约；up-to/avg 双口径——我们的指标本身是 AVG 单口径无分层可报；问题发现即贡献——无贡献 bullet 惯例已判定保持）。**唯一实缺 = Intro P4 门控句主谓失配+指代歧义**（"layer variation ... motivates this extension, **but uses** a different single-query statistic"——uses 无合法主语：variation 不能采用统计，若挂 extension 则语义反转（真义=panel (d) 诊断统计异于当前门控，caption "It uses ... unlike the current gate" 已明示）；zh「但采用」同样悬空；第 24 轮同类病灶再现）。
- paper_en.tex Intro P4：→ `...motivates this extension, but the diagnostic uses a different single-query statistic;`（+2 词补主语）。
- paper_zh.tex 直译同步：「但该诊断采用不同的单查询统计」。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：RL-CCD §2 Related Works（三小节 credit-then-focus 已在原文可见）、§3 框架、§4 实验。

### 第 25 轮（2026-10-18，Gamora §4 Experiment + §5 Conclusion —— 第 5 篇拆解完毕）
- 远端检查：fetch 无新提交（HEAD = origin/main cbbfbe9 已合并），无裁决需求。
- 拆解笔记（追加至 `notes/sentence_studies/2023_gamora.md`）：§4.A：编号双库枚举句/**双模型分述句**（shallow/deep 按设计复杂度分化）/**Setup 收尾泛化范围句**（trained on small bitwidth, typically less than 32-bit / evaluated up to 2048-bit——训练-评估不对称范围声明）；§4.B 开头：**编号三方面评估路线图句**（how 从句版）；1)：**First/Second/Third 三发现枚举**（the-larger-the-higher 单调句带收敛点+The-main-reason-is 充分性归因 / conspicuously-outperforms 带 indicating 归因 / there-is-always-a-boost 带 since 归因）/**实验发现→前作机制类比句**（synergy is analogous to structural hashing+functional propagation——corresponding-to 的评估节回声）/头牌句/**诚实失败句带可纠正性**（cannot be extracted, but can be easily corrected during post-processing）；2)：**From-X-aspect 双视角句**（model selection aspect / training aspect）/adequate-variety 训练多样性归因句；3)：**known-challenge 引用句+Thus 评估动机**/**库规模量化句**（161 gates including multi-output cells, which significantly increases）/**好情形双数字句**（over 99% and 92%）/**The-scenario-is-fairly-different 反转句+first/second/third 三点恶化枚举**/**takeaways 编号收口**（the-more-the-more 带 since / **comes-at-the-expense-of 代价句**/ERM+Vapnik 理论引用支撑）；§4.C：**In-addition-to-quality 递进登场**/粗体 run-in 标签/**规模量化锚定句**（34M nodes/67M edges）/头牌速度句/This-shows-not-only-but-also 收口/**当前限制披露带 future work**/Even-with 让步收口；§5：**编号三贡献复述句**（(1)(2)(3) 与摘要 A5 同构——质量/速度/泛化三贡献各一句数字全带）。
- 对照结论：16 项句式均有诚实镜像/等价物/判定不加（三发现枚举——Q2 是异质记录非平行轴；失败可纠正性——22.45 分损失无补救主张禁；In-addition-to 递进——无已确立质量优势可递进；编号贡献复述——既有判定诚实镜像）。**唯一实缺 = Q2 128K 句裸形容词+比较对象缺失**（"Historical 128K sparse scores 23.98 versus 46.43"——sparse 悬空缺中心词 arm，46.43 参照未点名稠密 FullKV；zh 同病）。
- paper_en.tex Q2：→ `The historical 128K sparse arm scores 23.98 against dense FullKV's 46.43.`（+4 词，远端压缩后版式有余量）。
- paper_zh.tex 直译同步：「历史128K稀疏臂得分23.98，对照稠密FullKV的46.43」。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- **Gamora 五节全部拆解完毕**（第 22-25 轮），累计手稿修改 3 处。下轮开始第 6 篇 **2023_RL_CCD.pdf**（Abstract+Intro）。

### 第 24 轮（2026-10-18，Gamora §3 Proposed Approach）
- **远端检查与裁决**：fetch 发现 cbbfbe9（Q2 重写 + 新增 fig:ruler32k-quality 双语全宽图 + \dbltextfloatsep 版式调整）。四点裁决**采纳**：(1) 图文一致——旧 fig:fullkv-quality 引用零残留、新图资产齐全（ruler32k_avg_compact_en/zh.pdf）、S8 补充图承接逐任务分数；(2) 量化口径完整——58.30→59.29、58.96、−0.09（评分器不同）、−1.58（清单匹配）、23.98/46.43、26.09/47.91 全保留，32B 逐任务数字移入图+S8（figure-promotion 压缩，非删除）；(3) 收口句支撑链完整（"establish neither allocation causality nor non-inferiority" 由前句披露的 different scorers/cross-machine/unreplayed 支撑）；(4) 诚实边界全存活（"not task-quality evidence"、k_f=0 推论、protection-only 混合披露）。en 版 "change jointly" 弱于 zh「归因存在混杂」，由收口句覆盖，不回插。另：本地第 23 轮提交被并行会话 rebase 推送为 6f5d1a5（内容核验存活），merge 无冲突。
- 拆解笔记（追加至 `notes/sentence_studies/2023_gamora.md`）：§3.A：**粗体段落标签句**（Problem Formulation./Case Study on Multipliers.）/**输入定义句带 without 排除声明**/**目标句带 by-机制与 after-which 下游后果**/**难度句带 especially-子类聚焦**/**前作 SOTA 点名句带量化上限与代价**（up to 2048-bit but with extremely long runtime——credit-then-but 正文版）/**Thus 收口登场句带双收益 which 从句**；§3.B 开头：**机制对应声明句**（需求→GNN 可模仿同位语从句）/**两步程序引用句**（编号流程挂引用）/**Therefore 提案句带本质贴合论证**（to approach its nature）/**编号路线图句**（(1)(2)(3) 与小节 1:1）；1)：**not-only-but-also 双收益句**/图例走查句链（节点特征向量实例枚举）/**模型选择句带强调理由回收**（With the emphasis on→GraphSAGE，回收 Intro Unlike 伏笔）；2)：**标注规则句带 Since 归因**/**MTL 双收益论证句**（not-only-but-also + since 单任务难度归因）/**超参具体值披露句**（α=0.8, β=γ=1）；3)：**完整算法图内走查长链**/**诚实失败案例句（方法节内）**（Figure 3(e) 一个 HA 无法提取 due to misprediction）/**失败定界句三级止损**（only several+due to+subtle impact）/**期望收口句带诚实动词**（is expected to reach as close as possible to）。
- 对照结论：16 项句式中 15 项有诚实镜像/等价物/判定不加（Thus 双收益句受诚实边界约束禁挂未证收益；编号路线图句与 §III 开头既有分句预览重复且零版式余量；方法节失败案例句需新测量支撑不可用观察图单查询数字泛化）。**唯一实缺 = §III-B 首句主谓失配**（"zero-based position $t$ includes prefill"——位置不能「包含」预填充，计数范围才能；zh 括注「在完整序列中从0计数的位置（包含预填充词元）」清晰而 en 语义悬空）。
- paper_en.tex §III-B 首句：→ `In decoding, $t$ indexes the current token from zero across the full sequence including prefill, and $S=t+1$ counts valid cached positions, excluding padding.`（主谓修复，主动动词 indexes；+4 词；zh 已清晰无需改）。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：Gamora §4 Experiment（约第 25-26 轮拆完）。

### 第 23 轮（2026-10-18，Gamora §2 Preliminary and Motivation）
- 远端检查：仅 4d51efc / abb5a12（构建产物对账，不动 tex），无内容裁决需求。
- 拆解笔记（追加至 `notes/sentence_studies/2023_gamora.md`）：§2.A：**成熟度声明句带多域枚举**（BNs well-studied+四域）/表示选择理由句带 since（AIG 统一数据结构=五任务共享同一表示）/定义句组/**转换保证句带性质副词**（fast and scalable）/cut 定义段图例走查句链（Thus 收口+By-pairing 提取机制句：机制→提取→聚合三层目的链）；§2.B：**收益+机制+普及三合一开场句**（分词双机制+引用簇）/**However 三连缺口句**（三方法逐个缺口各带机制归因——Intro 编号局限的正文展开版）/**Motivated-by 小节收口句**（双缺口→提案）；§2.C：**Since-自然适配句四任务版**/**Promising-as-they-are 让步缺口句**（ML 前作也双缺口：规模上限+泛化方向反——为 Unlike 句铺垫）/**双 corresponding/analogous 机制映射句**（把 §2.A/§2.B 铺垫机制逐点映射到 GNN——Intro corresponding-to 的正文完整版）/**Motivated-by-the-analogy 登场句**（与 §2.B 末句同构双收口）。
- 对照结论：表示选择理由句（§III-A 慢旋转动机句=动机+诚实限定版等价）、转换保证句（eq:bound 定理级保证等价）、Promising-as-they-are（领域内无学习型稀疏注意力前作可诚实枚举，不适用）、双 corresponding/analogous（重审维持第 22 轮不加判定——类比仍暗示消息传递级等价，需实测）、However 三连缺口（credit-then-gap 已建）、Motivated-by 小节收口 ×2（§III 开头段已有需求承接句+gate 小节承接句，方法直述结构等价判定不加）、By-pairing 组合结构识别句（§III 开头 "two explicit mechanisms" 已分句覆盖两机制角色分工）均不改。**唯一实缺 = en §III-A L1 句目的从句被远端 55b9a17 精化删去**（"for coarse ranking" 丢失；zh 版「用于粗粒度排序」仍在——L1/L2 定义级目的平行性 en/zh 不一致）。
- paper_en.tex §III-A：`L1 aggregates selected key coordinates into page summaries; L2 refines candidates...` → `...into page summaries for coarse ranking; L2 refines...`（恢复定义级目的从句，+3 词；zh 无需改）。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：Gamora §3 Proposed Approach、§4 实验。

### 第 22 轮（2026-10-18，Gamora Abstract + Introduction —— 第 5 篇开始）
- 远端检查：仅 9690ba7（构建产物对账，不动 tex），无内容裁决需求。
- 拆解笔记：新建 `notes/sentence_studies/2023_gamora.md`（Abstract 6 句表：stakes 多域枚举句/**前作 suffering-from 双缺口句**/**In-response 登场句带三能力枚举**（与缺口逐点对应）/**压力测试预告句带 since 归因**（主动评估已知更难场景——technology mapping known to make reasoning much more challenging）/**编号双口径证据句**（(1) 质量+速度对+点名基线 ABC；(2) 抗压质量+分析承诺）；Intro P1：时代定位句带冒号内嵌行业量化（40%+引用）；P2：**三层嵌套归因句**（工具优化→信息丢失→结构复杂化）+goes-further 递进句；P3：**Despite 功劳让步句+编号局限枚举**（三点各带机制归因，第 3 点是算法类共性缺口）；P4：**Since-自然适配句**（领域对象天然是图→GNN 天然适用）；P5：**双回收登场句**（Motivated by 局限+潜力两源）；贡献 bullet：**粗体短语标题式**+**corresponding-to 新旧机制逐点对应句**+Unlike-反向泛化句+双 Regarding 分口径）。
- 对照结论：压力测试预告（慢解码+128K 失败案例在摘要/Q2——诚实镜像）、编号双口径（三实测数字）、三层归因（Intro P1 机制链）、Despite+编号局限（credit-then-gap 已建）、Since 适配（慢旋转动机句）、双回收（Intro P4 句）、粗体 bullet vs 问句标题（既有判定保持）、量化背景句（无可引来源禁造）、Unlike 反向泛化（无诚实对应物）均不改。corresponding-to 新旧机制对应句：判定**诚实边界下不加**（对应关系会暗示等价性/优越性主张，AsyncTLS already-combines 句已覆盖组合先例）。**唯一实缺 = 摘要第 3 句缺标题词 near--far**（"Regional coverage" 未与题名 Near--Far Sparse Attention 对齐——摘要前两句均无区域属性词，标题关键词在摘要缺席）。
- paper_en.tex 摘要第 3 句：`Regional coverage involves trade-offs` → `Near--far coverage involves trade-offs`（标题词注入摘要，词数不变）。
- paper_zh.tex 直译同步（「近远覆盖存在取舍」）。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：Gamora §2 Preliminary and Motivation（AIG 基础+方法图解）、§3 方法、§4 实验。

### 第 21 轮（2026-10-18，Gemmini §5 Case Studies + §6 Conclusion —— 第 4 篇拆解完毕）
- **远端检查与处置**：fetch 发现 7d33672（LongBench 臂身份绑定：available→historical five-task LongBench shared-subspace arm is linked to——与 Q1 节 "the shared-subspace run uses the earlier 128/128-protection implementation" 证据分层一致，无需裁决、保留）与 d9de7d7（advice 实验 E143a，不动 tex）。**重定基处置**：本地第 20 轮提交 0323e0b 被并行会话 rebase 重写为 4b4d714（内容相同：Setting opener Q 标签；基底更新为含 7d33672 的远端链）——处置=接受重写版，0323e0b 弃用不作处置。账本核对：第 20 轮登记内容在 4b4d714 中完整存活。
- 拆解笔记：§5.A：**前作分歧双极句**（from NeuMMU 128 PTWs to Cong et al. 两级 TLB——两个极端各挂引用）/**This-lack-of-convergence 缺口句**（分歧本身成为平台动机）/迭代句含零配置极端设计点/**不对称双数字句**（4→16 提升 11% vs even 512 entries never >8%，just/even/never 三限定词）/This-is-because 归因+In fact 二次量化（87%/83%）/**低垂果实句**（a simple optimization: a single register）/极限结果句（0 cycles）/**诚实余量句**（only 2% less than maximum——near-max 报差距而非报达成）/边际递减句；§5.B：**决策命名句**（design decision: memory partitioning based on application characteristics）/**三级递减枚举**（high/less/almost none）/must-balance 责任句/**预算决策句**（问题即分配决策）/**负收益也报带 due-to**（residual 1-4% slowdown due to cache thrashing）/**双核反转句**（单核 BigSP 最优→双核 BigL2 反转）/**more-than-negated 净额句**/跨核干扰机制句/双指标收口（-7.1% miss rate +8.0% vs +4.2%）；§6：登场句四度同构+三贡献各一句。
- 对照结论：登场句四度同构（结论首句等价）、负收益全报（我们归因更保守——"cannot identify the cause"）、不对称双数字（β 比较镜像）、净额句（混杂披露=不可归因版的净额）、案例总结（识别要求镜像）、lack-of-convergence 缺口（需文献计量支撑，诚实边界禁）均不改或不可用。**唯一实缺（轻）= en 结论 β 句比较短语非平行形**（"higher at $\beta=.0625$ than .125" 尾端缺 at 与 math 模式；zh 已是完整形）。
- paper_en.tex 结论 β 句：→ `0.99 points higher at $\beta=.0625$ than at $\beta=.125$`（+2 词；zh 无需改）。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- **Gemmini 六节全部拆解完毕**（第 17-21 轮），累计手稿修改 5 处（Q 指针×2、enabling 从句、trade-off 句、β 平行形），en/zh 同步，其中 2 处被远端后续精化并裁决采纳。下轮开始第 5 篇 **2023_Gamora.pdf**（Abstract+Intro）。

### 第 20 轮（2026-10-18，Gemmini §4 Evaluation）
- **远端裁决**（本地 = origin/main，55b9a17 已由并行会话合并）：远端把第 19 轮我的权衡句 "The views narrow each dot product but add work" 再精化为 "Compact scoring narrows dot products, but view preparation and maintenance add work"——**采纳**：收益主语限定为 compact scoring（历史 width-128 L2 张量也是视图但不缩小点积宽度），"The views" 主语过宽；远端版与 §III-C 及全文 compact 术语一致。判例：收益归属主语必须精确到实现变体层级。
- 拆解笔记：§4 部分（开篇纯路线图句点名双基线；§4.A：平台披露句（FireSim）/负载枚举/环境完整句回扣 §3.C/**分解定位句带 only/a-higher 方向词**（67.1%/11.3%/16.6% 三连）；§4.B：**目的内嵌 setup 句**（to determine how X interact to impact Y）/**臂定义句带功能从句**（relieving the host CPU）/**As-illustrated 条件机制链**（配置→依赖责任转移）/**The-less-the-more 单调方向句带 giving 后果分词**/**However 反转句带 because 归因**/**设计原则句带 while-preserving**/工具价值收口/头牌双基线句（2670×/1130×）/**诚实失败句带机制+Therefore+only**（MobileNet 127× due to 低数据复用）/**对抗案例句**（SqueezeNet 专为 CPU 设计仍 1760×）/**匹配条件可比句带图锚**（same number of PEs as Figure 6a）/BERT 覆盖句）。
- 对照结论：问句标题+Intro Q 指针共同承担目的内嵌（问句形式写进 opener 会与 Intro P5 重复且 +13 词超余量）；分解定位（TPOT 比值方向句）、单调方向（β 定理级句）、However 反转、设计原则（诚实边界决定用 requirement 而非 principle）、诚实失败（我们的"cannot locate the cause"边界更强——MobileNet 的 due-to 有实测支撑我们没有）、对抗案例（双臂零分句）、匹配条件（报不可比性镜像）均不改。**唯一实缺（弱）= Setting opener 无 Q 标签**（评估节入口缺编号导航；Gemmini §3 登场句三度同构证明 deliberate reprise 合法）。
- paper_en.tex Experimental Setting 首句：`...task quality, regional candidate coverage and execution cost.` → 三方面各挂 Q 标签（+3 词）。
- paper_zh.tex 直译同步（「任务质量（Q1）、区域候选覆盖（Q2）与执行成本（Q3）」）。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：Gemmini §5 Case Studies（两案例：TLB 设计空间探索/filter registers 低成本优化句式）——下轮收尾本篇。

### 第 19 轮（2026-10-18，Gemmini §3 Generator）
- **远端裁决**（本地 = origin/main，1c3bcd8 已由并行会话合并）：远端把第 18 轮我的 enabling 从句 "which keep page and token access separate" 精化为 "distinguishing page-level eligibility from final token capacity"——**采纳**：与全文既有抽象链逐点对齐（图 2 caption "Page masks limit eligibility"、摘要 "far-page reservation need not preserve far-token capacity"、结论 "far-page reservation from far-token coverage"），我的版本更模糊且可误读为配额分离了访问本身。判例：关键抽象传播须用全文既有术语对。
- 拆解笔记：§3 部分（开篇：两端跨度句（from-edge-to-cloud）+用户能力句（interact to affect）；§3.A：**We-design-目的句**（设计决策+目的+图证据三合一）+中心单元定位句+层级走查句链（compositional 双层递归）+**双数字权衡句带双 due-to**（2.7× 频率 vs 1.8× 面积/3.0× 功耗，各带机理归因）；§3.B：**not-just-also 双层贡献句**+At-high/Alternatively-low 结构对称句对+**Therefore 派生需求句**（需求→自动化供给）+If-the-programmer-wishes 可选控制句+**As-we-can-see 观察句带数量级对比**+前向引用带数字预告；§3.C：range 两端句+**enables-so-that 能力链**+环境价值句（baremetal would not bring to surface）+**亲历 bug 双例句**——DREAMPlace 数值陷阱句的系统级版本）。
- 对照结论：We-design-目的句（§III 开头目的先行句等价）、层级走查（GQA 小节 L1→L2 走查链）、可选控制句（gate OFF 路径+S7）、观察句带数字（漏页三口径）、前向引用（Q 指针+S 网）、亲历证据（k_f=0 推论+代理句——我们边界更强）均不改；双数字权衡句诚实边界禁造。**唯一实缺 = §III-A 末段纯代价披露**（"adds work without compressing"——无 benefit 层；§III-C 定义级收益句 "Width-32 scoring narrows each dot product" 未传播到评分视图段）。
- paper_en.tex §III-A 末段：改为 `The views narrow each dot product but add work, without compressing the original KV cache (S7).`（benefit-but-cost 权衡句式，前半句＝§III-C 定义级事实传播）。
- paper_zh.tex 直译同步（「评分视图缩小每次点积，但辅助视图的准备与维护增加了执行工作，并未压缩原始KV缓存。」）。
- **版式教训复现**：初版 +7 词使 en 6→7 页（零余量第二次触发），压缩至 +2 词后恢复 6 页。
- 编译验证：en 6 / zh 6 页，0 未定义引用。
- 待后续轮次：Gemmini §4 评估（area breakdown 占比句/im2col-BOOM 交互实验）、§5 案例研究。

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
