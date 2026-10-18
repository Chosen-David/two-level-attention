# Entire Introduction: original sentences, bilingual revisions and evidence

This document records the complete bilingual Introduction revision: original sentences, revised sentences, reference functions and editorial reasons. Figure captions and other sections are outside the prose rewrite.

## Verified reference and sentence roles

Actually read and rendered Token-Picker author v1, physical page 1, §1 INTRODUCTION in both columns through its three contribution bullets. Page 2 was checked to distinguish §2 Background & Motivation; §5 evaluation is not the writing reference. Local SHA256 `9a3b476481d6ff5e33b3dd21adf86022f5485f8718df4894bec4ab3280f03486`. [Official DAC award](https://dac.com/61st-dac-best-paper-presentation-recipients) freshly confirms the research award, not just nomination. [Author v1 identity](https://arxiv.org/abs/2407.15131v1) matches title/authors; the actual PDF first-page imprint identifies DOI [10.1145/3649329.3655953](https://doi.org/10.1145/3649329.3655953). Direct DOI web opening returned403; no claim of accessing ACM full text.

Reference sentences below are indexed in reading order and paraphrased by argumentative role, not reproduced or translated. R01–02 are paragraph1; R03–12 paragraph2; R13–15 paragraph3 across the column break; R16–19 paragraph4; R20 contribution lead-in; R21–23 bullets. We borrow the problem→remaining gap→mechanism→contribution chain, not probability estimation, bit chunks, hardware, their numerical gains, or their prose. Exact own originals are in the next section and complete new bilingual sentences in the following section.

| Reference sentence | Role (paraphrase) | Exact own originals indexed below | Revision IDs |
| --- | --- | --- | --- |
| R01 · p1 | 应用场景引出需求 | OEN01/OZH01 | N01–N04 |
| R02 · p1 | 重要性落到推理效率 | OEN05/OZH05 | N03 |
| R03 · p1 | 交代自回归对象 | OEN02/OZH02 | N01 |
| R04 · p1 | 定位体系组件 | OEN02/OZH02 | N01 |
| R05 · p1 | 说明组件职责 | OEN02/OZH02 | N01 |
| R06 · p1 | 串行生成行为 | OEN02/OZH02 | N01 |
| R07 · p1 | 转入资源瓶颈 | OEN03/OZH03 | N02 |
| R08 · p1 | 介绍已有缓解方案 | OEN04/OZH04 | N05 |
| R09 · p1 | 说明已有方案为何有效 | OEN04/OZH04 | N05 |
| R10 · p1 | 指出仍未解决的成本 | OEN06–08/OZH06–08 | N06–N07 |
| R11 · p1 | 说明负载扩大如何放大成本 | OEN15/OZH14 | N11–N12 |
| R12 · p1 | 将问题收束为优化目标 | OEN05/OZH05 | N03–N04 |
| R13 · p1 | 介绍既有稀疏利用方式 | OEN07/OZH07 | N07 |
| R14 · p1 | 指出既有方法未覆盖的变化 | OEN09–12/OZH09–11 | N08–N10 |
| R15 · p1 | 补充适用代价 | OEN16–18/OZH15–17 | N13–N14 |
| R16 · p1 | 引出针对缺口的方法 | OEN19/OZH18 | N15 |
| R17 · p1 | 一句话说明核心机制 | OEN20/OZH18 | N16 |
| R18 · p1 | 展开机制步骤及开销处理 | OEN20–24/OZH18–22 | N16–N19 |
| R19 · p1 | 报告与缺口对应的效果 | OEN27/OZH25 | N21–N22 |
| R20 · p1 | 明确转入贡献 | OEN25/OZH23 | N20 |
| R21 · p1 | 第一贡献：算法与量化结果 | OEN25/OZH23 | N20 |
| R22 · p1 | 第二贡献：执行组织与量化结果 | OEN26/OZH24 | N20–N22 |
| R23 · p1 | 第三贡献：实现与量化结果 | OEN27–28/OZH25–26 | N22–N23 |

R03–06 provide architecture background in the reference; ours compresses it into N01 because the relevant bottleneck is selection over a growing cache. R08–12 describe batching in the reference; ours substitutes the established page-to-token precedent and its remaining selection cost, making no batching results claim. R19/R21–23 use positive performance numbers in the reference; ours must instead retain descriptive quality comparisons, mixed coverage and negative timing. This is a role comparison, not an assertion that the two papers solve the same problem.

## Every original sentence and disposition

The exact original TEX is retained verbatim below. A target ID points to the full EN/ZH revision in the next section. Deletions and merges are explicit; the new evidence sentence N21 comes from the already accepted Q2/S8/conclusion, not a new experiment.

### EN originals

**OEN01 → N01/N04**

Long-context benchmarks such as LongBench and RULER evaluate models on tasks with long input histories~\cite{longbench,ruler}.

**OEN02 → N01**

Long-context decoding repeatedly accesses previously generated keys and values.

**OEN03 → N02**

Sparse attention limits this work by selecting a smaller token set, but selection introduces its own scoring, ranking and memory accesses.

**OEN04 → N05**

A page-to-token cascade first decides which pages may contribute tokens to refinement.

**OEN05 → N03**

The aim is to retain useful candidates within a fixed token budget at a retrieval cost below the attention work saved.

**OEN06 → N06**

Page summaries approximate relevance with different scoring targets.

**OEN07 → N07**

Mean pooling measures an average key response, whereas sign-aware extrema bound individual unquantized token scores in the same coordinates; this bound can be loose.

**OEN08 → N07**

Neither score directly ranks pages by their summed attention weight.

**OEN09 → N08**

Following ClusterKV's token/page perspective~\cite{clusterkv}, Figure~\ref{fig:observation}(a) compares candidate access against a whole-page mass-rank reference on one profiled query: mean and extrema scoring miss 13 and 17 of the 32 mass-rank reference pages, respectively.

**OEN10 → N09**

These missed pages contain 59.65\% and 26.46\% of the reference pages' summed attention weight.

**OEN11 → N07**

Mean pooling can dilute isolated large token scores, while extrema bound each page's largest shared-space token score.

**OEN12 → N09/N10**

Neither property determines the observed ordering of missed-page counts or full-dimensional attention mass.

**OEN13 → N05/N06**

With enough positive-probability candidates to fill the quota, refinement cannot recover unprotected tokens from rejected pages.

**OEN14 → N03/N10**

Summaries must therefore be assessed by candidate access as well as scoring cost.

**OEN15 → N11/N12**

A global TopK budget lets high-scoring candidates from one region occupy slots without guaranteeing minimum capacity for the other.

**OEN16 → N13**

In Figure~\ref{fig:observation}(b), separate near and far reservations raise mean far retention from .6759 to .7207 while lowering it in 43 of 128 correlated observations.

**OEN17 → N13; 正文省略极值，原值保留图源/补充与本映射**

In these observations, the global arm reaches zero far retention, whereas the quota arm's minimum is .2667 (arm minima, not necessarily from the same query).

**OEN18 → N14**

This controllable coverage trade-off calls for explicit page and token capacities: reserving far pages alone need not preserve far tokens.

**OEN19 → N15**

These observations lead to a cascade with two explicit controls.

**OEN20 → N16**

A fixed set of complete low-frequency RoPE pairs supplies the scoring coordinates for both stages; separate near--far page quotas determine regional access to token refinement.

**OEN21 → 图1(c) caption仍完整保留混杂；正文不再靠该诊断论证频率收益**

The coordinate diagnostic in Figure~\ref{fig:observation}(c) motivates the compact view, although its random baseline does not preserve RoPE pairs and so cannot isolate frequency from pair structure.

**OEN22 → N17**

Original-dimensional Q/K/V are retained for final attention, so approximation enters through token selection.

**OEN23 → N19**

We also consider an optional prefill-informed far gate.

**OEN24 → N19；图1(d) caption仍完整保留统计差异**

The historical layer variation in panel (d) motivates this extension, but the diagnostic uses a different single-query statistic; the gate is disabled by default and must be assessed independently.

**OEN25 → N20**

We formulate shared-coordinate regional selection and analyze how page and token budgets jointly constrain retrieval.

**OEN26 → N20**

The evaluation examines task quality (Q1), regional coverage (Q2) and execution cost (Q3).

**OEN27 → N22**

Qwen3 diagnostics show changes in coverage, but historical scores do not isolate either mechanism and sparse decoding is slower than dense attention.

**OEN28 → N23**

The compact 4/128-protection implementation still lacks matched quality--cost evaluation.

### ZH originals

**OZH01 → N01/N04**

LongBench与RULER等长上下文基准在长输入历史的任务上评估模型~\cite{longbench,ruler}。

**OZH02 → N01**

长上下文解码反复访问先前生成的键和值。

**OZH03 → N02**

稀疏注意力通过选择较小词元集合减少这部分工作，但选择本身需要评分、排序与访存。

**OZH04 → N05**

页到词元级联首先决定哪些页能够向精筛提供候选词元。

**OZH05 → N03**

因此，设计需要在固定最终词元预算下保留有用候选，同时使检索成本低于省去的注意力计算。

**OZH06 → N06**

页摘要采用不同的评分目标来近似页面相关性。

**OZH07 → N07**

均值池化刻画平均键响应，符号感知的极值则在相同坐标下给出未量化单词元分数的上界，但该界可能较松。

**OZH08 → N07**

两种评分都不直接等同于按页内注意力权重和排序。

**OZH09 → N08/N09**

沿用ClusterKV的词元／页分析视角~\cite{clusterkv}，图\ref{fig:observation}(a)以整页注意力权重排序为参照，比较同一示例查询的候选访问：均值与极值评分分别漏选32个按页内注意力权重和排序的参考页中的13个和17个，漏选页的注意力权重和分别占全部参考页注意力权重和的59.65\%和26.46\%。

**OZH10 → N07**

均值池化可能稀释孤立的高词元分数，而极值给出每页共享空间最高词元分数的上界。

**OZH11 → N09/N10**

这两个性质均不能决定所观察到的漏页数或全维注意力权重和的次序。

**OZH12 → N05/N06**

在正概率候选足以填满配额时，精筛无法找回被排除页中的未保护词元。

**OZH13 → N03/N10**

因此，页摘要需要同时按候选访问与评分成本评价。

**OZH14 → N11/N12**

全局TopK预算允许一侧的高分候选占据名额，却不为另一侧保证最低容量。

**OZH15 → N13**

图\ref{fig:observation}(b)中，分别预留近远容量使远区平均保留率从.6759升至.7207，同时使128条相关观测中的43条下降。

**OZH16 → N13; 正文省略极值，原值保留图源/补充与本映射**

在这些观测中，全局臂的远区保留率最低为零，配额臂最低为.2667（两臂各自最小值，未必来自同一查询）。

**OZH17 → N14**

这种可控制的覆盖取舍要求明确页级与词元级容量：仅预留远区页并不保证最终保留远区词元。

**OZH18 → N15/N16**

据此，本文在级联中设置两项明确控制：用固定的完整低频RoPE旋转对为两级提供评分坐标，以独立近远页配额控制各区域进入词元精筛的机会。

**OZH19 → 图1(c) caption仍完整保留混杂；正文不再靠该诊断论证频率收益**

图\ref{fig:observation}(c)的坐标诊断为紧凑视图提供动机，但随机基线未保留旋转对，不能将频率作用与配对结构分离。

**OZH20 → N17**

最终注意力仍使用原始全维Q/K/V，近似通过词元选择引入。

**OZH21 → N19**

此外，本文考察基于预填充观测的可选远区门控。

**OZH22 → N19；图1(d) caption仍完整保留统计差异**

(d)中的历史层间差异为该扩展提供动机，但该诊断采用不同的单查询统计；门控默认关闭，需要独立评估。

**OZH23 → N20**

本文给出共享坐标区域筛选的具体形式，分析页与词元预算如何共同约束检索。

**OZH24 → N20**

评估考察任务质量（Q1）、区域覆盖（Q2）与执行成本（Q3）。

**OZH25 → N22**

Qwen3诊断显示覆盖发生变化，但历史得分不能隔离任一机制的作用，稀疏解码也慢于稠密注意力。

**OZH26 → N23**

紧凑4/128保护实现仍缺少匹配的质量--成本评估。

## Full original bilingual revision

N01–04: open with the actual cost tension instead of benchmark cataloguing. N05–10: credit the existing cascade, then explain the selection mismatch with one concrete diagnostic. N11–14: motivate two-stage regional capacities using both the average gain and observed losses. N15–19: state the mechanism and its actual computational limit; optional gating is not a demonstrated benefit. N20–23: name deliverables, summarize already reported mixed evidence, and conclude with the matched measurement requirement.

### N01

EN: Each new token in long-context generation attends to a growing history of cached keys and values.

ZH: 长上下文生成每产生一个新词元，都要对不断增长的缓存键和值进行注意力计算。

### N02

EN: Sparse attention restricts final attention to selected tokens, but selecting them requires scoring, ranking and memory accesses.

ZH: 稀疏注意力将最终计算限制在入选词元上，但选择这些词元仍需要评分、排序与访存。

### N03

EN: A useful selector must therefore preserve task-relevant tokens within a fixed budget while costing less than the attention work it removes.

ZH: 因此，有效的选择器需要在固定预算内保留与任务相关的词元，并使选择成本低于省去的注意力计算成本。

### N04

EN: This tension makes candidate selection, not just the size of the final attention set, a central design problem.

ZH: 这使候选选择过程与最终注意力集合的大小一样，成为设计的核心问题。

### N05

EN: Cascades such as AsyncTLS shortlist pages before selecting eligible tokens~\cite{asynctls}.

ZH: AsyncTLS等级联先筛选页面，再选择符合条件的词元~\cite{asynctls}。

### N06

EN: The coarse stage makes a consequential approximation: a cheap page score need not preserve the tokens that later attention would use.

ZH: 粗筛阶段引入一项关键近似：低成本的页面分数未必能保留后续注意力所需的词元。

### N07

EN: Mean pooling can dilute an isolated high token score, while sign-aware extrema provide a potentially loose upper bound on unquantized token scores in the scoring coordinates; neither ranks pages by their total attention weight.

ZH: 均值池化可能稀释孤立的高词元分数，符号感知极值则给出评分坐标内未量化词元分数的上界，但该界可能较松；二者都不按页内注意力权重总和排序。

### N08

EN: Following ClusterKV's token/page perspective~\cite{clusterkv}, our Qwen3 diagnostic in Figure~\ref{fig:observation}(a) exposes this mismatch: mean and extrema scoring miss 13 and 17 of 32 mass-rank reference pages, respectively.

ZH: 沿用ClusterKV的词元／页分析视角~\cite{clusterkv}，图\ref{fig:observation}(a)的Qwen3诊断展示了这种不一致：均值与极值评分分别漏选32个注意力权重排序参考页中的13个和17个。

### N09

EN: Those missed pages contain 59.65\% and 26.46\% of the reference-page attention mass, so fewer misses need not mean less missed mass.

ZH: 漏选页分别包含参考页注意力权重总和的59.65\%和26.46\%，因此漏页更少并不意味着漏失的权重更少。

### N10

EN: This single-query comparison motivates examining candidate access; it does not rank the selectors by task quality.

ZH: 这一单查询比较促使我们考察候选能否进入后续筛选，但不构成选择器任务质量的排名。

### N11

EN: Candidate access also depends on how capacity is divided between recent and distant history.

ZH: 候选能否进入后续筛选，还取决于近期与远端历史之间如何分配容量。

### N12

EN: A global TopK can devote all available slots to one region, whereas regional quotas can reserve nominal capacity for each.

ZH: 全局TopK可能将全部可用名额分配给一个区域，区域配额则可以为两侧分别预留名义容量。

### N13

EN: In Figure~\ref{fig:observation}(b), separate near and far reservations raise mean far retention from .6759 to .7207, but reduce it in 43 of 128 correlated observations.

ZH: 图\ref{fig:observation}(b)中，分别预留近远容量使远区平均保留率从.6759升至.7207，但128条相关观测中仍有43条下降。

### N14

EN: The resulting design question is how to express this coverage trade-off at both stages: admitting far pages does not itself reserve slots for their tokens.

ZH: 由此产生的设计问题是如何在两级筛选中表达这种覆盖取舍：允许远区页进入候选集，本身并不为页内词元预留最终名额。

### N15

EN: We study a two-level selector that couples a shared scoring view with explicit regional budgets.

ZH: 本文研究一种将共享评分视图与显式区域预算结合的两级选择器。

### N16

EN: A fixed set of complete low-frequency RoPE pairs supplies both stages with the same compact coordinates, while separate near--far page quotas control eligibility for token refinement.

ZH: 固定的完整低频RoPE旋转对为两级提供相同的紧凑坐标，独立近远页配额则控制各区域进入词元精筛的范围。

### N17

EN: Regional token quotas then divide the remaining final budget after prefix and recent-token protection; selected tokens use original-dimensional Q/K/V in one final attention operation.

ZH: 扣除前缀与近期词元保护后，区域词元配额分配剩余最终预算；入选词元使用原始全维Q/K/V完成一次最终注意力计算。

### N18

EN: Compact coordinates narrow scoring dot products, but ordinary token scoring still spans the full history before masking, so a smaller page shortlist does not by itself remove that work.

ZH: 紧凑坐标缩短评分点积的维度，但普通词元评分仍在掩码前遍历完整历史，因此减少候选页本身不会省去这部分工作。

### N19

EN: We also formulate a prefill-informed far gate as an optional extension; it is disabled by default, and its quality and cost require separate evaluation.

ZH: 本文还给出基于预填充观测的可选远区门控扩展；该门控默认关闭，其质量与成本需要单独评估。

### N20

EN: We contribute a shared-coordinate regional selector, an analysis of joint page/token capacity constraints, and separate evaluations of task quality, regional coverage and execution cost.

ZH: 本文的贡献是给出共享坐标区域选择器，分析页／词元容量的联合约束，并分别评估任务质量、区域覆盖和执行成本。

### N21

EN: On complete 11-task RULER32K sets, reported Qwen3-8B AVG is 0.99 points higher for $\beta=.0625$ than $.125$; a separate Qwen3-32B sparse/dense comparison shows a 1.58-point sparse deficit.

ZH: 在完整11任务RULER32K集合上，已报告的Qwen3-8B AVG在$\beta=.0625$时比$.125$高0.99点；另一项Qwen3-32B稀疏／稠密比较中，稀疏臂低1.58点。

### N22

EN: These descriptive comparisons do not isolate either mechanism, and the archived sparse configuration decodes more slowly than dense attention at every measured length.

ZH: 这些描述性比较不能隔离任一机制的作用，归档稀疏配置在所有已测长度上的解码也均慢于稠密注意力。

### N23

EN: Establishing a quality--latency benefit for the compact 4/128-protection design requires matched task-quality and complete-request measurements, including preparation, maintenance and selection.

ZH: 要证明紧凑4/128保护设计具有质量--时延收益，需要匹配的任务质量与完整请求测量，后者包括准备、维护与选择。

## Implementation/evidence binding and limits

| Sentences | Evidence / reason / limit |
| --- | --- |
| N01–04 | Existing Method and Experimental Setting define dense prefill/sparse decode and selection overhead. Frame a requirement, not established speedup, memory dominance on every workload, or a new benchmark result. |
| N05–07 | Related Work already credits AsyncTLS; Method shared-space extrema derivation and `sparse_attn/indexer/tli_indexer.py` establish scoring semantics. Bound applies to unquantized token scores in its scoring coordinates, not full-dimensional probability or quality. |
| N08–10 | Existing `sources/data/remote_observation_input.json` and summary; unchanged Fig1(a) caption records one query,32 reference pages,13/17 misses,59.65/26.46%, different regional allocation/actual counts. Do not attribute observed mass ordering solely to mean/extrema or assert minmax dominance. |
| N11–14 | Unchanged Fig1(b),128 correlated observations, .6759/.7207,43 decreases. Reservations describe nominal capacity; no universal quality guarantee. Page quotas alone do not guarantee far token quota. The dropped arm minima0/.2667 remain historical data and exact originals above. |
| N15–18 | Method/Table1, current indexer coordinate selection, L1 regional topK, L2 full-history scores before mask, protection and merged final support. This is a concrete formulation of existing components, not first-ever cascade/shared-space/GQA invention. Final attention remains restricted-support attention, not exact dense output. |
| N19 | Current gate is optional/defaultoff; unchanged caption and S5/S6 preserve old diagnostic mismatch, single-query logit aggregation and calibration limits. No MLA evidence promoted; no gate effectiveness claim. |
| N20–23 | Q2/S8 and conclusion already report complete11-task32K AVG: 8B59.290909−58.30=.990909; 32B50.642727−52.218182=−1.575455. β changes page and token allocations jointly; comparisons span machines and retain scorer/raw-prediction caveats in caption/S8. Archived E152v2 sparse TPOT slower at all lengths; current compact4/128 has no matched quality–complete-cost result. No noninferiority, no same-budget win over AsyncTLS, no lossless or production speedup. |

## Public scope

This complete sentence mapping is included in the curated public manuscript package. Independent review records, internal workflow records and the third-party reference PDF are excluded. Reference prose is represented by function paraphrases and precise locations; the full original and revised text reproduced here is our own manuscript.

## Sentence-by-sentence editorial iteration

Each original sentence is assessed for its DAC-paper function, specificity, logical placement and evidence scope. Its exact text and complete bilingual revision appear above. A second pass refines N02 in both languages and N07 in Chinese; the other revised sentences are individually rechecked below.

Reference locations count prose sentences within the four Introduction paragraphs, excluding the footnote; paragraph3 crosses the left/right column boundary. The three bullets follow paragraph4 in the right column. This locator convention makes every reference function recoverable on physical page1 without reproducing the reference wording.

### OEN01/OZH01

- DAC editorial diagnosis: 基准名称起笔只回答在哪里测，没有指出DAC读者关心的系统代价；信息正确但论证功能偏离问题入口。
- Comparable reference function/location: R01–02 · p1左栏¶1句1–2；对应功能见R01–23表。
- Actual revision and evidence limit: N01/N04改从逐词元缓存访问与选择成本起笔；LongBench/RULER介绍留在实验设置。
- Neighboring transition check: 从工作负载直接接N02选择开销，避免基准名→系统问题的跳跃。

### OEN02/OZH02

- DAC editorial diagnosis: 反复访问KV是准确背景，但没有把逐步增长的历史与每一步解码联系起来，具体性不足。
- Comparable reference function/location: R06–07 · p1左栏¶2句4–5；对应功能见R01–23表。
- Actual revision and evidence limit: N01明确每个新词元面对增长的缓存历史；不扩张为所有硬件上的带宽瓶颈断言。
- Neighboring transition check: N01给出对象，N02给出减少最终计算的做法与额外成本。

### OEN03/OZH03

- DAC editorial diagnosis: 同时包含节省与选择开销，功能正确；旧句work指代宽泛，应落到最终注意力集合。
- Comparable reference function/location: R07、R10 · p1左栏¶2句5、8；对应功能见R01–23表。
- Actual revision and evidence limit: N02首轮改为selected history后仍有finding history歧义；第二轮实际改为限制最终注意力于入选词元，并明确选择操作。
- Neighboring transition check: 承接N01历史对象，导向N03净收益判据；不暗示删KV存储。

### OEN04/OZH04

- DAC editorial diagnosis: 介绍已有解决路径，却未在首次出现时给出先例，容易把级联误读为本文独有。
- Comparable reference function/location: R08–09 · p1左栏¶2句6–7；对应功能见R01–23表。
- Actual revision and evidence limit: N05前置AsyncTLS引用并压缩两级流程；贡献落在本文具体组合及分析，不占用已有机制的新颖性。
- Neighboring transition check: 由N04设计问题转向已有路径，N06随即说明该路径尚存的近似。

### OEN05/OZH05

- DAC editorial diagnosis: 目标句方向正确，但retain useful candidates不够具体；需要区分设计要求与已测收益。
- Comparable reference function/location: R12 · p1左栏¶2句10；对应功能见R01–23表。
- Actual revision and evidence limit: N03明确任务相关词元、固定预算与选择成本小于省去的注意力成本，用must表达要求。
- Neighboring transition check: N03承接开销，N04将要求收束为候选选择问题。

### OEN06/OZH06

- DAC editorial diagnosis: “不同评分目标”像分类目录，尚未解释为什么差异妨碍系统目标。
- Comparable reference function/location: R13–14 · p1左栏¶3句1至右栏续句2；对应功能见R01–23表。
- Actual revision and evidence limit: N06先说明低成本页评分可能丢掉后续注意力所需词元，再让N07解释具体评分差异。
- Neighboring transition check: 从N05级联走向N06误筛风险，再以N07给出原因。

### OEN07/OZH07

- DAC editorial diagnosis: 平均响应/极值界的对照有技术依据，但句子与后文重复；上界必须保留同坐标未量化限制。
- Comparable reference function/location: R14 · p1右栏¶3续句2；对应功能见R01–23表。
- Actual revision and evidence limit: 合并到N07：平均可能稀释高分，极值界可能松；明确补回未量化对象。第二轮中文实际拆开生硬定语，科学范围不变。
- Neighboring transition check: 接N06近似风险，为N08页权重诊断提供解释背景，而非预言哪臂更好。

### OEN08/OZH08

- DAC editorial diagnosis: 否定两个评分等同页质量有必要，但单独一句打断机制解释。
- Comparable reference function/location: R14 · p1右栏¶3续句2；对应功能见R01–23表。
- Actual revision and evidence limit: 并入N07分号后，明确二者不按总注意力权重排序。
- Neighboring transition check: N07至N08自然引出与mass-rank参照的不一致。

### OEN09/OZH09前半

- DAC editorial diagnosis: 诊断信息充分但句子过长，名词重复；例子应支持缺口而非暗示普遍优劣。
- Comparable reference function/location: R14、R19 · p1右栏¶3句2、¶4句4；对应功能见R01–23表。
- Actual revision and evidence limit: N08保留ClusterKV视角、Qwen3、图a及13/17/32，压缩定位；单查询及对照混杂保留N10/图注。
- Neighboring transition check: 先漏页数N08，再漏失权重N09，随后N10限制用途。

### OEN10/OZH09后半

- DAC editorial diagnosis: 数字正确却缺直接解释：读者可能只依据漏页数给方法排名。
- Comparable reference function/location: R19 · p1右栏¶4句4；对应功能见R01–23表。
- Actual revision and evidence limit: N09保留59.65%/26.46%并明确“漏页少未必漏权重少”。
- Neighboring transition check: N08与N09构成同一例子的两个指标，N10阻止外推任务质量。

### OEN11/OZH10

- DAC editorial diagnosis: 重复OEN07机制，占用引言空间；不是新增证据。
- Comparable reference function/location: R17–18 · p1右栏¶4句2–3；对应功能见R01–23表。
- Actual revision and evidence limit: 并入N07，不重复讲均值稀释和极值界；不删除技术限制。
- Neighboring transition check: 缩短数值例子后回头解释机制的折返，保持机制→例子顺序。

### OEN12/OZH11

- DAC editorial diagnosis: 科学限制正确，但“性质不能决定次序”过于抽象，像审计注释。
- Comparable reference function/location: R14–15 · p1右栏¶3句2–3；对应功能见R01–23表。
- Actual revision and evidence limit: N09改为可直接由数字理解的关系，N10限定任务结论；保留图注中的区域/实际预算混杂。
- Neighboring transition check: 由指标差异转向诊断能回答的候选访问问题。

### OEN13/OZH12

- DAC editorial diagnosis: 正概率候选足够等条件来自实现细节，引言中展开会遮蔽主问题；删条件而保留绝对不可能结论也不正确。
- Comparable reference function/location: R14 · p1右栏¶3句2；对应功能见R01–23表。
- Actual revision and evidence limit: N05/N06保留候选资格与误筛风险，不再作无条件无法恢复断言；具体实现留Method。
- Neighboring transition check: 不插入执行分支，把N10作为本段结论。

### OEN14/OZH13

- DAC editorial diagnosis: 必要评价维度正确，但与第一段成本要求重复；可改为这项诊断带来的具体启示。
- Comparable reference function/location: R12 · p1左栏¶2句10；对应功能见R01–23表。
- Actual revision and evidence limit: N10用单查询候选访问诊断收束；N03负责整体成本目标，避免两处重复。
- Neighboring transition check: N10候选访问直接衔接N11区域容量。

### OEN15/OZH14

- DAC editorial diagnosis: 直接跳到TopK，缺少从页误筛到区域竞争的桥；reserve一词还易被误解为实际覆盖保证。
- Comparable reference function/location: R14 · p1右栏¶3句2；对应功能见R01–23表。
- Actual revision and evidence limit: 新增N11桥句，N12比较全局与区域名义容量；将保证性retain改为can reserve nominal。
- Neighboring transition check: N11说明同一候选访问问题的第二因素，N12说明机制，N13给混合观测。

### OEN16/OZH15

- DAC editorial diagnosis: 这句本身已有均值提升与负向样本，证据平衡；应保留而非为主张删除43例。
- Comparable reference function/location: R19 · p1右栏¶4句4；对应功能见R01–23表。
- Actual revision and evidence limit: N13基本保留准确量值，but明确转折，仍称128条相关观测。
- Neighboring transition check: 接N12容量机制，导向N14覆盖取舍而非普遍提升。

### OEN17/OZH16

- DAC editorial diagnosis: 再加两臂极小值和非配对说明使引言拥挤，且对主机制没有比N13更强的证据。
- Comparable reference function/location: R19 · p1右栏¶4句4；对应功能见R01–23表。
- Actual revision and evidence limit: 正文删去极小值句；原值/非配对限定保留本映射与原数据，平均提升及43例负向仍留N13。
- Neighboring transition check: N13之后直接提出N14两级容量问题，避免指标堆叠。

### OEN18/OZH17

- DAC editorial diagnosis: 准确指出页与词元预算区别，但结论略像规范要求，尚未接到本文要回答的问题。
- Comparable reference function/location: R16 · p1右栏¶4句1；对应功能见R01–23表。
- Actual revision and evidence limit: N14改为设计问题，并以“进远页不预留最终词元位”解释两级约束。
- Neighboring transition check: 从N13混合覆盖到N15本文方案，桥接明确。

### OEN19/OZH18前半

- DAC editorial diagnosis: “两个控制”计数抽象，未命名本文研究对象。
- Comparable reference function/location: R16 · p1右栏¶4句1；对应功能见R01–23表。
- Actual revision and evidence limit: N15命名两级选择器及共享评分/区域预算，避免宣称首次级联。
- Neighboring transition check: 接N14问题，N16–17展开具体实现而非继续抽象总述。

### OEN20/OZH18后半

- DAC editorial diagnosis: 完整RoPE对与区域页配额具体且必要，但还缺最终词元预算如何连接。
- Comparable reference function/location: R17–18 · p1右栏¶4句2–3；对应功能见R01–23表。
- Actual revision and evidence limit: N16保留两项实现，N17补扣保护后的区域词元配额；不把页名额当词元保证。
- Neighboring transition check: N16粗筛坐标/资格→N17最终预算/注意力，符合数据路径。

### OEN21/OZH19

- DAC editorial diagnosis: 随机对照混杂必须披露，但它不是频率优越性证据；放在机制主句中会让动机依赖无判别力对照。
- Comparable reference function/location: R18 · p1右栏¶4句3；对应功能见R01–23表。
- Actual revision and evidence limit: 引言正文不据此声称频率收益；图1(c)原caption与补充保留不配对结构的限制。N18转述代码可确认的计算宽度事实。
- Neighboring transition check: N16讲设计选择，N18讲其实际计算边界，避免以诊断替代机制验证。

### OEN22/OZH20

- DAC editorial diagnosis: 原维Q/K/V边界正确，但“通过选择引入近似”仍需读者自己连到最终输出。
- Comparable reference function/location: R18 · p1右栏¶4句3；对应功能见R01–23表。
- Actual revision and evidence limit: N17明确一次最终注意力使用原维Q/K/V；Method继续说明受限分母不是dense等价。
- Neighboring transition check: 接区域预算，随后N18说明筛选自身仍做哪些计算。

### OEN23/OZH21

- DAC editorial diagnosis: 独立一句引出门控，容易抬高未验证扩展在主贡献中的地位。
- Comparable reference function/location: R18 · p1右栏¶4句3；对应功能见R01–23表。
- Actual revision and evidence limit: 并入N19，明确optional/defaultoff且质量与成本单独评估。
- Neighboring transition check: 置于主路径之后，不与两项核心控制争夺主线。

### OEN24/OZH22

- DAC editorial diagnosis: 历史单查询与当前门控不匹配，不能承担已验证门控动机的强结论；正文细述像状态审计。
- Comparable reference function/location: R15、R19 · p1右栏¶3句3、¶4句4；对应功能见R01–23表。
- Actual revision and evidence limit: N19保留可选与独立验收边界，图1(d)/S5/S6继续完整披露历史统计差异。
- Neighboring transition check: 以简短扩展限定收束机制段，再进N20贡献。

### OEN25/OZH23

- DAC editorial diagnosis: 只说formulate/analyze没有集中呈现读者可获得的成果。
- Comparable reference function/location: R20–22 · p1右栏贡献引导句与bullet1–2；对应功能见R01–23表。
- Actual revision and evidence limit: N20明确selector、联合容量分析、三方面评估三个交付物，不冒充新硬件/新概率估计。
- Neighboring transition check: 先贡献清单，再N21现有质量证据，不从未验收方案直接跳收益。

### OEN26/OZH24

- DAC editorial diagnosis: Q1/Q2/Q3标签适合章节导航，放引言结尾像实验提纲而非贡献。
- Comparable reference function/location: R20–23 · p1右栏贡献引导句与bullet1–3；对应功能见R01–23表。
- Actual revision and evidence limit: 合并N20并直接写任务质量/覆盖/成本；详细编号留实验节。
- Neighboring transition check: 减少标签负担，让N21/N22实际结果回应评价维度。

### OEN27/OZH25

- DAC editorial diagnosis: 负结果诚实但概括过宽，没有让读者看到既有完整任务证据和具体比较对象。
- Comparable reference function/location: R19、R21–23 · p1右栏¶4句4与三bullet；对应功能见R01–23表。
- Actual revision and evidence limit: N21补已经在Q2/S8/结论中的+.99与−1.58，两种对照分开；N22保留机制不可隔离与更慢解码。
- Neighboring transition check: 贡献→有范围数字→负结果，不能用参考文献正收益句式替换自己的反证。

### OEN28/OZH26

- DAC editorial diagnosis: “仍缺少评估”是状态陈述，未说明怎样的系统测量才能支持论文目标。
- Comparable reference function/location: R23 · p1右栏bullet3；对应功能见R01–23表。
- Actual revision and evidence limit: N23改为收益主张所需的匹配质量与完整请求测量；明确补齐准备、维护、选择，不能只测覆盖代理和局部开销。
- Neighboring transition check: 结尾回扣N03净收益目标；不许把未完成条件写成已证结论。

## Actual second-pass edits

### N02

Prior EN: Sparse attention reduces the selected history, but finding that history also requires scoring, ranking and memory accesses.

Final EN: Sparse attention restricts final attention to selected tokens, but selecting them requires scoring, ranking and memory accesses.

Prior ZH: 稀疏注意力缩小参与计算的历史词元集合，但寻找这一集合本身也需要评分、排序与访存。

Final ZH: 稀疏注意力将最终计算限制在入选词元上，但选择这些词元仍需要评分、排序与访存。

### N07

Prior EN: Mean pooling can dilute an isolated high token score, while sign-aware extrema provide a potentially loose upper bound on unquantized token scores in the scoring coordinates; neither ranks pages by their total attention weight.

Final EN: Mean pooling can dilute an isolated high token score, while sign-aware extrema provide a potentially loose upper bound on unquantized token scores in the scoring coordinates; neither ranks pages by their total attention weight.

Prior ZH: 均值池化可能稀释孤立的高词元分数，符号感知极值则在评分坐标内给出未量化词元分数的、可能较松的上界；二者都不按页内注意力权重总和排序。

Final ZH: 均值池化可能稀释孤立的高词元分数，符号感知极值则给出评分坐标内未量化词元分数的上界，但该界可能较松；二者都不按页内注意力权重总和排序。

## Final sentence and whole-section recheck

| Sentence pair | Concrete editorial disposition | Transition |
| --- | --- | --- |
| N01 | 保留：对象是每一步解码和增长KV，未套用参考论文未经本项目测量的memory-bound结论。 | N02从对象转成本 |
| N02 | 实际再改：selected tokens替代history歧义，避免暗示删缓存存储。 | N01→N02→N03对象与代价连续 |
| N03 | 保留：must是设计要求，固定预算与净开销清楚，未假装收益已测。 | N04收束问题 |
| N04 | 保留：把N02/N03凝练成候选选择问题。 | N05以已有级联接问题 |
| N05 | 保留：先例引用明确，不认领已有级联创新。 | N06立即解释剩余近似 |
| N06 | 保留：指出低成本摘要可能丢候选，未声称任何页分数一定失败。 | N07具体化评分目标 |
| N07 | 英文保留、中文再改：上界对象/坐标/未量化限制完整，去掉生硬长定语。 | 机制解释→N08诊断 |
| N08 | 保留：Qwen3及13/17/32例子准确，未推出优劣。 | N09补漏失权重 |
| N09 | 保留：59.65/26.46与漏页数形成具体解释。 | N10限定任务推断 |
| N10 | 保留：单查询限定是必要证据边界，简短而非日志。 | N11继续同一候选访问主题 |
| N11 | 保留：页评分到区域容量的桥句有独立功能。 | N12机制,N13观测 |
| N12 | 保留：can与nominal避免保证覆盖，区别全局与分区。 | N13给真实混合结果 |
| N13 | 保留：平均提升和43下降共同呈现，correlated不删。 | N14从取舍提出问题 |
| N14 | 保留：明确页资格与最终词元容量不是一回事。 | N15回应两级问题 |
| N15 | 保留：study措辞不夸新颖性，研究对象命名具体。 | N16/N17展开 |
| N16 | 保留：完整低频对、同坐标、页配额均与代码对应。 | N17最终容量 |
| N17 | 保留：保护计入总预算，原维QKV边界清楚。 | N18说明开销范围 |
| N18 | 保留：全历史评分事实阻止“页少即计算少”的误读。 | N19可选扩展不承诺提速 |
| N19 | 保留：optional/defaultoff/独立评估三项足够，不重复历史审计细节。 | N20回主贡献 |
| N20 | 保留：三交付物具体，不声称新增硬件或首次级联。 | N21以已有完整集合证据支持 |
| N21 | 保留：+.99与−1.58的比较对象分开，非预算匹配胜出或β因果效果。 | N22立即给非因果和慢时延边界 |
| N22 | 保留：两个负向边界直接相关，不删除更慢结果。 | N23收益条件回扣开篇 |
| N23 | 保留：完整请求含准备维护选择，非局部kernel或代理指标。 | 与N03净收益目标闭合 |

Whole-section check: five paragraphs now form cost problem → existing cascade and page-summary gap → regional capacity gap → actual mechanism and cost boundary → contributions and mixed evidence. The two numerical examples serve different functions: Fig1 diagnoses selection behavior; complete RULER task sets report descriptive quality. They are not joined into a matched quality–latency point.
