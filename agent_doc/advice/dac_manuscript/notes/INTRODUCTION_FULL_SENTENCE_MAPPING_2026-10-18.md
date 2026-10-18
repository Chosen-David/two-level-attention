# Entire Introduction: original sentences, bilingual revisions and evidence

Baseline main `eea3bc7bcd178769ce6965d8c2491fe4adf93ae9`; advice-only checkpoint `800ddaafa910a241e67a1ab538fedc81dcaac67e`. This is the user-requested entire-Introduction revision, not the scheduled one-paragraph round. Captions and other sections remain outside the prose rewrite.

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

EN: Sparse attention reduces the selected history, but finding that history also requires scoring, ranking and memory accesses.

ZH: 稀疏注意力缩小参与计算的历史词元集合，但寻找这一集合本身也需要评分、排序与访存。

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

ZH: 均值池化可能稀释孤立的高词元分数，符号感知极值则在评分坐标内给出未量化词元分数的、可能较松的上界；二者都不按页内注意力权重总和排序。

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

## Review and publication status

Independent science and reader review, author disposition, final compilation/render checks and immutable delivery identities are recorded separately in the same run scratch evidence. This mapping is not itself a claim that those gates passed; publication receipt binds the accepted version. The Best Paper PDF and this internal study are not members of the public manuscript package.
