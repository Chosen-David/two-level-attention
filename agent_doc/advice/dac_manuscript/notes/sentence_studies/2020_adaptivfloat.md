# AdaptivFloat (DAC 2020 Best Paper) 逐句精读笔记 — Abstract + Introduction

> 2026-10-17 第 11 轮逐句学习，直接读 PDF 原文（`2020_AdaptivFloat.pdf`，DAC'20，Harvard + Cornell Tech）。
> 量化/数值格式方向的 algorithm-hardware co-design 论文。观察其「机制归因链」与「诚实双数字」写法。

## Abstract（5 句）

| # | 原文（缩写） | 功能 |
|---|---|---|
| A1 | "Conventional hardware-friendly quantization methods, such as fixed-point or integer, tend to perform poorly at very low precision **as their shrunken dynamic ranges cannot adequately capture the wide data distributions** commonly seen in sequence transduction models." | **痛点机制归因句**：性能差不是断言而是 as 从句给出机制（缩小的动态范围罩不住宽分布） |
| A2 | "We present an algorithm-hardware co-design centered around a novel floating-point inspired number format, AdaptivFloat, that **dynamically maximizes and optimally clips** its available dynamic range, **at a layer granularity**, in order to create faithful encodings..." | **方法句**：机制动词对（maximize+clip）+ **粒度限定语**（layer granularity）+ 目的从句 |
| A3 | "AdaptivFloat **consistently** produces higher inference accuracies **compared to block floating-point, uniform, IEEE-like float or posit encodings at low bit precision (≤8-bit)** across a diverse set of state-of-the-art neural networks, exhibiting narrow to wide weight distribution." | **比较主张句**：consistently 副词 + 四个点名基线 + 精度范围限定（≤8-bit）+ 分布范围限定 |
| A4 | "**Notably**, at 4-bit weight precision, **only a 2.1 degradation** in BLEU score is observed on the AdaptivFloat-quantized Transformer network **compared to total accuracy loss** when encoded in the above-mentioned prominent datatypes." | **头牌例外反转句**：我们的最坏情形（掉 2.1 分）对照他们的全灭（total accuracy loss）——同一句内完成悲观/乐观双口径 |
| A5 | "...demonstrate per-operation energy and area that is **0.9× and 1.14×**, respectively, that of an equivalent bit width NVDLA-like integer-based PE." | **诚实双数字句**：能耗更好（0.9×）与面积更差（1.14×）并排报，无遮掩 |

对比：5 句摘要全程带限定语与机制从句。A4 的「例外反转」与 A5 的「好坏双数字并排」是诚实边界的正面示范——我们 Q3 的 3.38/2.46/1.78 比值序列是同款（全坏也并排报）。

## Introduction

### P1（5 句）：目标语境 → 前作聚类 → 评价范围批评 → However+图证据 → 机制归因
句链：目标语境句（"In order to exact higher DNN compute density and energy efficiency at all computing scales, a plethora of reduced precision quantization techniques have been proposed"）→ 前作聚类句带引用（fixed-point [3,5,20] / uniform integer [21,27]）→ **评价范围批评句**（"These fixed-point techniques are frequently evaluated on shallow models or on CNNs exhibiting relatively narrow weight distributions"——前作的证据范围不足以支撑其结论）→ However + 图证据句（"as shown in Figure 1, sequence transduction NLP models contain weights more than an order of magnitude larger than those from ResNet-50"）→ **机制归因句**（"**The reason for this phenomenon is that** batch normalization effectively produces a weight normalization side effect whereas layer normalization adopts invariance properties that do not reparameterize the network"——图上现象当场给出双机制解释）。

### P2（1 句）：灵感定位
"there has been surging interest in floating-point based, logarithmic and posit number systems, which also form the inspiration of this work"——把自己放进流派谱系。

### P3（3 句）：Unlike 对比 + 机制 + However 背离声明
句链：改进声明句（"improves on the aforementioned techniques by dynamically maximizing its available dynamic range at a layer granularity"）→ **Unlike 对比句带失败机制**（"unlike block floating-point (BFP) approaches with shared exponents **that may lead to degraded rendering of smaller magnitude weights**, AdaptivFloat achieves higher inference accuracy by remaining committed to the standard floating-point delineation"——先给对方失败机制再给我们的坚持）→ **However 背离声明句带收益**（"we break from IEEE 754 standard compliance with a unique clamping strategy..., **which enables us to engineer leaner hardware**"——与标准背离是显式声明且绑定收益）。

### P4（1 句）：Rather-than 反范围句
"**Rather than proposing** binary or ternary quantization techniques evaluated on a group of carefully selected models, through AdaptivFloat, we aim to inform a generalized floating-point based mathematical blueprint... that can be easily applied on neural models of various categories"——预防「又是某个专用量化技巧」的误读。

### P5（2 句）：co-design 主张 + Altogether 收口
"By virtue of an algorithm-hardware co-design, we also propose a processing element implementation... to yield energy efficiencies that surpass those of integer-based variants" + "Furthermore, owing to the superior performance at very low word sizes, **as it will be shown**, higher compute density can be acquired at a relatively lower penalty" + "**Altogether**, the AdaptivFloat algorithm-hardware co-design framework offers accuracy and energy advantages"——总收口句。

### P6（1 句 + 贡献 bullet ×3）
"Finally, we note that the AdaptivFloat encoding is **self-supervised as it only relies on unlabeled data distributions** in the network"——**良性性质注记句**（as 从句给理由），然后贡献 bullet（propose/evaluate/propose+show 动词开头，第三条是 hybrid 权衡主张："cost-effective compromise between the high accuracy of floating-point and the greater hardware density of fixed-point"）。

### 对我们手稿的差距结论
| AdaptivFloat 句式 | 我们的状态 |
|---|---|
| A4 例外反转（only 2.1 vs total loss） | 我们无此结构（RULER 22.45 分损失是负向头牌，诚实边界无反转可用），不改 |
| A5 好坏双数字并排 | Q3 比值序列 3.38/2.46/1.78 已同款，达标 |
| P1 评价范围批评句（前作证据范围不足） | Experimental Setting "complete execution configurations are unavailable" 已有对等物，达标 |
| **P1 机制归因句（The reason for this phenomenon is that...）** | **本轮实缺**：Intro P2 报了 13/17 页与 59.65%/26.46% 双口径后直接跳到后果句，漏页-质量不对称（avg 漏页更少却漏质量更多）无机制解释 |
| Unlike 对比句带对方失败机制 | Related Work P1 However 缺口句（第 7 轮）功能等价，达标 |
| Rather-than 反范围句 / 灵感定位句 | 诚实边界下我们不做 generalized blueprint 主张，不加 |
| 良性性质注记句（self-supervised as...） | gate "disabled by default and must be assessed independently" 功能等价，达标 |

### 本轮修改（第 11 轮）
- paper_en.tex Intro P2：59.65%/26.46% 句后加机制归因句 "The asymmetry is mechanical: mean pooling dilutes pages whose mass concentrates on few tokens, whereas the extrema bound upper-bounds each page's best token score and so misses more pages while losing less total mass."（机制均可由 §III-B eq:bound 性质与均值池化算术推出，无新数字）
- paper_zh.tex 直译同步。

---

## §2 Related Work（第 12 轮，2026-10-17）

### 类别一：Adaptive Precision Techniques（~6 句）
句链范式 = **前作动词异质化**：每篇前作用不同动词开头且动词携带其贡献性质——[24] "**noticed** that large magnitude weights bear a higher impact... and **proposed** outlier-aware quantization, which **requires** separate... datapaths"（观察+提议+成本披露三连）；[26] "**observed** that the cell state variation... **can be used to** switch... **yielding** greater energy efficiency"（观察+利用+结果分词）；[22] "**performs**... **in search of** a minimum energy network"；Stripes "**performs** bit-serial computing **to enable** configurable per-layer adaptive precision"；[16] "**aim to find** the suitable per-layer bitwidth"。
类别收尾 = **Rather-than 定位句**："In this work, **rather than** physically changing the bitwidth to adapt to the network precision requirement, we seek to obtain the best numerical representation **out of a chosen bitwidth regimen** by dynamically adapting, per-layer, the floating-point exponent range."——一字一句与全部前作划清界限（他们改宽度、我们在固定宽度内调表示）。

### 类别二：Floating-point Inspired Data Types（~4 句）
句链：**生态归因引入句**（"Aware of the dynamic range limitation of fixed-point encoding, we have seen block floating-point data types, such as Flexpoint... and similar variants employed in the Brainwave NPU"——前作出现的原因前置）→ **appeal 句**（"Block floating-point's appeal **stems from** its potential to achieve floating-point-like dynamic range with hardware cost comparable to fixed-point"——先讲对方吸引力）→ **However 失败机制句**（"by collapsing the exponent value of each tensor element to that of the element with the highest magnitude, elements with smaller magnitudes **will be more prone to data loss**"——机制级缺口：崩塌运算的具体伤害）→ **前作近距离对比句**（"[26] reported... Our quantization process **formulates** a floating-point exponent bias from the maximum absolute value"——与最像的前作一句话点差异）。

### 类别三（尾段）：互补性声明句
"we note that Deep Compression techniques such as pruning and weight sharing **can be used in combination to this work**, which evaluates instead against **mutually-exclusive** numerical data types"——划分比较域（我们比的是互斥数据类型，剪枝等互补技术不在此列）。

### 对照我们 Related Work 的差距结论
| AdaptivFloat §2 句式 | 我们的状态 |
|---|---|
| 前作动词异质化（noticed/observed/performs/aim） | 我们 P1 用统一 "use query-aware approximate selection" 一刀切开头，P2 用 "estimates...schedules"。**可修**：P1 三个选择器可按各自性质差异化动词——Quest 的页极值是**构造保证**（supplies sign-aware page-extremum bounds）、SparQ 的部分比特是**近似**（approximates attention from partial key bits）、HISA 是**分层选择**（prunes attention hierarchically）。但改动涉及三个引用的贡献表述，诚实边界允许（都是公开机制描述），收益中等。**判定：值得做**——同类一刀切正是我们第 7 轮修 However 句时识别的平铺列举病的动词层残留。 |
| appeal 句（先讲对方吸引力再 However） | P2 "Token-Picker estimates probabilities from partial key bits" 直接机制无 appeal；但 Token-Picker 的 appeal（按需调度省访存）与我们保留全维 KV 的对比已在 "instead retains" 句中隐含。结构可加明示："Token-Picker **limits selector reads** by estimating probabilities from partial key bits and scheduling further accesses on demand; our coordinate-based scorer instead retains full-dimensional KV"——先功劳（limits reads）后分歧（instead retains），与我们第 7 轮 P1 结构对齐。**判定：做**。 |
| Rather-than 定位句 | P1 尾句 "We study these existing components with..." 是正面定位；P2 尾句是负向收口。无双类别 Rather-than 需求（我们只有一个类别簇），不加。 |
| 互补性声明句 | "KV compression and placement methods address complementary storage and movement constraints" 已有对等物，达标。 |

### 本轮修改（第 12 轮）
- paper_en.tex Related Work P1：统一动词句拆分差异化——"Quest supplies sign-aware page-extremum bounds for far-page scoring, SparQ approximates attention from partial key bits, and HISA prunes the attention set hierarchically~\cite{quest,sparq,hisa}."（原 "We adopt Quest's sign-aware page extrema for far-page scoring" 信息保留于第一分句）。
- paper_en.tex Related Work P2：Token-Picker 句加 appeal 动词——"Token-Picker limits selector reads by estimating probabilities from partial key bits and scheduling further accesses on demand~\cite{tokenpicker}; our coordinate-based scorer instead retains full-dimensional KV and derives auxiliary scoring views."
- paper_zh.tex 直译同步。

### 远端裁决记录（第 12 轮开始前）
- 远端 1010b06 改写第 11 轮归因句：我的 "The asymmetry is mechanical: ... so misses more pages while losing less total mass" → 远端 "Mean pooling can dilute...; extrema bound... **Neither property determines the observed ordering**"。**采纳远端**：对标 AdaptivFloat P1 机制归因句的适用前提是定理级机制（LayerNorm 不重参数化→权重绝对值大可证）；我们的稀释是启发式级趋势，只能「can dilute」+「不决定次序」，我的 "mechanical" 是过度断言，远端版恰是诚实边界要求的表述。
- Q3 选择器段逐轮差值（0.0237/0.0046 ms）与 gate 段代理/质量区分重写：诚实增强，保留。
