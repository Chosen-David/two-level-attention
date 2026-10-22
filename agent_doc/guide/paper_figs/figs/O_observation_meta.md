# O_observation 图元数据（随图同步更新）

> 规则：每次修改 `fig_o_observation.py`、任一数据源或面板叙事后，必须同步更新本文件、
> `O_observation_figure_data.tsv`（由渲染脚本自动重写）、
> `../observation/e140_validity.json`（reviewed 哈希更新），并运行
> `python -m exp.paper.e140_asset_provenance` 校验。

## 渲染

```sh
python3 agent_doc/guide/paper_figs/figs/fig_o_observation.py
# 输出：O_observation.{pdf,png,svg} + O_observation_figure_data.tsv（同目录）
```

版式（2026-10-22 第六轮起）：**1×4 网格**（(a) 条带；(b') 统计柱为**真实网格列**
（原 inset_axes 浮动定位 [1.16,·] 越界压 (c) ytick/标题，字形级门检出 9 处碰撞
后改列）；(c) 坐标三臂；(d) 逐层热图），figsize 10.2×3.9in，
width_ratios [1.85, 0.50, 0.78, 0.76]，wspace 0.56，对 IEEE 双栏
\textwidth≈7.16in 缩放 ≈0.70。用户指示「ab 只留一个，cd 分别另外两个创新点」：
面板与三创新点 1:1（(a) 分区 / (c) 共享坐标 / (d) 门控），区域互补统计为
(a) 右侧 (b') 列（两根堆叠柱 avg/mm × near/far，图例标题 N = 128）。
第六轮同步收敛：标题字号 T 15→13 且 (a) 英文标题收短为
「Avg misses far; minmax near」、(c)「Which 32 dims」（旧标题宽于所在列互叠）；
(a) 色标 shrink 0.72 锚底 + 横排标签置于色标顶右上空带（竖排两侧分别压
(a) 条带边与 (b') ytick 带）；尖峰 ▼ 移入 ylim(0, 4.35) 头部（旧 4.12 越轴顶
压图例带）；(b') ylabel「missed ref mass」退役进 caption；(c) xtick 单行
tail/rand/high（"32" 上移标题）。导出前经 `figure_lint`（**0 overlap**；
3 条 clipped 均为 (a) 色标 LogNorm(2e-4, 2e-2) 显示范围外的 locator tick
10⁻⁵/10⁻¹/10⁰ 位置，bbox 在画布外恰证明不实际渲染，属假阳性；最终 PDF
页外字形复核 = 0）。

**验收门（2026-10-21 第四轮起加严）**：lint 的 artist bbox 抓不住旋转/紧贴
字形碰撞，最终验收 = PyMuPDF **字形级（rawdict char）跨 span bbox 交集**：
>0.5pt² 必须为 0；0.05–0.5pt² 为容忍软带（bbox 相触而墨迹不碰）；页外 span
必须为 0。检测器：`exp/trace/check_glyph_overlap.py`（45% 比例阈值口径）+
pt² 面积口径复核。当前渲染（第六轮，双语）：**硬碰撞 0 / 软带 0 / 页外 0**
（优于第五轮的软带 83；图例行距 labelspacing 0.55→1.2 消除 (b') 图例三行
bbox 相触）。

**2026-10-16 版式约束（用户指示）**：图内不堆说明文字（note box 已全部移除），
面板数字与结论并入手稿 caption。图内仅保留：坐标轴/刻度、面板 (a) 的带名 ytick、
far/near 区域标签与区域底色、紧凑四态图例（蓝=in ref 且选中 / 红///=ref 漏页 /
橙..=ref 外入选 / 灰=未选）与尖峰页 ▼ 标记（page@7936）；(b') 列仅 near/far
图例（标题 N = 128）+ 坐标轴。

**2026-10-16 GPT 统计审查（e5f5b7e）五点修正已并入**（v3 → v4）：

1. 预算：所有选择循环改为接纳前检查 `acc+n_tok<=预算`、放不下跳过（旧版先接纳
   后检查，全局/oracle/avg 至多 97/128 条超额、最大 2110 token；修正后三臂
   max 均 =2048，配额臂 =1024，probe JSON 升 schema v2 并记录 supersedes）；
2. 区域归属：bin 绝对区间 `[q-d_hi, q-d_lo+1)` 对照 `region_bounds.near_lo`
   （旧规则 `d_lo>=near_len` 量纲错误，128 行每行把 65/128 个近区 token 误计
   入 far；far 份额中位 0.49550018→0.45642671，与 GPT 复算逐位一致）；
3. panel (b) 两臂最小值分离披露（单池 0.00 三条并列 query；配额臂 0.27 来自
   hotpotqa:s1:L8:q14212）+ 逐 query 差值保留负号（43/128 为负）；
4. 代表点披露统一为固定示例（撤销 docstring/TSV 残留的 closest-median 规则
   说法——该旧规则实际产出 L8 q13158，与固定点不符）；
5. panel (d) 撤销 e6_layer_skip.json（10 prompt 异源摘要），skip 范围改从同一
   张 16×36 热图逐行计数（交叉验证 hotpotqa_0=22 等与 GPT 复算一致）。

## 数据源（全部为 pin 住的真实采集）

| 面板 | 数据源 | 说明 |
|---|---|---|
| (a)(b)(c) | `exp/trace/results/e140_o_trace_v2.json` | E140 v2 重采（修复采集器：实际 q_norm/k_norm 输出 + 实际 RoPE）；4 源序列 × L8/L24 × 16 query；64-token bin 质量（GQA 组均概率） |
| (a) 叠加 | `exp/trace/results/e140b_pooling_anatomy.json` | E140b pooling 解剖 probe **v2**（2026-10-16 预算修正重跑，本机 H20，transformers 4.56.2 / torch 2.8.0；supersedes v1 sha256 3327e9…）；同口径 import 复用 v2 采集器数学；页边界与 o1 bin 逐一对齐断言 |
| (d) | `exp/trace/results/e6b_far_profiles.json` | E6 逐层 far mass（16 序列 × 36 层，单一来源；e6_layer_skip.json 已撤销） |

O4 v2（`e140_o4_select_trace_v2.json`）有部分末页缺陷，任何面板均不读取
（见 `../observation/E140_VALIDITY.md`）。

## 代表 query（面板 a 条带）

固定为 **musique sample1, layer 24, q_pos=14212**（固定示例，非规则产出，披露如下）：

- far 份额 0.494 = 128 条 query 第 55 百分位（中位 0.456；位置分段有代表性，
  非极端样本）；
- 在同等代表性的候选中，其 avg 稀释机理最清晰（尖峰页 page@7936：单 token
  占页质量 99.8%）；
- 身份与 dac_manuscript 正文/caption 一致；手稿正文须保持同一口径。

区域口径：sink/swa 各 128 保送；near := 未保护 mid 中距 query 最近的 25%（α=.25，
floor 页对齐）；q=14212 时 near_len=3525，far 占 mid token 的 74.7%。

## 面板定义与关键数字（当前渲染，2026-10-21 第四轮）

### (a) Avg pool misses sparse far spikes — 分区 + 异构打分器动机

四条带（x = key 绝对位置，query 在最右端），逐 64-token 页宽矩形，对称四态编码
（蓝=in ref 且该臂选中 / 红///=ref 页该臂漏掉 / 橙..=ref 外入选 / 灰=未选）。
**近/远区域底色带贯穿全高**（近区主色 alpha .12 / sink 区高亮色 alpha .12），
漏页的区域归属（avg 漏在远区红带 / minmax 漏在近区红带）一眼可读——
这是 2026-10-21 用户指示「体现 avg 漏远端、minmax 漏近端」的视觉承载。

1. **attn mass**：逐 bin 的真实注意力质量（log 色标，vmin 2e-4 / vmax 2e-2），非选集；
2. **mass ref**：按真实 bin 质量排序的单池 top-2048 token 参考集（32 页，**不是
   oracle、不是精确 token TopK、不是 DSA**，只是比较基准）；
3. **avg**：共享 32 维 tail 子空间（[48,64)∪[112,128)）token 分的页均值单池 top-2048；
4. **minmax**：同子空间 Quest 符号感知极值单池 top-2048。

关键数字（repr query，预算修正后，对称差集口径）：mass ref 32 页 / 1989 token；
avg 漏 13/32 ref 页（漏质量 59.65%）、ref 外多选 13 页；minmax 漏 17/32 ref 页
（漏质量 26.46%）、多选 17 页——**页数准则与质量准则必须分开陈述**：minmax 漏页
更多但漏质量更少。avg 漏掉的 ref 页中 minmax 救回 6 页（载 ref 质量 49.37%，
其中 far 3/4）；avg∩ref=19 页 vs minmax∩ref=15 页。
尖峰页 page@7936 单 token 占页质量 99.84% → avg 页均分被 63 个邻居稀释而落选，
minmax 上界 ≥ 该 token 分而保留。
128 query 汇总：avg 平均漏质量 25.43% vs minmax 12.08%（页数 avg 平均漏
7.88/~29.5 ref 页）。
→ 结论：near 均匀用 block-avg；far 稀疏尖峰用 minmax 无漏上界 + 精筛。

诊断口径披露：token 分为 32 Q-head 均值（生产按 KV 组内 4 head 逐头选页）；
漏筛同时含 32 维表示误差与 pooling 稀释，二者未在图内分离；
参考集是质量排序单池上界，集合差异不能判定打分器质量优劣。

### (a) inset Missed mass by region — 区域互补性（2026-10-22 起内嵌于 (a)，独立 (b) 面板退役）

两根堆叠柱（block-avg / minmax），每柱 = 128 query 平均漏失参考质量按 near/far
区域分解（近区段主色在下 / 远区段高亮色在上）。区域归属 = bin 绝对区间
[q−d_hi, q−d_lo+1) 对照 region_bounds.near_lo，页↔bin 一一对齐断言 fail-closed。

关键数字（128 query 均值，占各自参考总质量份额）：

- block-avg 漏失：near **0.0843** / far **0.1699**（约 2/3 的漏质量在远区）；
- minmax 漏失：near **0.0787** / far **0.0420**（约 2/3 的漏质量在近区）；
- 近区两打分器打平（0.0843 vs 0.0787，near 区域 mass 均匀 → 页均分不损失，
  block-avg 在近区免费）；远区 minmax 比 avg 好 **4 倍**（0.0420 vs 0.1699）。
- 漏页数（128q 均值）：avg near 3.98 / far 3.89 页；minmax near 5.27 / far 3.77 页
  （总量 avg 7.87 < minmax 9.03，与质量准则排序相反：页数准则偏 avg、质量准则偏 minmax）。

→ 结论：**两打分器的失效区域互补**——avg 漏在远区（稀疏尖峰被页均分稀释），
minmax 漏在近区（均匀 mass 下极值不占优）。设计蕴含 = 分区异构打分
（near block-avg + far minmax），预算分配不在本图判定（见下 hybrid 扫描披露）。

**hybrid 模拟臂披露（不做进图）**：near 预算扫描（128..1536 步长 64，
总预算 2048 不变，near 用 avg / far 用 minmax）显示 hybrid 最优漏失 0.1345
@near=1408，仍高于全局 minmax 0.1208——在本诊断平面上预算拆分无收益。
可复现脚本：`dac_manuscript/sources/build_region_miss_stats.py`（生成
`region_miss_stats.json`，选择复现断言 fail-closed）。预算分配属
**E155d 固定配额网格轴**（nb/nl 绝对配额，任务精度口径），mass ≠ e2e，
本图不做 hybrid 模拟臂（TSV 留 retired 披露行）。

**旧 (b) 面板已退役（2026-10-21）**：原「quota floors far capture」
（配额臂 far 捕获率曲线，0.68→0.72 / p10 0.18→0.34 等）随 nt 轴降格整体
退役——nt 已从「挤兑保护」降格为显式安全配额（E155d 69 配对主效应 null，
+0.04/42 正），挤兑叙事不再是质量论据。旧数字保存在 git 历史
（e021281 版渲染器）与 e140b probe JSON 中，不再入稿。

### (c) Which 32 of 128 dims — 共享子空间动机（未变）

等宽 32/128 维预算：tail32（低频完整对）0.84 / random32（非成对）0.51 /
high32（高频）0.24。主指标 = 选中质量 / 参考 top-K 质量（非全部注意力质量）。

### (d) Far mass by layer — 逐层 far 门控（D′）动机

16 序列 × 36 层 prefill far mass 热图（log 色标，单一来源 e6b_far_profiles.json）：
24/36 层逐样本中位 <0.01；同一热图逐 prompt 计数 12–32 层 <0.01。prefill 统计，
不约束 decode（正文第五节有反例：prefill <.05 → decode .1678）。

## 诚实约束（任何改写不得违反）

1. 4 个源序列；128 条 query 是重复观测，非独立样本。
2. (a) mass ref/avg/minmax 条带是诊断性模拟（mass ref = 质量排序单池参考，非最优
   oracle；avg/minmax = 32 维共享子空间诊断打分器），非生产 near-avg/far-minmax 逐
   KV 头实现；集合差异是候选访问差异，不能判定打分器质量优劣。
3. 质量捕获 ≠ 任务精度；e2e 证据属于评估章节（E155d 网格 + β 让渡消融）。
4. (d) 是 prefill 统计；门控有效性不得由本图断言。
5. (b) 区域互补漏失是**诊断平面的 128q 均值口径**（分 query 的 near/far 参考质量
   归一），非逐 query 配对差异、非任务精度；近区「打平」指均值差 0.0056，非逐
   query 相等。
6. (b) hybrid 扫描结论只覆盖诊断平面（near/far 打分器已定时预算拆分无收益），
   不外推为「E155d nb/nl 轴无收益」——后者是任务精度口径，由网格判决（冠军
   nb32/nl4096）作数。
7. 旧 (b) 配额臂口径（两臂最小值不配对、43/128 负间隙）随面板退役不再入稿；
   引用旧数字须回 git 历史 e021281 渲染器版本核对。

## 口径叙事（2026-10-21 起：E155d nb/nl 绝对配额）

手稿与 caption 叙事已切换到 E155d 固定配额口径（nb 近区页数 / nl 近区 token 长度 /
nt 显式安全配额）。本图 docstring 与 TSV 均按该口径书写；图内不再出现
α/β/γ 比例语言。诊断 probe 采集时的区域切分参数（near := 未保护 mid 的 25%，
页对齐）属数据来源描述，保留在「代表 query」一节，不作设计参数引用。

## 半覆盖律桥接（2026-10-21，E155d 网格判决后新增）

O 图 (a) 的「近区质量集中、远区稀疏尖峰」观察在 E155d 固定配额网格中得到配额侧
对应物：最优格落在对角线 nb·64 = nl/2（近区 L1 页池恰覆盖近区一半）。机制数据
= `exp/trace/results/e155d_half_coverage_anatomy.json`（离线重析 E140b，120/128
query）：近区 64-token 页按质量降序，top-½ 页累计份额 median 0.933 / mean 0.909 /
p10 0.802（均匀基线 0.5）——近区 mass 本身集中在约一半页里，与网格最优一致。

- 该机制面板渲染在 **A6 消融图 (c)**（`figs/fig_a_e155d_grid.py`，
  `A_e155d_quota_grid.*` + TSV panel_id=c），O 图四面板不变、不重渲染。
- 口径边界沿用 E140b：32 Q-head mean z 分数诊断平面，非生产评分器、非任务质量；
  一致性证据非因果证明。
- 叙事链：O(a) mass 异质性 → A6(c) 半覆盖律 mass 集中度 → A6(a) 网格最优对角线
  nb·64=nl/2。观察图 caption 引 A6 即可，勿在 O 图内堆数字（版式铁律）。

## 下游同步清单

- `agent_doc/advice/dac_manuscript/figures/o_observation.pdf`（从本目录拷贝）
- `dac_manuscript/source/paper_{en,zh}.tex` 的 `fig:observation` caption 与引言段
- `dac_manuscript/MANIFEST.json` 哈希
- `dac_manuscript/OBSERVATION_UPDATE_2026-10-16.md`（远端 GPT 交接）
