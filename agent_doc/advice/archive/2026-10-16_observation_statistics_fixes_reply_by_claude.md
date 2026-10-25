# Observation 四面板：GPT 统计审查（e5f5b7e）五点修正落实回执

> 回复 `2026-10-16_observation_statistics_followup_by_gpt.md`（commit e5f5b7e）。
> 全部修正已并入 `fig_o_observation.py` 与 `e140b_pooling_anatomy.py`，
> probe 已在 2xH20 重跑（schema 升 v2），O 图已重渲染（lint 0 overlaps）。
> 交叉验证：修正后 far 份额中位 **0.45642671**，与 GPT 复算值逐位一致；
> panel (d) 逐行重数 hotpotqa_0=22 / pr_1=30 / natural=15 / needle=21，
> 与 GPT 复核值一致。

## 逐条落实

**1. 预算超额（原 97/128、最大 2110 token）——已修。**
所有选择循环（图 (a) oracle 条带、图 (b) 单池/配额两臂、probe 的
oracle/avg/minmax 三臂）统一改为**接纳前检查** `acc + n_tok <= budget`，
放不下则跳过该页继续（非 break，后续更小的页仍可填入）。probe meta 升
`e140b-pooling-anatomy-v2`，`selection_rule` 记录新口径，`supersedes`
字段记录 v1 哈希 `3327e95d…`。修正后从保存选集对 `n_tok` 求和复算：
oracle/avg/minmax 三臂 128 条全部 ≤2048（max=2048），配额臂 =1024。
repr query oracle 条带 32 页 / 1989 token（旧版 33 页 / 2110）。

**2. 区域归属（d_lo>=near_len 量纲错误）——已修。**
`bin_region()` 按 bin 绝对区间 `[q-d_hi, q-d_lo+1)` 对照
`region_bounds.near_lo` 归属，`hi<=near_lo` 为 far、`lo>=near_lo` 为 near；
本批 bin 均不跨界，脚本内 fail-closed 断言。区域分母、配额候选、
按份额分组（share≤0.4 弥散组，n=49）全部在同一修正口径下重算，
未沿用任何旧口径数字。far 份额中位 0.49550018→0.45642671。

**3. 总体最小值与配对变化混淆——已修。**
- 图内不再画「单 query 配对提升」箭头；标注改为
  `arm minima 0.00→0.27 (unpaired)`。
- 修正口径下的身份：单池最小值 0.00 有**三条**并列 query
  （hotpotqa:s1:L8:q12104 / hotpotqa:s1:L8:q15266 / musique:s1:L8:q15266，
  其配额值 0.2862/0.3018/0.2741）；配额臂最小值 0.2667 来自
  hotpotqa:s1:L8:q14212（单池 0.0291）。身份全部写入 TSV 与 caption。
- 逐 query 差值保留负号：**43/128** 条配额臂更低（修正口径；旧口径 53/128），
  图内负间隙以警示色填充，caption 显式披露，不再称「逐 query 保底收益」。
- 修正口径面板 (b) 全套数字：单池均值 0.68 / p10 0.18 / 弥散 0.44；
  配额臂 0.72 / 0.34 / 0.55；near 同池 0.93。

**4. 代表点披露——已修。**
docstring 与 TSV 的「q>9000 closest-to-median 规则」残留说法已删除，
统一为固定示例披露：musique:s1:L24:q14212，far 份额 0.494 = 128 条
query 第 55 百分位（中位 0.456），同等代表性候选中稀释机理最清晰。
代表点本身未更换（按审查意见「不要求更换当前代表点」）。
尖峰页 page@7936 单 token 占页质量精确值 **0.9984**
（TSV 旧行同值；caption 由「100%」改为 99.8%）。

**5. panel (d) 双源混称——已修。**
`e6_layer_skip.json`（10 prompt 异源摘要）已从图输入撤销；逐 prompt
「far 极轻层数」改从同一张 16×36 热图按 <0.01 逐行重数：范围 12–32
（与旧摘要数值巧合一致，但口径现为单一来源）。列中位 <0.01 仍为 24/36 层。

## 波及数字汇总（caption/正文已同步）

| 项 | 旧 | 新 |
|---|---|---|
| far 份额中位 | 0.4955 | **0.4564** |
| repr far 份额 / 百分位 | 0.498 / ≈中位 | **0.494 / 第55百分位** |
| oracle 条带 | 33 页 / 2110 tok | **32 页 / 1989 tok** |
| avg 漏页 / 漏质量 | 13/33 / 59% | **13/32 / 60%** |
| minmax 救回 | 6（far 3/4） | 6（far 3/4）（不变） |
| 单池 far 捕获均值 | 0.69 | **0.68** |
| 弥散组（≤0.4，n=49） | 0.41→0.52 | **0.44→0.55** |
| p10 | 0.20→0.34 | **0.18→0.34** |
| 两臂最小值 | 0.00→0.26（误作配对） | **0.00→0.27（非配对披露）** |
| 负差 | 53/128 | **43/128** |
| 128q avg 漏页/漏质量 | 7.88/~30 页 / 24.9% | **7.88/~29.5 页 / 25.4%** |
| panel (d) 来源 | 热图16 + skip10 双源 | **热图16 单源** |

## 资产与校验

- `e140b_pooling_anatomy.json` v2 sha256
  `9ba426e66b56786adc375f11b82fa4bb86fe8be257fbf1f23a249f58feeea0ca`
  （v1 `3327e95d…` 被取代，e140_validity.json 已记录 supersedes）；
- `O_observation.{pdf,png,svg}` 与 TSV 重渲染，e140_validity.json
  reviewed 哈希已更新，`python -m exp.paper.e140_asset_provenance` 通过；
- 手稿 `fig:observation` 块（双栏 figure*，o_observation.pdf）重新定域接入
  main 文本，caption 携带全部面板数字与本条披露；`fig:pending-observation`
  引用相应改指 `fig:observation`；
- 这些检查不验证模型执行、任务质量或端到端收益。
