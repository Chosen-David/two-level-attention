# current-main MinMax-far / Avg-near observation

状态：`usable-with-scope`。40 份真实 Q/K、L8/L24、每份 16 个 query，共 1280 个 query cells 与 10240 个 KV-group cells；独立实现和独立进程完成 377147 项断言。范围仅为固定真实 Q/K 的 CPU dense-oracle/L1 选择观察，不是 production-Triton bit-exact 或端到端质量证据。

## 几何与判定

- 对齐 current-main `e87c91e5fd2d76a594e2c46000b4427ea2ecce7b`：sink=4、SWA=128、page=64、K1=128、K2=1024。交付前另核验最新 `2c0c7a783a16883355110e8534d1957eb4107258` 的 selector 差分仅涉及 replay 当前 K/V 绑定与 full/L2 维度张量复用，near/far 几何和 tail32 L1 口径未变。
- near 候选池由 `nfa_near_len=4096` 决定，典型长文档为 64 页；固定选择预算 `nfa_near_pages=32`，不是把 32 页候选全部选中。
- far 选择预算为 96 页；selector 使用 tail32 坐标，保留 Avg/MinMax 页统计及逐头、组均值的 BF16 中间舍入。
- 主 oracle 为每个 KV 组内四个 Q head 的 full128 causal softmax 均值，Top-K=892；shared 口径先将 32 个 Q head 平均，再取统一 Top-K。
- 主指标是区域分层 oracle-token 命中数：far 看 `MinMax-Avg`，near 看 `Avg-MinMax`；严格互补要求两者同时大于 0。自动 bypass token 不计入候选人口。

## 主口径全人口结果

竞争人口要求 far 候选页数大于 96、near 候选页数大于 32，且两个区域都有 oracle token；共 256 个 query cells。

| 切片 | 人口 | far MinMax 严格更优 | near Avg 严格更优 | 两者严格互补 |
|---|---:|---:|---:|---:|
| 全部 | 256 | 99（38.7%） | 244（95.3%） | 93（36.3%） |
| L8 | 128 | 71（55.5%） | 127（99.2%） | 70（54.7%） |
| L24 | 128 | 28（21.9%） | 117（91.4%） | 23（18.0%） |
| HotpotQA L8 | 88 | 59（67.0%） | 87（98.9%） | 58（65.9%） |
| GovReport L8 | 40 | 12（30.0%） | 40（100%） | 12（30.0%） |

全部竞争人口的 far delta 分布为 `>0/=0/<0 = 99/5/152`，near delta 为 `244/0/12`；严格反向双劣势有 6 例。KV-group 级严格互补为 647/2017（32.1%）。

**结论：near Avg 的优势在整体及各主切片中稳定成立；far MinMax 与整体严格互补不是普遍多数现象。L8、尤其 HotpotQA L8 支持设计动机，但 L24 和 GovReport 是明确负例，不能用单个强例或 attention mass 代替全人口结论。**

## Oracle 敏感性：严格互补比例

每格为 `互补数/竞争人口（比例）`。shared 口径在小 K 下部分 query 的某一区域 oracle 为空，因此竞争人口会变化；未删除不利 K 或负例。

| oracle | K | 全部 | L8 | L24 |
|---|---:|---:|---:|---:|
| group | 64 | 95/256（37.1%） | 48/128（37.5%） | 47/128（36.7%） |
| group | 128 | 116/256（45.3%） | 54/128（42.2%） | 62/128（48.4%） |
| group | 256 | 107/256（41.8%） | 54/128（42.2%） | 53/128（41.4%） |
| group | 512 | 96/256（37.5%） | 62/128（48.4%） | 34/128（26.6%） |
| group | 892 | 93/256（36.3%） | 70/128（54.7%） | 23/128（18.0%） |
| shared | 64 | 78/233（33.5%） | 25/105（23.8%） | 53/128（41.4%） |
| shared | 128 | 89/244（36.5%） | 39/116（33.6%） | 50/128（39.1%） |
| shared | 256 | 94/254（37.0%） | 50/126（39.7%） | 44/128（34.4%） |
| shared | 512 | 101/256（39.5%） | 65/128（50.8%） | 36/128（28.1%） |
| shared | 892 | 97/256（37.9%） | 72/128（56.3%） | 25/128（19.5%） |

没有任何 K 或 oracle 聚合口径使整体严格互补超过 50%；因此整体非多数不是由某个偶然 Top-K 或 group/shared 选择造成。正式敏感性结果只保留命中数，临时探索中 group mass 误用 shared plane 的字段已全部剔除。

## 原始代表案例

`hotpotqa_r0006_l8_q12915_oracle.json`：HotpotQA row6、L8、q_pos=12915，far 134 页候选、near 64 页候选，八个 KV 组合计：

| 区域 | oracle tokens | Avg hits | MinMax hits | 优势 |
|---|---:|---:|---:|---:|
| far | 1085 | 692 | 893 | MinMax +201 |
| near | 5621 | 4811 | 4450 | Avg +361 |

far recall：Avg 0.637788、MinMax 0.823041；near recall：Avg 0.855898、MinMax 0.791674。文件保存每组完整 892 个 ranked/sorted oracle token 位置、概率、区域子集、四种方法选中页及 hit/miss，不只是汇总指标。

独立复算曾用 `repeat_interleave` 展开 GQA，最大概率差为 6.71e-8，第 1/2 组出现排名顺序差异，但 Top-892 集合完全一致；保留 KV-group 维度的独立收缩后，所有位置、概率、选页和 hit/miss 均精确吻合。

## 文件

- `summary.json`：完整主口径汇总、逐文档和逐层/数据集切片、输入与源码 SHA。
- `query_population.csv`：1280 个 query cells 的完整主口径人口，保留所有负例。
- `group_population.csv`：10240 个 KV-group cells 的完整主口径人口。
- `oracle_sensitivity_summary.json`：两种 oracle、五档 K 的独立命中数敏感性结果。
- `hotpotqa_r0006_l8_q12915_oracle.json`：代表案例原始 oracle 位置与选择明细。
- `../../validation/current_main_complement_independent_check.json`：377147 项独立断言及交付文件 SHA。
