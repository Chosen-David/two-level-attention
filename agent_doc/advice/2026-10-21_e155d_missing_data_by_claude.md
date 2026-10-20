# E155d 冠军配额叙事的缺失数据清单（投稿前后优先级排序）

> 作者：Claude（2026-10-21）。触发：E155d 固定配额网格终版判决（87/87 格零失败，
> 冠军 nb32/nt448/nl4096 AVG5 48.668）+ 半覆盖律 mass 机制验证（top-½ 页 median
> 0.933）后，手稿是否切到两轴故事（nb+nl 主轴、nt 安全降格）取决于下列数据缺口。
> 本文只列缺口与判据，不启动作业。

## 已闭环的部分（不缺）

- 网格判决本身：`exp/trace/results/e155d_grid_verdict.json`（排序/配对 CI/CTRL 单池
  对照/69 对 nt 配对，全部程序化读取）。
- 半覆盖律机制：`exp/trace/results/e155d_half_coverage_anatomy.json`（120/128 query，
  离线重析 E140b；分析器 `paper_figs/trace/e155d_half_coverage_anatomy.py`）。
- 可视化：A6 五面板图 `figs/A_e155d_quota_grid.*` + figure_data.tsv（lint 0/0）；
  O→A6 桥接已记 `figs/O_observation_meta.md`。

## 缺口 1：nb 主张的配对 CI（投稿后第一优先）

**问题**：两轴故事的唯一正向数字是冠军 vs CTRL 单池 +1.21（AVG5），以及 nb4→8
+0.97 最大台阶；两者都只有点估计。top-3 配对 CI 全含 0（同族格噪声带内等价），
意味着「nb 是最强轴」目前是边际均值口径的排序陈述，不是显著性陈述。

**需要**：
1. 冠军 (32,448,4096) vs CTRL 单池的文档簇 bootstrap 配对 CI（同批、预算对齐
   1021.5 vs 1024.0，配对材料已在 verdict JSON 内，缺的是簇级重抽样协议）；
2. 冠军 vs 当前手稿紧凑族 (8,320)（in_place_top1=false 已披露）：决定手稿是否
   需要换冠军配置数字——**这是 T6 冻结的根因**；
3. nb 边缘配对 CI（advice 2026-10-21 已定口径：正式配对 CI 属补充实验）。

**判据**：若 1 的 CI 下界 > 0，两轴故事获得显著性支撑；若含 0，手稿维持
「descriptive grid + mechanism」口径，冠军句式降为「within noise band」。

## 缺口 2：半覆盖律的长度外推（RULER 侧）

**问题**：半覆盖律 mass 证据来自 LongBench 4 源序列（E140b 128 query，最长约
32K 诊断上限）。网格最优 nb·64=nl/2 在 RULER 64K/128K 是否仍成立未测——若
长序列近区 mass 更分散（top-½ 份额下降），对角线配额将系统性偏小。

**需要**：RULER 64K 代表序列的 near 页 mass 累计份额曲线（同一
`e155d_half_coverage_anatomy.py` 协议，换 trace 源即可，GPU 半小时级）。
与 T4（nt 轴 RULER 64K 七臂判决，0809d3f 预登记）可同批采。

**判据**：top-½ 份额 p10 仍 ≥0.7 → 对角线设计跨长度成立；若明显下降 →
nl 轴在长上下文需重调，手稿 half-coverage 句加长度边界披露。

## 缺口 3：nt 轴最终判决（T4，已预登记）

LongBench 先验 nt448 vs ntoff +0.405 [+0.099,+0.786] 排除 0（显著性集中在
nt448 甜点臂）；69 对格内口径均值 +0.04 为 null。两者不矛盾（格内最优臂
≠ ntoff 的逐格配对）。nt 去留等 RULER 64K 七臂：若 64K 上 nt-on 无保护作用，
nt 从「安全配额」进一步降为「可移除」，S6 披露句可删。

## 缺口 4：完整请求 E2E 计时（历史欠账，与本批并列）

第 34 轮收官结论不变：与 best paper 最大诚实差距 = 开销计入声明句。冠军配置
变了（nb32/nl4096 vs 旧 (8,320) 族），预算口径 budget_avg 1021.5 基本不变，
但 L1 页数 8→32、nl 320→4096 会改变 L1/L2 成本构成——旧 E152 选择器计时
不再代表冠军路径。TG（batch + CUDA Graph）完成后需重测冠军路径的
prepare/select/gather 占比。

## 顺序建议

T4（预登记在跑）→ 缺口 2（与 T4 同批）→ 缺口 1（纯离线重析，材料已齐）→
T6 解冻判断（缺口 1.2 出结果后）→ 缺口 4（TG 之后）。手稿数字冻结至缺口 1.2
出结果；在此之前 A6 图与 S6 披露可先行（已按「descriptive + mechanism」口径
完成，不依赖显著性）。
