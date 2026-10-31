# B 组 NFA 消融独立科学验收报告

生成时间：2026-10-31T12:10:21.200516+08:00。PROJECT_ROOT：`/home/wangyuanshuo02/two-level-attention`。角色：独立只读 reviewer，未参与数据生成。

**状态：usable-with-scope。** 本次指定的 2000/2000 请求完整性、identity、官方评分复核与配对统计检查全部通过；适用范围见文末限制。

## 1. 完整性与评分复核

- 覆盖 5 个 cell × 2 个任务 × 200 个原始 row；请求文件、唯一 row、`status=complete` 均为 **2000/2000**，缺失 0、重复 0、额外 row 0。
- 每份请求的 dataset、row_index、文件名和 manifest identity 均匹配；manifest identity 与 protocol 摘要、cell id 与 spec 摘要也一致，五组 spec 与题设配置一致。
- `2d52db39d26d/gov_report` 有三份重叠续跑 manifest（0–200、100–200、150–200），identity/protocol 完全一致；实际请求仍严格为 200 个唯一 row，不构成重复请求。
- 使用官方 `benchmark.LongBench.eval.sample_score` 对全部请求由 `prediction` 与 `answers` 重算：2000/2000 误差 ≤1e-12，最大绝对误差 **0**。`eval.py` 与 `metrics.py` 当前 SHA-256 均和全部 manifest 记录一致。
- 所有 cell 与生产默认基线在同任务同 row 上的完整 input 元数据及 answers 全部一致，共 2000 个配对身份检查通过。

## 2. 配对 bootstrap 方法

- 基线：`2d2a466b4e9a`（生产默认：far=minmax、near=avg、boundary=True、partition=True）。
- 每格 n=200；分数以百分数报告，差值与 95% CI 以百分点（pp）报告。采用同 row 配对差值，percentile paired bootstrap，B=10000、seed=20261031、线性分位数插值。
- 每个任务生成一套重采样 row 索引并由五个 cell 共用；均值与 CI 另以标量 `math.fsum` 和排序插值独立复核，全部通过。

## 3. 汇总表

| 任务 | cell | 配置 | 均值 (%) | 相对生产默认 (pp) | 95% CI (pp) |
|---|---|---|---:|---:|---:|
| hotpotqa | `2d2a466b4e9a` | 生产默认：MinMax/Avg，分区，有边界页 | 59.1649 | +0.0000 | [+0.0000, +0.0000] |
| hotpotqa | `4284d54b0e30` | 全局 MinMax/MinMax，无分区，有边界页 | 59.1617 | -0.0032 | [-0.0096, +0.0000] |
| hotpotqa | `2d52db39d26d` | 分区 MinMax/MinMax，有边界页 | 59.1526 | -0.0123 | [-0.0368, +0.0000] |
| hotpotqa | `06daedae75b2` | 分区 Avg/Avg，有边界页 | 59.4149 | +0.2500 | [+0.0000, +0.7500] |
| hotpotqa | `a5f4cb3f7cf2` | 生产默认打分/分区，无边界页 | 59.1649 | +0.0000 | [+0.0000, +0.0000] |
| gov_report | `2d2a466b4e9a` | 生产默认：MinMax/Avg，分区，有边界页 | 33.8183 | +0.0000 | [+0.0000, +0.0000] |
| gov_report | `4284d54b0e30` | 全局 MinMax/MinMax，无分区，有边界页 | 33.8173 | -0.0010 | [-0.3868, +0.3929] |
| gov_report | `2d52db39d26d` | 分区 MinMax/MinMax，有边界页 | 33.4394 | -0.3788 | [-0.7462, -0.0232] |
| gov_report | `06daedae75b2` | 分区 Avg/Avg，有边界页 | 33.8224 | +0.0042 | [-0.2597, +0.2652] |
| gov_report | `a5f4cb3f7cf2` | 生产默认打分/分区，无边界页 | 33.4931 | -0.3252 | [-0.6534, -0.0042] |

## 4. 结果解读

### hotpotqa

- `4284d54b0e30`：差值 -0.0032 pp，95% CI [-0.0096, +0.0000] pp；区间含 0，本次未检出相对生产默认的均值差异。
- `2d52db39d26d`：差值 -0.0123 pp，95% CI [-0.0368, +0.0000] pp；区间含 0，本次未检出相对生产默认的均值差异。
- `06daedae75b2`：差值 +0.2500 pp，95% CI [+0.0000, +0.7500] pp；区间含 0，本次未检出相对生产默认的均值差异。
- `a5f4cb3f7cf2`：差值 +0.0000 pp，95% CI [+0.0000, +0.0000] pp；区间含 0，本次未检出相对生产默认的均值差异。

### gov_report

- `4284d54b0e30`：差值 -0.0010 pp，95% CI [-0.3868, +0.3929] pp；区间含 0，本次未检出相对生产默认的均值差异。
- `2d52db39d26d`：差值 -0.3788 pp，95% CI [-0.7462, -0.0232] pp；区间不含 0，本次配对重采样中检出均值低于生产默认。
- `06daedae75b2`：差值 +0.0042 pp，95% CI [-0.2597, +0.2652] pp；区间含 0，本次未检出相对生产默认的均值差异。
- `a5f4cb3f7cf2`：差值 -0.3252 pp，95% CI [-0.6534, -0.0042] pp；区间不含 0，本次配对重采样中检出均值低于生产默认。

以上为未校正多重比较的探索性区间，不作冠军选择或等效/无损声明。

## 5. 适用范围与限制

- 本验收只覆盖现有 B 组最终请求文件、最终预测文本、官方重评分、manifest/请求 identity 和行级配对统计；未重跑 GPU，也未重新生成任何请求。
- 未独立复核 token→prediction 解码、EOS/生成长度、逐层门控/选择回执、KV 几何、完整运行环境或模型权重，因此不把本报告解释为整个生成实现的无 bug 证明。
- 每个任务每格只有单次生成；CI 不含重复运行方差，且未做多重比较校正。CI 含 0 仅表示未检出差异，不是等效证明。
- 生产默认基线不是 dense/full-KV 基线；HotpotQA 与 GovReport 不合并为跨任务总分。

机器可读明细见 `completeness_checks.json` 与 `paired_statistics.json`。
