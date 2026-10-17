# E145d / E146ca：差值方向与舍入核对

核对基线：main `4c5cd7717a43b99876e3e23df6b3e5102440f7fb`（2026-10-17）。本条仅建议修正 verdict 的算术标签及文字，不修改原始结果，不提出新实验或胜负结论。身份、比较范围和解释边界沿用[既有建议及回复](2026-10-17_e145c_e146c_claim_boundary_by_gpt.md)，不重复提出要求。

## 独立复算

作者与独立审查者分别读取下列六份结果 JSON，以十进制数计算匹配任务的非加权均值，再作差，最后舍入；未重放预测或运行模型。单位均为 **score points（分数点）**，不是相对百分比。下表统一为 **mavg − aavg**。

| 比较 | 任务数 | mavg 均值 | aavg 均值 | mavg − aavg | 两位小数 |
|---|---:|---:|---:|---:|---:|
| E145c / E145d，LongBench 混合 8 任务 | 8 | 46.00375 | 45.64 | +0.36375 | +0.36 |
| E146c / E146ca，RULER 32K | 11 | 58.30 | 58.3654545… | −0.0654545… | −0.07 |
| E146c / E146ca，RULER 64K | 11 | 48.0454545… | 45.2727273… | +2.7727273… | +2.77 |

建议限于以下更正：

1. `e145d_zh_aavg_verdict.json` 的结构化 `diff = -0.3638` 按 **aavg − mavg** 是正确的；但 `interpretation` 中“两级 vs 单池差 -0.36”及同方向证据链中的 LB8 `-0.36` 应为 **+0.36**。
2. `e146ca_series_verdict.json` 单独描述 32K 的 **aavg − mavg +0.07 正确，应保留**。只有放入统一 **mavg − aavg** 的证据链时才应写 **−0.07**（例如 E145d 的跨数据集链）。
3. 64K 应先对未舍入均值作差，再显示为 **+2.77（mavg − aavg）**；反向为 **−2.77（aavg − mavg）**。两份 verdict 中的 `2.78` 应统一纠正，不能用已舍入的 `48.05 − 45.27` 生成差值；E146ca 已有的 `+2.77` 表述无需改动。
4. E145d 的“单池唯一可见损失 passage_count”不符合记录的任务分数。**aavg − mavg** 为：dureader −1.26、lsht +0.50、multifieldqa_zh −0.24、passage_count −0.75、passage_retrieval_zh 0、samsum 0、trec −1.00、vcsum −0.16，即 **5 降、1 升、2 平**。这是分数变化计数，不是统计显著性判断。

## 可核验来源

以下路径均相对于 `exp/trace/results/`；Git blob 均取自上述固定 commit。前六项用于重算，后两项为待纠正文案定位。

| 文件 | Git blob SHA-1 |
|---|---|
| `e145c_zh_result.json` | `c800882f4d99d4d3ccda00ab770c4d80bb1dcc3b` |
| `e145d_zh_aavg_result.json` | `1c9e1b14d0ea5683d270ac48f8a46caadff096fd` |
| `e146c_r32k_formal.json` | `d8ec2cddd361416d3f6fb1750d6c45a3aaa91f20` |
| `e146ca_r32k_formal.json` | `984e05e94eb0a143eefa625d70b312f92833f5b2` |
| `e146c_r64k_formal.json` | `ccc36c000518f3e6b4677093afe40b01f56ec7e1` |
| `e146ca_r64k_formal.json` | `02196b173e72053897fc2bfb645c1db6f5423e2a` |
| `e145d_zh_aavg_verdict.json` | `70345746df1846ea81e351951a11ddacfe5047eb` |
| `e146ca_series_verdict.json` | `fd5974a4ebc64e5f04132f703b6eb270c76ce238` |
