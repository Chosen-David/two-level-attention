**独立核验补充结论：旧 100 配对 screen 的官方评分缺口已闭合；整体实验验收仍为 `pending`，全量 1300 仍为 `pending`。**

本轮由独立审查者执行，全部 `exec login=false`，仅使用 CPU、只读文件及内存脚本；未启动模型或 GPU，未安装依赖，未写源文件、数据或脚本。本文交外层保存到指定轮次目录。

已读取前次 `independent_screen/REPORT_access_retry.md` 及主控生成的 scorer 脚本、结果和环境探针。主控的 `errors=[]` 未被直接作为通过依据。

本轮实际完成：

- 在 pc02 冻结源码目录独立调用 `sample_score`，重算 **300/300** 个评分，包括 GovReport **60/60** 个 ROUGE 分数；与保存分数的绝对误差均不超过 `1e-12`，全部有限且处于 `[0,1]`。
- 核对 **100 个唯一样本、每任务 20 个、每样本三臂完整**，与冻结 `selected_rows` 一致。
- **5 个数据集哈希匹配，100 份原始答案匹配数据行**；seed=42 重新抽样顺序一致。
- 离线 CPU tokenizer 重建 **100 个 prompt**，长度、截断标记和输入 token 哈希全部一致；**300 个生成序列解码**与保存 prediction 全部一致。
- 冻结源码 **72 项哈希**在检查前后全部匹配；原始文件摘要与前次报告、主控逐文件清单一致，结束时再次核对未变。

实际评分环境为 Python **3.9.9**、`rouge 1.0.1`、`jieba 0.42.1`、`numpy 2.0.2`、`transformers 4.56.2`，后端为 **`difflib:stdlib`**。这些版本固定本次重评分环境，不代表已补齐历史生成环境闭包。

百分制复算结果如下，差值单位为百分点：

| 任务 | full | far_0.05 | dual_0.05 | far/dual − full |
|---|---:|---:|---:|---:|
| Qasper | 56.100883 | 54.042059 | 54.042059 | −2.058824 |
| HotpotQA | 65.523810 | 65.523810 | 65.523810 | 0 |
| Musique | 45.083333 | 45.083333 | 45.083333 | 0 |
| GovReport | 35.227082 | 34.835478 | 34.835478 | −0.391604 |
| RepoBench-P | 72.050000 | 71.950000 | 71.950000 | −0.100000 |
| **宏 AVG5** | **54.797022** | **54.286936** | **54.286936** | **−0.510086** |

**这批固定 screen 确有实际评分下降，但 Musique 没有下降。** 目录中的 `followup_prompt_correction.md` 也明确说明“保留 Musique 下降”是交接笔误。本轮依据原始数据纠正该措辞，不预设下降方向，也不推断全量 Musique 的结果。

六项检查的更新判定：

| 检查 | 判定与剩余缺口 |
|---|---|
| `implementation` | **限定 pass**：沿用前次静态调用链审查，本轮重新验证 72 项源码绑定；未新增 GPU 执行认证。 |
| `reference_boundary` | **inconclusive**：独立张量参考及极端边界仍缺；near-only 跨页老化转 Far 尚无该 screen 的运行覆盖。 |
| `data_integrity` | **限定 pass**：100 样本配对、原始答案、数据集、抽样、prompt 和输出解码均已独立核对；模型权重完整绑定仍缺。 |
| `numerical_sanity` | **限定 pass**：300 个分数、任务均值和宏平均独立复算一致；前次 cache 不变量检查未在本轮重复执行。 |
| `measurement_validity` | **评分范围 pass**：官方评分、答案、分母与配对缺口闭合；不认证耗时、峰值显存或性能收益。 |
| `reproducibility` | **整体 inconclusive**：评分与输入重建可复现；未重跑模型生成，历史依赖闭包和权重绑定仍不完整。 |

Near-only 与 Both 的限制必须随结果保留：dual 臂共 **3600 个层—请求决策**，其中 keep=3272、far-only=326、near-only=1、both=1。后两类各仅占 **0.02778%**，各涉及一个 Musique 请求，不能据此宣称广泛质量覆盖。唯一 near-only 请求未实际覆盖跨页老化迁移。

关键绑定：

```text
100 个原始请求文件清单 SHA-256：
6404104e4c5fcd711415924b3d125b89df59be1a594d44404db51d6dd23c93ea

manifest_rank0.json：
ae460b2a95babf579492379b433a7cab05f089571c1e2453be3e4b21b1c97e50

dispatch_plan.json：
a8960f29aede6aacad63b92dfbf9d843cae9a7a06997d6a4347db5303ad15f4c
```

清单摘要计算规则与前次一致：按路径排序，拼接 `相对结果根路径 + 空格 + 文件SHA256 + 换行` 后计算 SHA-256。

**保持 `pending` 的原因**是参考/边界及生成复现仍有缺口，且未取得事前固定的完整 validation plan 和结果合同。本轮事后复算不能补造事前计划，也不能把未执行检查改为 N/A。因此不签发整体 `usable-with-scope`；可以确认上述固定预测的评分核查事实，但不能据此宣称算法无损、普遍等价或全量验收完成。这里的 `full` 仍是关闭 region gate 的 TLI 基线，并非 dense full-attention decode。