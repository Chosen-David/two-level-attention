# E135 / E148 sink 探针的解释边界

## 判定与范围

**DAC-SINK-TRACE-PROTOCOL-20261015 / evidence-scope / confirmed**。这是对已有 JSON 与生产者的只读核验，不新增实验矩阵、不修改默认配置、不启动 GPU，也不表示停止任何既有作业。

历史证据冻结于 `2014eeb479c748e2a430ad0cafe17bbfb2b12f53`；复查到 `9f3f1ca7212e248f2012bb5e4743d39d7e56530c` 已修正子块 sink 页坐标。坐标问题的根因、修复及 CPU 测试见 [既有修复记录](https://github.com/Chosen-David/two-level-attention/blob/9f3f1ca7212e248f2012bb5e4743d39d7e56530c/agent_doc/advice/2026-10-15_sink_subblock_trace_coordinates_fix_by_gpt.md)，这里不重复报缺陷。该提交没有生成新的 v2 trace 分数；原 sink4/16 champion 数值仍须重算。

本记录补充三个此前未明确登记的 E135/E148 专属解释问题：固定 mid 预算不等于固定最终预算；同源同 trace 复算不等于独立复证；历史小幅 mass 差不能命名为噪声级、等效或 Qwen3 质量充分性。

## 1. 冻结证据与算术复核

以下 Git blob 属于历史快照 `2014eeb`：

| 路径 | Git blob SHA |
| --- | --- |
| `exp/trace/analyze_e135_sinkswa_grid.py` | `c5686db380dc938fc719f7869acc9c5cb8e09cde` |
| `exp/trace/analyze_e135_sink_ext.py` | `8f5db0131af2ce8a3ad0eef2e42edac93a752a75` |
| `exp/trace/analyze_e148_sink4_grid.py` | `7735e98ee7e7f37279f973d3cac5cb56b436cd0b` |
| `exp/trace/results/e135_sink_ext.json` | `de2a9c95b36075493a14ef68f4e30707b61cc452` |
| `exp/trace/results/e148_sink4_massgrid.json` | `0ed93221c3f9039bd39b9cc70c4b6666b8b28ff8` |
| `exp/trace/results/e135_sinkswa_grid.json` | `816f31cd83539b0118973e3f714edb69ada75b2a` |

两个新 JSON 的 SHA-256 分别为 `2749ff879454a5c1e8d0c52ed59747b4d99593d392574bd9198f6d641d910d9e`（E135 ext）与 `f90b4db9eb57c8a8314c71be1677f9499697c09db3390a06d9abfbe0d2769f61`（E148）。逐样本均值复核与记录值一致，最大误差约 `1.11e-16`；这证明汇总算术正确，不验证 trace 输入或选择语义。

历史输出，均不能当作坐标修正后的结果：

| 配置 | champion mass | 相对 sink128/SWA128 的 mass 差 |
| --- | ---: | ---: |
| sink128 / SWA128 | 0.8195661942 | 0 |
| sink4 / SWA128 | 0.8174228687 | −0.0021433256，即 −0.21433 个百分点 |
| sink4 / SWA124 | 0.8167967510 | −0.0027694432，即 −0.27694 个百分点 |

两条 sink4 对照的 **16 个样本标签均为负差**。没有重复运行方差、预设等效界限或独立确认集，不能从差值较小推出“噪声级”或“等效”。E98 对齐锚点的实测最大绝对差为 `1.8887e-6`，自校验接受阈值为 `1e-5`；这项实现一致性检查不是模型质量或样本变化的噪声估计。

## 2. 固定 B_TOK 实际固定的是 mid 配额

E148 历史生产者 `:134–157` 先强制 sink/SWA，再按 `nt_near=320`、`nt_far=1728` 做 mid top-k，即名义 mid 配额恒为 `B_TOK=2048`。保护宽度没有从该常量扣除。因此名义最终选择上界为：

- sink128/SWA128：`2048+256=2304`
- sink4/SWA128：`2048+132=2180`
- sink4/SWA124：`2048+128=2176`

这些是配额导出的**名义上界**，不是实测 unique token 数。JSON 没有逐行最终计数、候选不足计数或被 `-inf` top-k 补位的计数，不能断言实际恰好保留这些数量。将 128+128 改为 4+124 也不是保持保护总量不变：保护总量由 256 变为 128。

生产分区路径 `sparse_attn/indexer/tli_indexer.py:2166–2179` 则令 `K2_mid=K2−sink−SWA`，保护计入 K2 总预算。生产 sink4 还可能与 mid 池重叠并虚耗名额，探针在 `ts=sc_tok[:,sink:mid_hi]` 中排除了该头部，两者不能直接互证。

故该网格回答的是旧 trace 选择器在固定 mid 配额下的 mass 敏感性，并未回答生产同一最终 K2 下换保护带是否不掉质量。`indexer_gain=champion_mass−sinkswa_only_mass` 的增加也不是单独识别出的质量贡献：保护集合及其 mass 变小、mid 域变大，该差值会随这些集合变化。

## 3. 同源复算、样本身份与分母

E135 ext 直接导入原 E135 分析器并改网格；E148 明示克隆相同协议、相同 `/tmp/trace/qwen3-8b`。两份新 JSON 有相同的 16 个样本标签，**144/144 个重叠逐样本数值逐值相同**；E148 与原 E135 的 112 个共享数值也全部相同。它们可作为同源重算的一致性检查，不能作为独立实现、独立数据或独立质量复证。

`lb_narrativeqa_0` 与 `lb_narrativeqa_1` 的整条结果向量完全相同。没有输入 hash 不能断言它们的输入必然重复，但也不能直接把 16 个标签视为 16 个独立统计单位。

实际分母是：每行先以所有因果可见 token 的全维 softmax 为分母，求选集概率质量；再平均 Q heads；样本内平均被抽到的 layer/尾部 query 行；最后对样本等权平均。代码取最后 4 条 qpos，layer 步长为 `max(1,n_layers//12)`，不是所有 query 或所有层。JSON 未保存 layer/qpos 列表、有效行数、输入/模型 hash，无法从汇总文件核验实际行数。

E148/ext 网格的外层过滤是 `t+1≥4624`；sink128/SWA1024 锚点还要求 `t+1≥5248`，锚点未必与网格拥有同一行集。原 E135 的 sink128/SWA512 臂还要求 `t+1≥4736`，所以注释中的“按 (16,512) 最坏点保证所有臂同行集”并非对任意 qpos 都成立。现有 SWA128 重叠结果一致不能替代实际行 ID 对拍。

## 4. 不能把 mass 探针升级为生产或 StreamingLLM 质量复现

承接既有 [实验设计记录](2026-10-15_paper_measurement_experiment_design_by_gpt.md) 的同预算与评分身份要求，本案还存在明确映射差异：探针 L1 先合并 query 向量，再求 minmax/avg；L2 按 query-head 原始 logit 求和排名，未经过生产量化。生产 `tli_indexer.py:2054–2057` 是逐 Q head softmax 后再做 GQA mean。sum-logit 排名与 mean-probability 排名一般不等价。页坐标修正没有消除这些差异。

[StreamingLLM 原文 Table 2](https://arxiv.org/html/2309.17453v4) 的初始 token 消融采用 PG19 长流 perplexity、固定缓存总量，并研究 Llama-2、Falcon、MPT、Pythia。本案是 Qwen3-8B 冻结 Q/K trace 的 attention mass；没有流式位置重映射与 perplexity/任务质量评估。因此最多称“受四-token sink 文献设置启发的候选”，不能称“StreamingLLM 4 suffices 已在 Qwen3 复现”。也不能据此补足 Figure 1 所需的距离密度与近/远区集中度观测。

## 最小下一步与不变边界

1. 历史 JSON 保留；已修坐标问题不重复修复。先将旧数值标为历史、待 v2 重算，不写“噪声级”“独立复证”或“4 已充分”。
2. 在既有 trace 重跑获得授权且输入可用时，只复用现有分析器、冻结的 16 样本 cohort 与原配置。沿 `9f3f1ca` 写 `_v2.json`，同时核对输入 hash、实际 layer/qpos、有效行数及 unique 选择数。不得无声扩大样本或变更预算；若 cohort 身份无法对上，保留未验收状态。
3. v2 同协议重算只修复历史探针。生产采用决定另需在已批准的同最终 K2 口径内完成选择身份/保护语义对拍及质量验收；本记录不新建矩阵、不提出新的 GPU 启动作业。SWA124 仍被生产块对齐门拒绝，sink4 仍不通过批量快路的 sink 对齐门而回退。

现有代码默认仍为 sink128/SWA128；构造允许 sink4 不是默认已改或某个作业已在使用 sink4 的证据。本次没有核验到实际 sink4 运行的配置/日志关联。

以下 CPU 汇总复核命令仅在已有的只读 `2014eeb479c748e2a430ad0cafe17bbfb2b12f53` 历史快照中运行，快照须含上述原始 JSON；不切换分支、不改动工作区、不执行分析器或加载模型：

```bash
python - <<'PY'
import json, statistics
p='exp/trace/results/'
a=json.load(open(p+'e135_sink_ext.json'))
b=json.load(open(p+'e148_sink4_massgrid.json'))
shared=[(s,k) for s,v in a['per_sample'].items() for k in v
        if s in b['per_sample'] and k in b['per_sample'][s]]
print('shared',len(shared),'maxdiff',max(abs(a['per_sample'][s][k]-b['per_sample'][s][k]) for s,k in shared))
for arm in ('sk4_sw128','sk4_sw124'):
    d=[v['mavg_champ_'+arm]-v['mavg_champ_sk128_sw128'] for v in b['per_sample'].values()]
    print(arm,'n_labels',len(d),'mean_delta',statistics.mean(d),'negative',sum(x<0 for x in d))
PY
```

本次验证：GitHub 固定提交内容读取、源文件 blob/字节哈希核验、Python CPU JSON 算术重算。没有原始 trace，没有重跑模型、生产 selector、GPU 或性能测试。已提交修复的 258 子项与 34 ops 测试属于该修复记录的验证结果，本次未冒称再次运行。
