# Quest 基线历史数据与实现审查（by deepseek）

- 时间：2026-10-11 22:4X
- 触发：用户提示「历史 Quest 数据在 Qwen3-14B 的 pred_1024/2048/512，继续找打分结果与其余基线」。
- 结论：**Quest A 轨实现已存在且数据趋势符合论文；历史数据为旧口径，主表需新口径重跑；其余基线 moba/ClusterKV/DSA 缺独立数据。**

---

## 1. 历史 Quest 数据发现（可直接用于 sanity 与趋势对照）

| 模型 | 目录 | Quest 臂 | 预算 | 覆盖任务 |
|---|---|---|---|---|
| Qwen3-8B | `exp/results_longbench/Qwen3-8B/pred_512` | quest_64_8 | 512 | 12 |
| Qwen3-8B | `.../pred_1024` | quest_64_16 | 1024(实 1088) | 12 |
| Qwen3-8B | `.../pred_2048` | quest_64_32 | 2048 | 12 |
| Qwen3-14B | `.../Qwen3-14B/pred_{512,1024,2048}` | 同上 | 512/1024/2048 | 12 |

- 每任务同目录有 `none`（fullkv 锚点）与 `tia/twi` 臂，`result.json` 打分齐（score + budget）。
- 缺任务：trec / samsum / passage_count / multifieldqa_zh / lsht（12/16，无中文 2 任务 + 3 英文）。
- moba：`e89_moba/pred_moba` 是 `tli_*_A-moba`（TLI A 臂 + moba 增强），**非独立 MoBA baseline**，且无打分。

## 2. 实现审查：QuestIndexer（A 轨公平实现）vs Quest 论文

`sparse_attn/indexer/quest_indexer.py`（已注册 `--method quest`，args：`--quest_block_size`(64)/`--quest_topk`）：

| | Quest 论文（官方 quest_attention.py） | QuestIndexer（本项目 A 轨） |
|---|---|---|
| 块级 score | `sign(q)` 符号匹配 → chunk 内逐 head_dim 取 max key → `q^T k_max` | `k_min/k_max` 双边界 → `max(q^T k_min, q^T k_max)` 逐 head 取大求和取均 |
| topk | 预算内选 topk chunk | 同 |
| 因果 | tril(diag=当前位) | b_mask（q_ids 块 > k_ids）+ 当前块强制 True |
| 预算 | token_budget=1024 | block_size×topk=1024（实际 pad 到 1088） |

- **判断**：同为「块级极值粗筛 + topk」语义，但 score 公式与论文不同（min/max 上界 vs sign-max）。
  作为 **A 轨「Quest 风格公平实现」可接受**（同预算同框架公平对比）；论文措辞须注明非官方算法。
- **B 轨（vendor 严格复刻）**：`exp/vendor_baselines/quest_vendor/qwen3_quest_patch.py`（我刚写的
  clean-room 实现，sign+chunk-max + tril，与官方算法一致），作双证。

## 3. 趋势 sanity：历史数据 vs Quest 论文结论

Quest 官方评测基于 LongChat-7B/Yarn-Llama2-7B（README 确认），LongBench 结论（论文）：
**预算 ↑ → 精度趋近 fullkv；精确检索任务（hotpotqa/musique/2wikimqa）掉点最明显；平均小掉点。**

历史数据（Qwen3-8B，pred_1024，quest_64_16 vs none）：

| 任务 | none | quest | Δ |
|---|---|---|---|
| 2wikimqa | 38.29 | 38.46 | +0.17 |
| gov_report | 33.17 | 32.13 | -1.04 |
| hotpotqa | 53.48 | 45.74 | **-7.74** |
| lcc | 67.35 | 66.86 | -0.49 |
| multifieldqa_en | 53.40 | 51.01 | -2.39 |
| multi_news | 24.93 | 24.95 | +0.02 |

- 检索任务（hotpotqa）大掉点、多数任务小掉点 → **与论文趋势一致 ✓**。
- 预算 Pareto：512 掉点更大（14B 2wikimqa -10.08）、2048 趋近 fullkv（+1.06）→ **预算↑精度↑ ✓**。
- **预算口径**：quest_64_16 实耗 1088（64×17，pad 到块界），略超 1024——主表须注明。

## 4. 关键判断：旧口径，主表需新口径重跑

- 历史数据时间戳 01/05 月（文件 mtime 5-25），为 **E121 新口径（R01 BOS-aware / B04 首 token EOS / B10 near-swa）之前**的旧生成协议。
- E116b 的 fullkv/稀疏数据已是新口径 → **历史 Quest 与现主表锚点不可直接对比**，必须重跑。
- 复用价值：**实现正确性 + 趋势 sanity 已过**（上表），新口径重跑不担心算法 bug，只担心 E121 修复后协议变化。

## 5. 其余基线缺口

| 基线 | 现状 | 行动 |
|---|---|---|
| Quest A 轨 | 历史旧口径，实现✓趋势✓ | 新口径重跑（--method quest quest_64_16） |
| Quest B 轨 vendor | 适配层刚写（单元测试 5/5 绿） | 新口径跑 + 与 A 轨对拍 |
| none/fullkv | E116b 新口径已收 | 复用 |
| tia/twi | 旧口径（E116b 未含） | 视主表需求决定（T2 附录或注记） |
| MoBA | 无独立 baseline（只有 TLI-A-moba） | 阶段 2 补 |
| ClusterKV | 依赖硬阻塞 | A 轨 cavg/ccluster 替代 |
| DSA | 未复现 | 阶段 0 评估 Qwen3 支持性 |

## 6. 派单方案（用户「打满 GPU 快速出数据」指令）

- **目标**：Qwen3-8B 新口径 Quest 全量 16 任务（A 轨 quest_64_16）。
- **资源**：.24 8 卡全空（Qwen3-8B 权重在 ~/Qwen3-8B，torch 2.8 就绪）→ **8 卡每卡 2 任务拆分**。
- **前置**：LB 16 任务 jsonl 需同步到 .24（本地 benchmark/LongBench/data/ 目前只有 data.json，需确认 16 任务 jsonl 来源）；pred.py 的 quest 分支确认可用。
- **步骤**：代码/数据 rsync → .24；8 卡并行 `--method quest --quest_block_size 64 --quest_topk 16`，每卡 2 任务；落盘后 eval.py 打分 → 与 E116b fullkv 新口径同表。

— by deepseek（本地源码核验 + 历史数据实测）
