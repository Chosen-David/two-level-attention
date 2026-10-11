# Opus 历史基线盘点评估 + 可信度审查（by deepseek）

- 时间：2026-10-11 22:5X
- 触发：用户「评估整理 Opus 的代码/脚本/数据，审查有无 bug、数据是否可信；可信的部分不用重复测试」。
- 结论：**Quest/MoBA 历史数据 usable-with-scope（排序可信、绝对值旧口径、Quest 有 GQA 粒度偏离）；KV 压缩四臂最低可信不采用；sanity 门禁可用历史数据过、主表 Quest 臂须新口径 vendor 重跑。**

---

## 1. 逐项审查 Opus 文档结论

| Opus 断言 | 我方核验 | 判定 |
|---|---|---|
| Quest 历史 LB13=47.72 vs FullKV 50.36（掉 2.64） | result.json 实测：pred_1024 quest_64_16 各任务分数齐，hotpotqa -7.74 / 多数小掉点 | **趋势符合 Quest 论文 ✓** |
| Quest 预算档覆盖 pred_{512,1024,2048} | 8B/14B 均有 quest_64_{8,16,32} 12 任务 | ✓ |
| Quest 是「算法语义复现」非官方代码 | 源码核验：QuestIndexer 81 行，min/max 页上界 + topk | ✓ 但见 §2 GQA 偏离 |
| MoBA 是 TLI 框架内 `_moba_mask` 复现 | 源码核验：chunk gate（q·chunk-mean）top-K 块展开 + sink/swa 保送 | ✓ 语义对齐 MoBA 论文（training-free 版），非官方 wrapper |
| TIA/TWI 自家前代，不进对比表 | indexer_type_dict 注册、tia/twi 无外部论文锚点 | ✓ 采纳 |
| KV 压缩四臂（SnapKV/H2O/PyramidKV/StreamingLLM）无接入代码无脚本 | 本仓 data.json 仅提及，无代码可核验 | **最低可信，不采用**（Opus 也倾向重跑） |
| 全旧口径（E121 修复前） | 文件 mtime 5-25/01 月，pred 协议旧 | ✓ 不可进新口径主表 |

## 2. 新发现：QuestIndexer GQA 粒度偏离官方（审稿敏感，Opus 自己已标注）

- **Quest 官方**（quest_attention.py:139+211）：`repeat_kv` 展开 → 全 head 维度算 chunk score →
  `local_heavy_hitter_mask` 在 (bsz, **num_heads**, q, kv) 上 **per-head 独立 topk**。
- **QuestIndexer（Opus）**（quest_indexer.py:54）：`b_score = rearrange('(h g) tk -> h g tk', g=G).mean(dim=-2)`
  → **每组 G 个 head 的 score 取 mean → 组内共享同一批页选择**。
- **影响**：选择粒度更粗（每 GQA 组共享选择，非逐 head）。方向不定（可能偏保守），
  作为「Quest 复现」不严格。**我的 vendor（qwen3_quest_patch）与官方一致（per-head）**。
- **处置**：A 轨 QuestIndexer 历史数据只当 sanity/排序参照；主表 Quest 臂用 vendor（官方算法）。

## 3. 可信度分级与「哪些不用重复测」

| 数据 | 可信度 | 处置 |
|---|---|---|
| Quest 历史（LB/RULER 32K） | usable-with-scope（排序可信，GQA 偏离，旧口径） | **sanity 门禁可用**（趋势已证符合论文）→ 阶段 1 官方复现部分豁免；**主表仍须新口径 vendor 跑** |
| MoBA 历史（49.19） | usable-with-scope（training-free 语义复现） | 可作下界对照参照；官方 wrapper 另行 |
| fullkv 锚点 | E116b 新口径已收 | 复用，不重跑 |
| TIA/TWI | 自家前代 | 论文只作 ancestor/消融，标注 ours-predecessor |
| KV 压缩四臂 | 最低（无代码无脚本） | **不采用**；论文若要 T2 附录 → KVCache-Factory 重跑 |

**结论：可免测** = Quest sanity（历史证趋势）+ fullkv（E116b 已有）。
**必测** = 新口径 Quest 全量（vendor 官方算法 + A 轨交叉）、MoBA vendor、ClusterKV（A 轨替代）、DSA（复现性评估）。

## 4. 派单（用户「打满 GPU」指令，承接 2240 报告）

- **新口径 Quest 全量**（Qwen3-8B，16 任务）→ .24 8 卡打满，每卡 2 任务：
  - A 轨：`--method quest --quest_block_size 64 --quest_topk 16`（pred.py 现成）
  - B 轨 vendor：qwen3_quest_patch 接 pred 入口后同链路（per-head，官方粒度）
- 前置：LB 16 任务 jsonl 同步到 .24（本地 data/ 仅 data.json，数据在 e103/e116 归档需确认 16 任务全集）；pred.py quest 分支已支持。
- 派单记录：节点 .24 / 卡 0-7 / GPU PID 起跑后回填。

— by deepseek（本地源码核验 + 历史数据实测 + 与官方 GQA 粒度逐行比对）
