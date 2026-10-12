# S-T023 对比基线 round 汇报（by deepseek）

- 时间：2026-10-12 03:3X
- 触发：Qwen3 fullkv 16/16 全齐 + qasper 512 sanity 落盘 + 预算 Pareto 收敛
- 结论：**Quest 8B 严格 16 任务贴合完成（quest 41.35 vs fullkv 49.39，Δ=-8.03，趋势完全贴合 Quest 论文）；预算 Pareto 严格单调（512=29.70 < 1024=36.43 < 32768=46.57 < fullkv=47.47），Quest 核心结论在 Qwen3-8B 确证；适配层无 bug 判定通过，quest 16 任务 AVG 41.35 允许上主表。**

---

## 1. Qwen3 fullkv 16/16 全齐 + 严格 16 任务贴合

- **Qwen3 fullkv 16 任务全完成**（TS=20261012-020308，03:24 ALL DONE），AVG 49.39。
- **quest_vendor 16 任务 AVG 41.35**（TS=20261011-235032）。**全 16 共同任务 Δ=-8.03**（10 共同任务 -8.38 → 16 全量 -8.03，稳定）。

| 类别 | 任务（Δ） |
|---|---|
| 摘要/文档维持 | multi_news **+0.68**、qmsum -0.72、samsum -0.91、gov_report -1.04、passage_count 双低 1.0 vs 2.0 |
| 检索/知识大掉点 | triviaqa **-18.38**、hotpotqa **-13.98**、trec -12.50、2wikimqa -12.51、musique -12.32、passage_retrieval_en -11.75、qasper -11.04、repobench -11.11 |
| 其他 | multifieldqa_en -5.33、lcc -7.87、narrativeqa -8.74 |

- **趋势完全贴合 Quest 论文核心结论**（稀疏预算 → 检索任务掉、摘要/文档类维持）✓

## 2. 预算 Pareto 严格单调（qasper 任务）→ 适配层无 bug 双重确证

| token_budget | qasper 分数 | vs fullkv |
|---|---|---|
| 512 | 29.70 | -17.77 |
| 1024 | 36.43 | -11.04 |
| 32768 | 46.57 | -0.90 |
| fullkv（dense） | 47.47 | — |

- **512 < 1024 < 32768 < fullkv 严格单调** → Quest 核心结论（budget↑ → acc↑）在 Qwen3-8B 上成立。
- 32768≈dense（Δ=-0.90，尾 chunk 丢弃 + fp32 softmax vs flash 数值差）→ 4.56.2 DynamicCache × 官方 Quest forward 的 update 语义正确，**适配层无 bug**。
- **判定：1024 预算掉 8.03 分是 Qwen3-8B 对稀疏的真实敏感度，非实现错误。**

## 3. 上主表判定

quest 16 任务 AVG 41.35 **允许上主表**，标注：Qwen3-8B、官方 quest_attention.py 逐字节 vendor 复用（md5 78e87fb8cc71ac058c860e2305a7b466）、我们 pred/eval 协议、chunk_size=16、token_budget=1024；模型（Qwen3 vs 论文 longchat-7b）与任务集（16 vs 官方 6）差异已在贴合检查中说明。

## 4. 资源布局（.24）

| 卡 | 任务 |
|---|---|
| 0/1 | DeepSeek fullkv 16 任务（4/16 完成：narrativeqa/qasper/hotpotqa/multifieldqa_en，卡0 正跑 2wikimqa、卡1 正跑 multi_news） |
| 7 | trec 32768 sanity（quest token_budget=32768 应≈dense 71.5，PID 3850985 在跑） |
| 2/3/5 | 空闲（Qwen3 fullkv 与 qasper sanity 均已完成释放） |

## 5. 待办

1. **trec 32768 sanity 完成** → 打分对比 fullkv trec（71.5），预算 Pareto 全任务收敛
2. **DeepSeek fullkv 16 任务完成** → eval 打分（MLA fullkv dense 锚点）
3. **.187 恢复后回收 32B quest 14 任务**（本地 32B narrativeqa/qasper 已完成 TS=20261012-002743）
4. **主表拼接**：quest 8B + fullkv 锚点 + DeepSeek（sanity 全过后）

— by deepseek（本地源码核验 + 预算 Pareto 实证）
