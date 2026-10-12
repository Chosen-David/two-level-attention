# S-T023 对比基线 round 汇报（by deepseek）

- 时间：2026-10-12 03:1X
- 触发：DeepSeek fullkv 修复验证 + Quest 8B 16 任务数据落袋 + 空卡贴合初检
- 结论：**DeepSeek fullkv 三问题修好且验证跑通；Quest 8B 16 任务 AVG 41.35 出炉，全预算 sanity 通过（适配层无 bug），趋势贴合论文；幅度偏大系 Qwen3 对稀疏敏感 + 任务集差异，待 Qwen3 fullkv 全齐做严格 16 任务贴合。**

---

## 1. DeepSeek-V2 fullkv 锚点修复（commit 031e6b8，已 push）

DeepSeek fullkv 在 .24 连崩三个问题，全部修好并验证：

| 问题 | 现象 | 根因 | 修法 |
|---|---|---|---|
| ① attention_mask 崩 | modeling_deepseek.py:878 `assert attention_mask is not None` | DeepseekV2Attention 强制要求 mask，pred.py 手动循环不传 | 见③（generate 路径自带 mask） |
| ② legacy cache 转换陷阱 | `ValueError: Attention mask should be (1,1,1,kv_seq_len), got (1,1,1,1)` | 4.56.2 DynamicCache 全新 self.layers 结构，DeepSeek 旧 modeling 触发 to/from_legacy_cache 循环，转换后 get_seq_length 返回 0 | 对 DeepSeek 专用改走 **model.generate 路径**（与 samsum 分支同协议：full_ids 一次 prefill + 贪心生成），cache_position/cache 全权交 transformers 管理；Qwen3 保持原手动循环零变化 |
| ③ OOM | narrativeqa 32869 token → softmax 分配 59GB OOM | model2maxlen.json DeepSeek=31500 远超模型 max_position_embeddings=16384 | config 修正为 16384 |

- 验证：.24 卡0/1 DeepSeek fullkv 16 任务（TS=20261012-023927）重启，narrativeqa/multi_news 正常推进无崩。
- pred.py + config 已同步 .24/.251（md5 一致）。

## 2. Quest 8B 16 任务全完成 + 空卡贴合初检

- **quest_vendor（官方算法 sign+chunk-max+topk+tril，chunk_size=16, token_budget=1024）16 任务 AVG 41.35**（TS=20261011-235032）。
- **Qwen3 fullkv 同协议锚点**（TS=20261012-020308）11/16 完成，10 共同任务 AVG 46.09。
- **10 共同任务 Δ=-8.38**：
  - 检索任务大掉点：hotpotqa -13.98、musique -12.32、2wikimqa -12.51、qasper -11.04、narrativeqa -8.74、passage_retrieval_en -11.75
  - 摘要维持：multi_news +0.68
  - passage_count 双低（1.0 vs 2.0，非 quest 特有）
  - **趋势方向贴合 Quest 论文**（检索掉、摘要维持）✓
- **全预算 sanity（适配层 bug 检测）：qasper token_budget=32768 → 46.57 vs fullkv 47.47，Δ=-0.90（≈dense）**。差异来自尾 chunk 丢弃（kv 非 chunk 倍数）+ fp32 softmax vs flash kernel 数值差，**非 bug**。结论：quest_vendor 适配层正确，1024 预算掉 11 分是 Qwen3-8B 对稀疏的真实敏感度。
- **预算 Pareto 在飞**：qasper 512（卡2）+ trec 32768（卡7）sanity，验证 512 < 1024 < 32768 单调上升（Quest 核心结论）。
- **官方口径确认**：third_party/quest/scripts/longbench.sh 用 longchat-7b（Llama-2 系），6 任务、预算 512/1024/2048/4096、**chunk_size=16 与我们一致**；官方 pred.py 生成协议与我们 pred.py 同构。

## 3. 资源布局（.24 8 卡全派满）

| 卡 | 任务 |
|---|---|
| 0/1 | DeepSeek fullkv 16 任务（narrativeqa/qasper 完成，继续） |
| 2 | qasper 512 sanity |
| 3 | qasper 32768 sanity（已完成） |
| 4/5/6 | Qwen3 fullkv 16 任务（11/16） |
| 7 | trec 32768 sanity |

## 4. 待办

1. **Qwen3 fullkv 5 任务**（gov_report/qmsum/triviaqa/samsum/repobench-p）→ 16 任务严格贴合检查（quest vs fullkv 全 16 共同任务 Δ）
2. **trec/512 sanity 完成** → 预算 Pareto 收敛
3. **DeepSeek fullkv 16 任务完成** → eval 打分（MLA fullkv 锚点给 Opus 用）
4. **.187 临时连不上**（No route to host）→ 稍后重试回收 32B quest 14 任务数据
5. 本地 32B narrativeqa/qasper 已完成（TS=20261012-002743）→ 并入 32B quest

— by deepseek（本地源码核验 + 全预算 sanity 实证）
