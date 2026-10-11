# two-level GPU 监督器 Round 2（by deepseek）

- 时间：2026-10-12 01:03
- 触发：cron 244242d1（每 2h）
- 结论：**发现 1 个真 bug（Qwen3-32B hidden_size reshape 崩）→ 已修已推**；**12 卡空闲全派满**（Quest vendor 8B/32B + DeepSeek-V2-Lite-Chat MLA fullkv）

---

## 1. GPU 占用表（核验后）

| 池 | 卡 | 占用 | 在跑任务 |
|---|---|---|---|
| 本地 | 0,1 | 96/69GB，98/85% | **Qwen3-32B quest_vendor**（本地 62G 模型直跑，分担 narrativeqa + qasper） |
| 10.238.9.187（.187） | 0-7 | 86-108GB，100% | **Qwen3-32B quest_vendor** 14 任务（00:43 TS=20261012-004348，8 卡均衡） |
| 33.32.33.24（.24） | 0-7 | 17-39GB，52-100% | **Qwen3-8B quest_vendor** 16 任务（旧 TS=20261011-235032 继续跑，multifieldqa_en/2wikimqa 已完成） |
| 10.238.139.251（.251） | 0,1 | 30/27GB，74/77% | **DeepSeek-V2-Lite-Chat fullkv** dense 锚点（MLA 矩阵，16 任务每卡 8 串行） |

**E125（.187）gov_report redo 已收官**（此前 GPU6 207min redo 完成，.187 无 pred 进程 → 全空 → 本轮全派给 32B）。共 12 卡空闲 → 12 卡全派满。

## 2. bug 审查：发现 1 个真 bug（已修已推）

### Bug：Qwen3-32B quest_vendor decode reshape 崩溃

- **现象**：本地 32B 冒烟 decode step 崩
  `RuntimeError: shape '[1, 1, 5120]' is invalid for input of size 8192`（quest_attention_official.py:258 reshape 到 self.hidden_size）。
- **根因**：官方 Quest forward 末尾 `attn_output.reshape(bsz, q_len, self.hidden_size)`，要求
  `hidden_size == num_heads * head_dim`（Llama/Mistral 天然成立）。适配层 enable_quest 注入的
  `hidden_size = config.hidden_size`；Qwen3-32B config 是 **num_heads=64, head_dim=128 → 8192，而
  hidden_size=5120**，两者不等 → reshape 崩。Qwen3-8B 恰好 4096==4096 未暴露（所以 .24 8B 一直正常）。
- **修复**（exp/vendor_baselines/quest_vendor/qwen3_quest_patch.py）：注入
  `module.hidden_size = module.num_heads * module.head_dim`（对 Llama/Mistral/Qwen3-8B 与
  config.hidden_size 相同，零影响；对 32B 正确）。**官方算法文件 quest_attention_official.py
  逐字节不动**，选择算法零改动。
- **验证**：本地 Qwen3-32B 冒烟重跑 PASS——64 层全 patch、prefill 0.6s、decode 每步 0.1-0.2s、
  66GB 单卡、全有限。commit ace966b 已推 origin/main。

### 新增模型支持（配合 S-T023 MLA 矩阵）

- pred.py：`--model` choices 加 `DeepSeek-V2-Lite-Chat` + build_chat apply_chat_template 分支；
  model2path/maxlen config 加条目 → 跑 MLA fullkv dense 锚点（method=none 命名 "none"，
  S-T023 协议：MLA 无稀疏原生支持，fullkv 为一切对比锚点）。

## 3. 本轮派单（12 卡全派满，用户「不要心疼 GPU」指令）

| 池 | 任务 | 协议 |
|---|---|---|
| .24 8 卡 | Qwen3-8B quest_vendor 16 任务 | vendor（官方 quest_attention 逐字节）+ q_norm/k_norm/scaling 适配，token_budget=1024/chunk=16 |
| .187 8 卡 | Qwen3-32B quest_vendor 14 任务（narrativeqa/qasper 由本地分担） | 同 vendor 协议，hidden_size 修复后 |
| 本地 2 卡 | Qwen3-32B quest_vendor narrativeqa + qasper | 同 |
| .251 2 卡 | DeepSeek-V2-Lite-Chat fullkv 16 任务 | method=none dense 锚点（MLA） |

- .187 曾误启动全 16 任务与本地重复 → 已 kill 旧批，重派 14 任务 8 卡均衡（card6=lcc、card7=repobench-p 单任务）。
- 模型：Qwen3-32B 62G 已 rsync 到 .187；DeepSeek-V2-Lite-Chat 30G 已 rsync 到 .251。

## 4. 下轮待办

1. **Quest vendor 8B 16 任务完成后**（.24，预计 ~1.5h）：eval.py 打分 → 空卡贴合检查（对照 Quest 论文：检索任务大掉点、平均小掉点）→ 主表拼接。
2. **32B + DeepSeek fullkv 完成后**：同流程打分 + 主表拼接；DeepSeek fullkv 是 TLI-MLA 对比锚点，留给 Opus。
3. **ClusterKV B 轨**：pylibraft 网络窗口待开（pypi.nvidia.com 超时）；MoBA 依赖 flash-attn varlen 钉 2.6.3 待评估。
4. **DSA 复现性评估**：查官方开源对 Qwen3 支持性。
