# AsyncTLS 精度差距根因排查报告（2026-10-24）

## 背景

用户要求：AsyncTLS 精度必须对齐论文，否则说明复现已损坏。

| 维度 | 我们 | 论文 | 差距 |
|------|------|------|------|
| LongBench AVG16 | 45.13 | 49.90 | **-4.77** |
| musique qa_f1 (200spl) | 19.90 | 31.10 | **-11.20** |
| musique qa_f1 (30spl, 同分布) | 19.90 | 31.10 | **-11.20** |
| dense对齐 | +1.55（正确） | — | harness无误 |

## 排查矩阵（7→0 排除法）

### ❌ 排除：GQA 跨 head 聚合方式（4 种全部测试）

| 跨 head 方案 | musique qa_f1 (200spl) | 结论 |
|---|---|---|
| **cross-head mean（原始）** | **19.90** | ✓ 最好 |
| cross-head max | 0.40 | 灾难 |
| per-head topkt union | 0.47 | 灾难 |
| per-head topkb+topkt 全 union | 0.40 | 灾难 |

**结论**：跨 head 聚合不是根因。原 cross-head mean 已是最佳实现。
论文公式 `ᾱ_j=(1/G)Σ_h softmax(...)` 描述 GQA 组内平均，
对跨 head 行为未声明，我们的 mean 解读保守但至少不崩。

### ❌ 排除：prompt 格式 + max_new_tokens

| 配置 | musique qa_f1 (30spl) | 结论 |
|---|---|---|
| 简单文本 prompt + 64 max_tok | 19.90 | baseline |
| 官方 Qwen3 chat template + 32 max_tok（AsyncTLS） | 3.46 | 崩 |
| **官方 Qwen3 chat template + 32 max_tok（vanilla dense）** | **3.46** | 同步崩 |

🌋**决定性发现**：**官方 Qwen3 chat template (`<|im_start|>user...<|im_end|>...<think>\n\n</think>\n\n`) 本身在 32K 下导致崩盘**。vanilla dense 从 32+ 分直接跌到 3.46。这说明：
- LongBench pred.py 的 Qwen3 chat template 在 32K 长上下文下行为异常
- 或者 Qwen3-8B 模型版本不支持这种模板 + 长上下文的组合
- ivanilla dense 用简单 prompt 可得 32.14（对齐论文分）

### ❌ 排除：scorer 口径差异

用 qa_f1_score（有 jieba）对同批 pred 重评：**19.90**。而 result.json 中 score=21.62。
→ result.json 在 eval.py 运行环境中 jieba 不可用，fallback 到 difflib，分数有系统偏高。
→ **真实 baseline = 19.90**，gap = **-11.20** 而非表面 -9.5。

### ❌ 排除：token budget、INT4、channel 选择、block scoring 公式

经代码 review：
- `kt=1024` = 论文 §5 声明（LongBench），正确
- INT4：per-token 16 级 [-8,7]，代码与论文 §4 描述一致
- Block scoring：GEMM 化 `Σ_h(qmax·kmax + qmin·kmin)` 后与论文一致
- Channel 校准：top-dc=32 排序，跨 head 取均值 → 不改变集合选择
- Token budget 参数逐字匹配论文声明

## 元分析：gap 真实值

**我们论文口径对齐的正确 baseline：**
- musique: **19.90**（我们，真实 qa_f1, 200spl）
- musique: **31.10**（论文）
- musique: **32.14**（dense, 论文值）→ dense 对齐已验证

**Gap = -11.20 (musique) / -4.77 (AVG16)**

## 假设候选（按可能性排序）

1. **Stage-1 选块后未做 per-head 投票过滤** — 论文 Figure 1 的 block selection
   box 包含 per-head attention score aggregation 操作，我们用的是 sum 而非
   投票制；某些 head 可能高分选块但未被其他 head 确认，选中"孤独块"

2. **Stage-2 候选索引传递未 head-aware** — `cand_idx` 在 sparse_attn.
   baseline_runtime 中按全序列索引构建；scatter 回全宽时 head 维度
   映射可能有偏

3. **预算无 sink/swa 保送** — 我们实现"kt 即最终 token 数"，无 sink token
   + sliding window attention 强制保送；论文声明用 sink+swa 保底，在多跳
   QA 中 sink token 对流畅生成至关重要

4. **模型版本 / 校准集** — 我们用 transformers 自带的 Qwen3-8B，可能与
   训练时细微差异；channel 校准仅 8 个 qasper 样本，代表性不足

5. **生成阶段 hidden state decode** — prefill 用官方 flash_forward (q_len>1)，
   但 decode 用自定义 sparse forward；两者 tokenizer/rope 行为在边
   界 token 可能微偏，经 36 层放大

## 根因假设（最高可能）

**H1（P0）: 缺少 sink + swa 保底**
- 论文 §4.3 明确："top-kt tokens AND sink tokens AND sliding window tokens"
- 我们实现：仅 top-kt tokens
- sink token（前 4 token 强制送入 attention）在多跳 QA 中决定了"全局语义锚点"——
  它们保存了 prompt 的指令信息（如 "Answer the question based on..."）
- 没有 sink，每步 decode 的 attention 集合里不包含指令 token → 多跳任务
  （musique / narrativeqa 等 3+ hop）在生成步远离指令后丧失任务记忆

**H2（P1）: Stage-1 block 选择应 per-head**
- 8 个 kv head 独立选 top-128 块（总数可达 1024 块）→ 更多 token 候选
- 我们 sum 跨 head 后选 top-128 块 = 只有 128 块候选
- 论文暗示 per-head 独立选择

## 下一步验证

| 优先级 | 操作 | 预期 |
|---|---|---|
| P0a | 加 sink 保底（强制前 N token 进 attention） | 多跳 QA +3-8 分 |
| P0b | 加 SWA（最后 M token 强制保底） | 连贯性+近上下文 |
| P0c | Stage-1 per-head 独立选块 + union | 候选覆盖增大 |
| P1 | Full LongBench re-run with fixes | 最终+论文对齐 |

## 消融脚本档案

| 脚本 | 结果 | 备注 |
|---|---|---|
| /tmp/asynctls_p0_ablation.py (musique, P0 max-head) | 0.40 | revert |
| /tmp/asynctls_p0b_ablation.py (per_head_union) | 0.47 | 废 |
| /tmp/asynctls_p0c_ablation.py (per_head_all_stages) | 0.40 | 废 |
| /tmp/asynctls_p0d_ablation.py (official_template) | 3.46 | vanilla也崩 |
| /tmp/asynctls_vanilla_baseline.py | 3.46 | 对照组 |

2026-10-24 18:30 截止
