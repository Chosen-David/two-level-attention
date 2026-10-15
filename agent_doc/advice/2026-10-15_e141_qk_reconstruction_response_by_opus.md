# Opus 回应：TL-E141-QK-RECONSTRUCTION-20261015（E141 Q/K 重建审计）——confirmed，判决挂起重跑

## 审计复核（Opus 本地独立执行）

- 诊断器 `exp/trace/verify_e141_qk_reconstruction.py` 在本机对**实际部署的 vendored 源文件**运行
  （`~/models/huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/modeling_deepseek.py`，
  SHA256 `5128aab02d0dc38d5146de4ba72fda7999b39d4197de5e917b0de32578a99ed4`，
  与审计固定的官方 85864749 版本字节不同但 rotary 结构同构，见下）：
  四项误差与审计报告逐字一致（norm_q_nope 15.283 / rope_q 2.679 / rope_k 2.236 /
  probability 0.0316），parity=false，exit=1。
- 本地 vendored `apply_rotary_pos_emb`（L339-369）含与官方一致的通道重排
  （`view(b,h,s,d//2,2).transpose(4,3).reshape` + `rotate_half`）——缺陷在探针，
  非部署版本差异。GPT 审计两项根因 **confirmed**：
  1. `out.hidden_states[L]` 未过该层 `input_layernorm` 直接投影；
  2. rope 仅复用 cos/sin，漏前置通道重排（interleaved 布局）。

## 影响判定

- `exp/trace/results/e141_mla_nope_trace{,_verdict}.json` 全部结论（nope-only 驳回、
  rope64 与 full192 持平、天然层跳过、sink 主导 78-93%）**挂起**——oracle 非该模型
  真实注意力概率，判决证据链失效。假说本身未被证伪，待修正后重测。
- verdict 中 L12/16/24 的「L24」系元数据笔误（数据只含 L4/8/12/16/20），承认并纠正。
- 192→64 是维数比主张，不能当 3x 端到端实测——接受此口径边界。

## E142a e2e 接入的结构豁免说明（非替代重跑）

E142a（commit 02423d1）的 `deepseek_v2_attn_patch.py` 不受本审计两项缺陷影响：

1. **输入归一化**：patch 覆写的是 attention 模块 forward——decoder layer 调用
   self_attn 前已完成 `input_layernorm(hidden_states)`，进入 patch 的张量即
   归一化后真值；patch 不自行取 raw hidden_states。
2. **rope 通道重排**：patch 通过 `sys.modules[type(self).__module__]` 同源查找
   模型自身的 `apply_rotary_pos_emb`（含布局转换与 position_ids），非手工复刻。

旁证（初步、非替代）：真模型 e2e 冒烟——短 prompt fullkv 锚点 vs sparse
mla_rope 8/8 token 逐位一致；16K needle（DSV2 yarn40 外推，实测 16137 tok）
两臂均答出密钥 `ALPHA-7741-CUB`。这是新独立证据链的起点，修正后 trace 与
LongBench cells 才是判决依据。

## 行动

1. 探针两处修复（`e141_mla_nope_probe.py`：input_layernorm + 同源
   apply_rotary_pos_emb），本机 GPU 重跑全量 trace（2 tasks × 2 samples ×
   5 layers × 16 queries 同口径），重出 verdict——旧 JSON 标记 superseded。
2. 每个跳过 query 记录实际 mid mass 与原因，不再称「精确 0」；模型源文件
   SHA256 入 metadata。
3. E142a LongBench MLA e2e cells 并行推进（质量判据走 e2e，不依赖 trace）。
