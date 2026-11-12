# TL-GLM-FULLKV-FI-LBV1-20261112

## 目标与口径

- 用户要求继续测 GLM-4.7-Flash 的 FullKV × LongBench v1 英文 16 数据集，并核实、使用 FlashInfer；测量只用非本机 GPU。
- 既有 `native_full` 不是 FlashInfer，而是 HF SDPA prefill/decode + `DynamicCache`。本任务新增独立 FullKV runner，保持 SDPA prefill 和 GLM RoPE 数学，decode 将同一 post-RoPE K/V 写入 page=64 的固定 paged cache，再调用 FlashInfer。
- 默认跑全部 16 任务、每任务全量，`seed=42`、`context_cap=31500`，目标闭集 3750 请求；`trec/triviaqa/samsum/lcc/repobench-p` 不走 chat template，`samsum` 使用 EOS+newline 且 `min_new_tokens=1`。
- 本任务不改变 NFA/frozen-region 路径。结果在独立验收前保持 `pending-independent-verification`；请求级 `wall_seconds` 不冒充 clean TPOT 或全数据集性能结论。

## 实现

- `sparse_attn/patches/glm4_moe_lite_flashinfer_patch.py`：GLM 展开后的 20 Q heads、20 KV heads、QK/V head_dim=256 的 FullKV FlashInfer prefill-append/decode 路径；每请求独立 runtime，结束后显式解绑。
- `sparse_attn/patches/qwen3_flashinfer_patch.py`：公共 paged runtime 接受显式 `head_dim`，兼容无 `sliding_window` 属性的模块；mask metadata 同时支持旧版 `get_mask_sizes(cache_position, layer_idx)` 与 Transformers 5.12 的整数 `q_length`，新增 `get_seq_length()` 并在每个 decode step 更新已见 token 数。
- `benchmark/LongBench/pred_glm_fullkv.py`：SDPA/FlashInfer 双臂、16 任务、确定性互斥 rank 分片、断点续跑、输入/源码/模型/数据集身份、逐请求评分和后端调用计数。
- `tests/test_glm_fullkv_flashinfer.py`：mask 新旧签名、GLM 256 维几何传递和非法 V 维度拒绝。

## Progress

### 2026-11-12：真实 GPU smoke 已完成，全量待派发

环境：pc02（10.238.122.187）非本机 H20-3e，Transformers 5.12.1、FlashInfer 0.5.2。隔离 worktree 基准提交为 `84bd19ee947c2498db7c988a9fa2ca766caf1267`，叠加本轮三份源文件；该基准号只描述 smoke 载体，不是本轮代码最终提交身份。

1. 静态编译通过；基础 Python 缺少 GLM 模块，复用旧 GLM 生产脚本的 `pylibs_tf512` 后 import 通过。
2. 首次真实 prefill 暴露 Transformers 5.12 需要 `get_seq_length()` 且 `get_mask_sizes` 第一参数改为整数；按实际 mask-builder 契约修复后重跑。
3. `trec:152`、1024 tokens、64 generated tokens：47 层 prefill append、2961 次 decode append 和 2961 次 wrapper 调用断言通过。SDPA/FI 前 12 个 token 的轨迹在 index 12 首次分歧；固定历史 teacher-forced 对拍中 prefill logits 逐位一致，64 个位置只有该点 argmax 改变，多个候选 BF16 logit 同为 134，属于数值并列下的贪心分歧，不能宣称逐 token bit-exact。
4. `narrativeqa:152`、23183 tokens、chat suffix=2：SDPA/FI 均生成 `[36,23470,7919,154827]`，预测均为 `Eatonville`；FI 的 47 次 prefill append、141 次 decode append/wrapper 调用正确。
5. `gov_report:152`、7186 tokens：SDPA 生成 434 tokens、得分 0.36325678、26.294s；FI 生成 411 tokens、得分 0.36440677、25.992s，index 56 首次分歧；FI 47 次 prefill append、19270 次 decode append/wrapper 调用正确。该单样本耗时相近，不外推为加速结论。
6. 每次进程退出后 pc02 GPU 显存均释放。原始 smoke 回执位于 pc02 `~/ak_verify_results/glm_fullkv_fi_smoke_20261112/`；尚未并入正式全量结果池。

下一步：运行聚焦测试，提交并推送本轮代码；按固定新 commit 部署非本机 GPU。pc02 历史实测每 GLM worker RSS 约 64GB，正式派单不得仅按四卡数量启动四个高内存 worker；点火后 30 秒内核对进程、显存和日志，最终审计 3750 请求、输入身份、源码身份、后端计数和逐样本分数。
