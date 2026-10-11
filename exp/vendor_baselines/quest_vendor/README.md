# quest_vendor — Quest 稀疏 attention 的 Qwen3 适配层（B 轨）

## 来源与 Provenance
- 论文：*Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*
  (Tang et al., ICML 2024)。
- 官方仓库：`third_party/quest/`（snapshot `quest@01c1623b`，git 主仓子目录，
  仅保留源码不嵌子模块）。
- 本目录为 **clean-room 独立实现**：不 import 官方代码，按论文算法
  （sign 符号匹配 + chunk 级 max-pool score + topk chunk 选择 + 因果 tril）
  重新实现，对齐 Qwen3 (transformers 4.5x) attention 接口。

## 文件
- `qwen3_quest_patch.py` — 适配层：`enable_quest(model, token_budget, chunk_size)`
  patch 全部 Qwen3Attention；prefill/浅层走 dense，decode 走 Quest 稀疏选择。
- `test_qwen3_quest_patch.py` — CPU 单元测试（5 项，全绿）。

## 超参与口径（对齐 Quest 论文）
- `token_budget=1024`（等效 token 预算，与 TLI 的 K2=1024 同口径）
- `chunk_size=16`
- 前 2 层 dense（`layer_id < 2`，对齐论文策略）

## 已知边界
- 仅支持 Qwen3（GQA）。MLA（DeepSeek）不适配 → 注记。
- kernel 线（libraft）不接入，精度走纯 PyTorch；性能引用官方论文 + 平台差注记。
- decode 稀疏选择在 GPU 上为 eager 矩阵路径（未用 flash-attn），效率副表另行评估。
