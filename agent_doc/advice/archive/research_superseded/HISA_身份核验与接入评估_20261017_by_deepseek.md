# HISA 身份核验与接入评估（vs TLI）

**日期**：2026-10-17　**执行轨**：deepseek 轨（#23 身份核验 + 接入评估）
**前置**：`exp/vendor_baselines/FOLLOWUP_20261015.md` 已登记身份线索；本档将其升级为正式核验结论 + 源码落地 + 接入判定。

## 1. 身份核验结论（唯一身份冻结）

| 维度 | 核验结果 |
|---|---|
| 题名 | **HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention** |
| 作者 | Yufei Xu, Fanxu Meng, Fan Jiang, Yuxuan Wang, Ruijie Zhou, Zhaohui Wang, Jiexi Wu, Zhixin Pan, Xiaojuan Tang, Wenjie Pei, Tongxuan Liu, Di Yin, Xing Sun, Muhan Zhang（14 人，Peking Univ. MuLab PKU 等） |
| 会议 | COLM 2026（2026-07-09 accepted） |
| 论文 | arXiv:2603.28458v4（CC-BY-4.0，PDF 已落地 `other_paper_pdf/HISA-2603.28458v4.pdf`，sha256 bb6c13f5…） |
| 官方入口 | MuLabPKU 实验室 TransArch 仓库：`github.com/MuLabPKU/TransArch`（HISA_COLM_2026/），pinned **dc9f3a3b** |
| GPU kernels | `tile-ai/tilelang` examples/dsa_hisa（MIT），pinned **7940102813** |
| Ascend kernels | `leavelet/hisa-ascend` |
| 端到端集成 | `xuyefei-a/sglang_hisa` branch `hisa_pr`（SGLang PR #24672） |
| 判定 | **身份已冻结**。不是 HiP/MoBA 等其他方法；「Hierarchical Indexing」= 两级块粗筛 + token 细筛的 indexer 替代 |

## 2. HISA 算法（官方 README + 论文口径）

Token 级稀疏注意力的 indexer 每 query 扫全历史 token（O(L²) per layer）。HISA 把它替换成两级层次过程：

1. **Block-level coarse filtering**：对 pooled 块代表向量打分 → 剪枝远区 → 选 top-m 块；
2. **Token-level refinement**：仅在幸存候选块内应用原始 indexer → 选 top-k token。

- 对 DSA/Sparse-MLA 的 indexer 是 **drop-in replacement**，下游 Sparse MLA 算子不变（保持相同 token-level top-k 稀疏模式）。
- 免训练；与原始 DSA 的 token 选择集合 **IoU > 99%**。
- kernel 级：32K 2×、128K 4× 加速。
- 已在 DeepSeek-V3.2 / GLM-5 上替换 indexer，Needle/LongBench 匹配原始 DSA 质量。
- 端到端需 SGLang v0.5.11（CUDA13）+ TileLang 0.1.9；启用参数 `--json-model-override-args '{"use_hisa":true,"hisa_k_block_size":128,"hisa_block_topk":64}'`。

## 3. 仓库形态判定

- **形态**：端到端 kernel（TileLang dsa_hisa prefill indexer）+ SGLang 集成。核心是**可提取的选择器**：块均值打分 → 块 top-k → token 细筛 → token top-k，输出 indices 供 Sparse MLA。
- 消费 **DSA indexer Q/K 与 head weights** → **不是 Qwen3/GQA attention 替代**，不能仅换模型名接 Qwen3-8B。
- 官方端到端硬件栈（SGLang v0.5.11 + CUDA13 + TileLang0.1.9 + DSA 模型）四台机均不具备。

## 4. 接入可行性评估（诚实口径，kernel 级已确认）

- **B 轨官方源码**：TileLang dsa_hisa kernels（MIT）已逐字落地（见 §5）→ 走 B 轨官方源码复用，Qwen3 适配层自写（同 JSSA/ClusterKV 路径）。
- **选择器语义（官方 kernel 口径，可提取性确认）**：
  1. **Stage-0 块均值池化**：K 按 `k_block_size` 分块 → `fp8_native_block_mean_pooling(k, k_scale)` 产每块 fp8 均值 + f32 scale `[Nb, D]`；块坐标翻译 `cu_ks//bs`（floor）、`ceil(cu_ke/bs)`。
  2. **Stage-1 块粗筛**：`Q·BlockedK` fp8×fp8 打分（`pool_mqa_fp8.py`，**ReLU + per-head weights[M,H] reduction**）→ `clean_and_maintain_logits` in-place：`idx==cu_k_s` 或 `idx==cu_k_e-1` 置 +inf（**强制保留首块=句首与末块=近期**）、越界置 -inf → `torch.topk(block_topk_eff, bf16)` 选块/query。
  3. **Stage-2 token 细筛**：`block_sparse_mqa` 只在选中块内原始 tokens 上全分辨率 MQA 打分（越界 -inf）→ `torch.topk(topk_tokens_eff)` 选 slot/query。
  4. **Stage-3 索引翻译**：slot → `block_id_in_topk×bs + offset_in_block` → 绝对 K 位置 → per-query 相对偏移（减 `cu_seqlen_ks[m]`）→ 越界置 -1（对齐 vLLM indexer 输出缓冲）。
  → 纯「块均值池化打分 + topk + 首末块保护」管线，**无 DSA 潜空间特定耦合**（仅输入是 fp8 Q/K + per-head weights），**可提取为 Qwen3 适配的选择器**。打分是**线性块代表点积**（非上界、非符号感知）。
- **Qwen3 适配要点**：Q 用 Qwen3 GQA query 组内投影（或每 KV head 共享），K 用 Qwen3 key；块均值池化 + 打分 + top-block + 块内细筛 + top-token 全部可 torch 等价；「统一后端机制轨」标签，不冒称 DSA 原生。
- **校准需求**：HISA 免训练、无校准数据依赖（结构先验）→ 比 JSSA（数据驱动谱基）轻，比 AsyncTLS（channel 校准）轻。
- **与 TLI 关系**：HISA 属「粗筛细筛轴」（与 Quest/AsyncTLS 同轴）。其 coarse 打分=块均值池化代表点积（线性，无上界），与 Quest minmax 上界、AsyncTLS 符号感知 GEMM 不同；首/末块 +inf 保护 ≈ TLI sink/swa 强制区（HISA 是块级 keep-first/last，TLI 是 token 级 sink+swa）。接入后可与 Quest/AsyncTLS 横比块粗筛质量-成本。
- **建议**：身份冻结 + 官方源码（TransArch pointer + TileLang kernels）已落地。真正 Qwen3 适配与跑批**排 AsyncTLS 派单之后**（当前 GPU 全被 AsyncTLS 与 Opus e146ca 占满）；由用户确认是否作主表臂后开工。

## 5. 本轮落地清单

- `third_party/TransArch/`（pinned dc9f3a3b，HISA_COLM_2026/README.md + 根 README，UPSTREAM.json 登记）
- `third_party/tilelang_dsa_hisa/`（pinned 7940102813，7 文件逐字 + LICENSE，UPSTREAM.json 登记）
- 本档：`research/docs/HISA_身份核验与接入评估_20261017_by_deepseek.md`
- `agent_doc/guide/paper_figs/evaluation/evaluation.md` HISA 行已更新（身份核验完成）
- 待办：Qwen3 适配层（`--method hisa_vendor`，参数 --hisa_budget 对齐 K2=1024）→ smoke → 派单（AsyncTLS 派单完成后 + 用户确认作主表臂）
