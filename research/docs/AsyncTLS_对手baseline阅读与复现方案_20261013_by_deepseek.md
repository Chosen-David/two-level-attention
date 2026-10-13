# AsyncTLS 对手 baseline 阅读与复现方案

- 日期：2026-10-13（deepseek 轨，用户指令「这个论文也需要阅读并且作为对手」）
- 论文：**AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention**
  - 作者：Yuxuan Hu 等（人大 + 美团 Meituan）
  - arXiv:2604.07815v1, 2026-04-09
  - PDF 来源：`agent_doc/advice/other_paper_pdf/AsyncTLS.pdf`（11 页）
- 性质：**与 TLI 同构的直接竞品**——同是「块级粗筛 + token 细筛」两级稀疏注意力。

## 1. 方法核心（§4）

### 1.1 两级稀疏注意力（Two-Level Sparse Attention）
解码每步三段流水：

1. **Stage-1 粗粒度块选择（Block Selection）**：
   - 按 Quest 把 KV 切成 m=⌊n/B⌋ 块（B=64）。
   - 每块压缩表示：kmax_i,k = max_j∈Bi(kj,k)、kmin_i,k = min_j∈Bi(kj,k)。
   - 论文把 Quest 的逐元素 max/min 打分**重写为 GEMM 友好**（针对 MQA/MLA 共享 KV 的 Tensor Core 效率）：
     - qmax_k=max(qk,0)、qmin_k=min(qk,0)
     - s_i = Σ_h (qmax⊤_h·kmax_i + qmin⊤_h·kmin_i) = Σ_h(Qmax⊤Kmax + Qmin⊤Kmin)_h
   - 选 top-kb 块（论文默认 kb=128 块 ≈ 8192 token 粗筛候选）。

2. **Stage-2 细粒度 token 选择（Token Selection，Double-Sparsity 口径）**：
   - 校准（held-out Dcal）channel 重要性：s_i = (1/G)Σ_h [max_(q,k)∈Dcal |qh[i]| · max_(q,k)∈Dcal |k[i]|]。
   - 选 top-dc channel 组成代表 channel 集 C（**GQA 模型 dc=32**，MLA dc=128）。
   - 压缩 q/k 到 C 维：q̃(h)=q(h)[:,C]、k̃j=Quantize(kj[:,C])（**INT4 量化**压缩索引足迹）。
   - 近似分：ᾱ_j = (1/G)Σ_h softmax(q̃(h)·k̃⊤j / √d)（GQA 组内平均，与 TLI kv-head 共享口径同构）。
   - 在 Stage-1 选中的块 M_t 内 top-kt token：S_t = TopK_kt({ᾱ_j | j∈M_t})。
   - 最终：o(h)_t = softmax(q(h)K⊤_S_t / √d)·V_S_t（全维 KV 上算）。

3. **Stage-3 Sparse Attention**：全维 KV 对 S_t 计算。

### 1.2 异步 KV 卸载引擎（§4.2，AsyncTLS 第二贡献）
- 交错执行：当前步 token 选择用**上一步**的块选择结果（M_{t-1}），同时当前步算 M_t 并为下一步预取 → 隐藏 PCIe 传输延迟。
  - M_t = BlockSelect(q_t, K, V)；S_t = TokenSelect(q_t, M_{t-1})。
- 增量块传输：只传 M_t \ C_t 的新块（时域局部性 |Δ_t| ≪ kb）。

### 1.3 复杂度（每步，§4.3）
- O((n/B)·d) 块打分 + O(kb·B·|C|) token top-k + O(kt·d) sparse attn；总传输 O(|Δ_t|·B·d)。

## 2. 关键实验参数（§5，复现必须对齐）

| 参数 | 论文值 |
|---|---|
| block_size | 64 |
| top-kb 块数 | 128 块（粗筛 8192 token） |
| token budget kt | RULER 用 **512**；LongBench 用 **1024**；消融 512/1024/2048 |
| channel 子空间 dc | GQA 32 / MLA 128 |
| key 索引量化 | INT4 |
| 模型 | Qwen3-8B / Qwen3-14B / GLM-4.7-Flash |
| RULER 档位 | 32k（10 任务子集：S1/S2/MK1/MK2/MQ/MV/QA1/QA2/VT/FWE） |
| LongBench 档位 | 32k（14 任务） |
| 校准数据集 | held-out Dcal（Double-Sparsity 口径） |

## 3. 论文结果（Qwen3-8B 32k，与我们主表可比）

**LongBench**（token budget 1024，AVG13）：Full 50.14 > DS 50.10 ≈ AsyncTLS 49.90 > Quest 47.49。
**RULER**（token budget 512，AVG10）：AsyncTLS 90.54 ≈ Full 90.45 ≈ DS 90.44 ≫ Quest 38.92。
- 注意：论文 RULER 里 Quest 仅 38.92（块级无 token 细筛在 niah 系崩盘）；AsyncTLS 两级保住 token 精度。
- 效率：GQA kernel 1.7×–6.2× vs FA、1.2×–4.0× vs DS；MLA 3.3×–10.0× vs FA。端到端吞吐（含 offload）96k 时 1.84×（Qwen3-8B）/4.70×（GLM）。

## 4. 与 TLI 的同构/差异（我方方法对照）

| 维度 | AsyncTLS | TLI |
|---|---|---|
| 两级结构 | 块粗筛(Quest max/min) + 块内 token 细筛(DoubleSparsity channel+INT4) | far/near 分区各自 粗筛(块 max/min 或均值) + 细筛(子空间+4bit) |
| 分区 | 无 near/far 分区，全局 top-kb 块 | near/far 双池 + (α,β,γ) 配比 |
| 保送 | 未见 sink/swa 显式保送 | sink + sliding window 保送 |
| 预算口径 | kt = 512/1024/2048（细筛后参与 token） | 有效预算 1024（tia 128/1024/c4 口径） |
| 子空间 | channel 校准 top-dc（GQA 32 维） | 子空间选择（full/rope/nope/tail 等，E134 sub+full 矩阵） |
| 量化 | INT4 key 索引 | 4bit 细筛（q/k 量化） |
| 离线校准 | 需要（channel set C） | 需要（投影基/sim 等，视配置） |
| KV offload | 异步预取 + 增量传输（第二贡献） | 无（当前 TLI 不涉及 offload 维度） |

**差异化亮点**：TLI 的 near/far 分区配比、(α,β,γ)、sink/swa 保送、层跳过、kv-head 共享 vs per-q-head 消融是 AsyncTLS 没有的维度；AsyncTLS 的 GEMM 化块打分与异步卸载是 TLI 论文可讨论的对照点。

## 5. 复现策略（对手 baseline 接入）

**现状**：arXiv v1（2026-04），WebSearch/WebFetch 均未发现官方开源仓库（网络受限 + 论文太新）。按项目铁律「一律用官方开源源码」——先登记为**待官方开源后走 vendor**；若官方长期不开源，则退化为 **统一 harness 内语义复现**（与 MoBA 复现臂同法，标注清楚）。

复现拆解（无校准也能跑的部分 vs 需校准部分）：
1. **块选择（无需校准，动态）**：块 max/min 代表 + top-kb。TLI harness 已有近等价实现（far/near 的 minmax 粗筛）。可直接用 `--tli_far_method minmax` 类逻辑复现。
2. **token 细筛（需校准 channel C）**：Double-Sparsity channel 校准 + INT4。与 FASA Phase 1 校准管线相似（校准脚本 + convert），可复用 `exp/vendor_baselines/` 的校准脚手架。
3. **异步卸载引擎**：RULER 精度评估**不需要**（它是系统优化，不影响模型输出）——精度复现只需两级选择算法；吞吐对比另做 kernel 微基准（与 TLI 性能对比口径）。

**下一步（待资源）**：
- [ ] 检查 AsyncTLS 是否后续开源（`third_party/` 挂仓库）
- [ ] 无开源 → 在 `sparse_attn` 加 `asynctls` 复现臂（块 max/min + channel 校准 + INT4，budget 对齐 1024）
- [ ] RULER 32K/64K/128K + LongBench 接入对手汇总表

## 6. 引用
- 论文 PDF：`agent_doc/advice/other_paper_pdf/AsyncTLS.pdf`
- arXiv:2604.07815v1（2026-04-09）
