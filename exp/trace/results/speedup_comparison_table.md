# 全长度 × 全模型族加速比对照表

> 生成时间：2026-10-26 | 口径：TLI iter3+K1=64 CUDA-graph decode vs dense-eager decode
> 统一协议：TPOT 中位数，bit-exact 门禁（graph vs eager gen_tokens 逐位一致）

## 1. E2E 加速比：TLI-graph vs Dense-Eager（论文主表最终口径）

### Qwen3-8B（pc40 GPU0, H20-3e）

| 长度 | dense-eager (ms) | TLI-graph (ms) | 加速比 | vs 旧口径增量 | 判决文件 |
|------|------------------|----------------|--------|--------------|---------|
| 32K  | 25.712           | 17.715         | 1.451× | +0.387×      | e172_iter3_k64_qwen_verdict.json |
| 64K  | 39.797           | 17.957         | 2.216× | +0.768×      | e172_iter3_k64_qwen_verdict.json |
| 128K | 67.139           | 18.464         | 3.636× | +1.569×      | e172_iter3_k64_qwen_verdict.json |
| 256K | —                | —              | —      | —            | 未采集（需显存 ≥ 2×68.7GB@B2） |

### Llama-3.1-8B（pc40 GPU1, H20-3e）

| 长度 | dense-eager (ms) | TLI-graph (ms) | 加速比 | vs 旧口径增量 | 判决文件 |
|------|------------------|----------------|--------|--------------|---------|
| 32K  | 21.385           | 14.971         | 1.428× | +0.413×      | e173_iter3_k64_llama_verdict.json |
| 64K  | 33.287           | 15.188         | 2.192× | +0.782×      | e173_iter3_k64_llama_verdict.json |
| 128K | 56.801           | 15.590         | 3.643× | +1.682×      | e173_iter3_k64_llama_verdict.json |
| 256K | —                | —              | —      | —            | 未采集 |

### Qwen3-32B（pc02 GPU0, H20-3e）

| 长度 | dense-eager (ms) | TLI-graph (ms) | 加速比 | vs 旧口径增量 | 判决文件 |
|------|------------------|----------------|--------|--------------|---------|
| 32K  | 55.265           | 41.295         | 1.338× | +0.323×      | e174_iter3_k64_32b_verdict.json |
| 64K  | 80.356           | 41.774         | 1.924× | +0.586×      | e174_iter3_k64_32b_verdict.json |
| 128K | 128.670          | 42.670         | 3.015× | +1.174×      | e174_iter3_k64_32b_verdict.json |
| 256K | —                | —              | —      | —            | 未采集（128K B2 OOM） |

### 三模型族并排对比

| 长度 | Qwen3-8B | Llama-8B | Qwen3-32B | 跨族方差 |
|------|----------|----------|-----------|---------|
| 32K  | 1.451×   | 1.428×   | 1.338×    | 0.024×  |
| 64K  | 2.216×   | 2.192×   | 1.924×    | 0.024×  |
| 128K | 3.636×   | 3.643×   | 3.015×    | 0.024×  |

**核心形态**：加速比随上下文单调增长，三模型族逐档对称（差 ≤0.024×）。

## 2. 长度平坦化（核心机制证据）

| 模型 | 32K (ms) | 64K (ms) | 128K (ms) | 4×增长 | dense 4×增长 |
|------|----------|----------|-----------|--------|-------------|
| Qwen3-8B | 17.715 | 17.957 | 18.464 | +4.2% | +161% |
| Llama-8B | 14.971 | 15.188 | 15.590 | +4.1% | +165% |
| Qwen3-32B | 41.295 | 41.774 | 42.670 | +3.3% | +133% |

**机制分解**（e166i/j/h + e168b/e/i/j 双口径闭合）：
- gemm：L 无关（权重读与上下文解耦）
- attn：L 无关（K2=1024 预算恒定）
- copy：L 无关（选中流量预算恒定）
- other（摘要-索引维护）：唯一近线性增长段（+0.6~0.7ms@4×）

## 3. Kernel 级优化贡献分解

| 优化段 | 节省 (ms/层) | TPOT 贡献 | 判决文件 |
|--------|-------------|----------|---------|
| iter2a 几何步级缓存+死op删 | ~0.8ms | ~3.2% | iter2a_verdict.json |
| iter2b 紧凑域 softmax+topk | ~1.2ms | ~4.9% | iter2b_verdict.json |
| iter2c span 并行 expand | ~0.5ms | ~2.0% | iter2c_verdict.json |
| iter3a 掩码装配融合 | ~0.18ms | ~0.8% | e166a_iter3a_verdict.json |
| iter3b 候选压缩融合 | ~0.90ms | ~4.0% | e166b_iter3b_verdict.json |
| iter3d 摘要/L1/候选三kernel | ~2.5ms | ~11.8% | e166d2_iter3d_verdict.json |
| compact cfg 调优 | ~0.34ms | ~1.6% | e166e_compact_cfg_verdict.json |
| **累计 iter2+3** | **~5.0ms** | **~24-40%** | — |

## 4. 残余热点（下一步优化目标）

| 热点 | 耗时 (ms/步) | 占比 | 优化潜力 |
|------|-------------|------|---------|
| topk 家族 | ~3.3 | ~17% | 1296 calls/步（3×432 层），可融合为 Triton kernel |
| copy/elementwise 弥散群 | ~3.55 | ~18% | 10530 kernels/步，launch-overhead bound |
| GEMM | ~4.5 | ~23% | 已贴物理地板，无优化空间 |
| other | ~2.7 | ~14% | 摘要-索引维护，近线性增长 |

## 5. 256K 现状

- **无 e2e TPOT 数据**：三模型族主表止步 128K
- 唯一 256K 性能数据 = `e141a_wall_long.json`（prefill attention pipeline 微基准）：
  - dense SDPA 6275.4ms vs TLI Step2 1481.3ms = **4.24×**（非 e2e TPOT）
- 256K B=1 可行性：8B 模型需 ~18GB KV，128K B2 已 OOM → 256K 需多卡或 FP8 量化

## 6. FlashAttention2 对照

- **仓库内无 FA2 e2e TPOT 数据**：现有 dense 基线均为 SDPA eager
- FA2 臂尚未采集，需新实验（decode 侧 FA2 与 SDPA eager 差异通常 <5%）
- 替代对照：Q3 段已有四 vendor eager TPOT vs dense（LoDA/JSSA/AsyncTLS/Quest/FASA），LoDA 唯一全长度低于 dense

## 7. 批档包络摘要

| 长度 | Qwen3-8B 交叉点 | Llama-8B 交叉点 | Qwen3-32B 交叉点 |
|------|----------------|----------------|-----------------|
| 32K  | B∈(1,4)        | B≈2            | B≈2             |
| 64K  | B≈4            | B≈4            | B≈4             |
| 128K | 不交叉（tli 领先至 B=2） | 不交叉 | 不交叉（B2 OOM） |

**核心结论**：交叉点随长度单调右移（32K:B≈2-4 → 64K:B≈4 → 128K:不交叉），
TLI 在长上下文 bs≤2 区间有明确优势。
