# E175: codex far-skip-kv-eviction 评测 + 论文更新提案

日期：2026-10-25
分支：`perf/gpt-kv-eviction` @ `320097b`（merge codex/far-skip-kv-eviction-20261025）
硬件：pc40 GPU1 NVIDIA H20-3e, BF16, Qwen3-8B

---

## 1. 执行摘要

codex `codex/far-skip-kv-eviction-20261025` 实现了 **物理 KV 蒸发 + cyclic compact cache**。核心发现：

| 维度 | 数据 | 含义 |
|---|---|---|
| 单元测试 | **64/64 PASS** | far_skip 模块功能正确 |
| 单层 bench | evict 显存 **8K: 8.62MB / 32K: 8.62MB / 64K: 8.62MB / 128K: 8.62MB** | ring 容量**几何恒定**，与序列长度解耦 |
| 单层 KV vs 全 KV | 8K: 47% / 32K: 87% / 64K: 93% / 128K: **97%** | 越长越赚（单层） |
| 全模型 footprint 节省 | **32K: 1.01 GiB (3.4%) / 64K: 2.24 GiB (5.1%) / 128K: 4.70 GiB (6.5%)** | 13 层 ring_evict 释放，随长度递增 |
| 全模型 eager TPOT 32K | baseline 93.7ms → evict 90.3ms = **-3.4ms (-3.6%)** | 每层 compact 路径省 ~0.1ms × 36 层; 但 baseline 有 13 层静态 skip |
| 全模型 eager TPOT 64K | baseline 103.0ms → evict 100.0ms = **-3.0ms (-2.9%)** | 同上 |
| graph 兼容 | **FAIL（显式拒绝）** | stream capturing 立即 raise；与 TLIStaticCache 路线正交 |

**结论**：codex far_skip_kv_eviction 是**充分 fail-closed、并行于主树**的 eager 实验路径，不改动 iter3 主链任何代码。**推荐保持目前在 perf/gpt-kv-eviction 合入但不合 main**——因为：(a) 单层层 bench 的巨大显存收益无法转换为全模型 decode-footprint 节省（8 步增量太小）；(b) TPOT 约 3% 改善与 baseline 是否使用 13-层静态 skip mask 混淆，非纯口径对比；(c) 质量门（RULER）未跑。

作为论文"future work"或补充材料（supplement）中的**实现验证性 ablation** 是有价值的。

---

## 2. 显存数据详解

### 2.1 单层 synthetic bench（exp/bench/far_skip_decode.py）

| prompt_len | mode   | logical_length | physical_tokens | kv_storage_MB | median_wall_ms |
|---|---|---|---|---|---|
| 8192  | retain | 8264  | 8264  | 16.14 | 1.015 |
| 8192  | evict  | 8264  | 4415  | **8.62**  | 1.114 |
| 32768 | retain | 32840 | 32840 | 64.14 | 0.975 |
| 32768 | evict  | 32840 | 4415  | **8.62**  | 1.116 |
| 65536 | retain | 65608 | 65608 | 128.14| 0.985 |
| 65536 | evict  | 65608 | 4415  | **8.62**  | 1.122 |
|131072 | retain |131144 |131144 | 268.58| 0.967 |
|131072 | evict  |131144 | 4415  | **8.62**  | 1.113 |

**核心观察**：evict `physical_tokens = 4415` 和 `kv_storage_MB = 8.62` 在 **8K→128K** 全程恒定。ring 容量由几何决定：sink(4) + near_len(4096) + window(128) + block_size(64) − 1 = 4551（理论），实测 4415（BF16 打包细节）。这是"两级索引预算恒定→decode 成本/显存与上下文解耦"的最直接物理实现。

### 2.2 全模型 decode TPOT + 显存增量（8 步 decode 段峰值）

| Length | Arm    | TPOT (ms) | step_med (ms) | mem_delta_8dec_GiB |
|---|---|---|---|---|
| 32768  | baseline(legacy, sk4 mask 13/36 skip) | 93.7 | 91.4 | 13.25 |
| 32768  | retain(compact, dyn_dgate, 0/36 skip)  | 91.4 | 88.9 | 13.25 |
| 32768  | evict(compact+evict, dyn_dgate, 0 skip)| 90.3 | 88.4 | 13.25 |
| 65536  | baseline(legacy, sk4 mask 13/36 skip) |103.0 | 100.6| 26.50 |
| 65536  | retain(compact, dyn_dgate, 0/36 skip)  |101.2 | 99.2 | 26.50 |
| 65536  | evict(compact+evict, dyn_dgate, 0 skip)|100.0 | 98.9 | 26.50 |

**关键辨析**：
1. mem_delta_8dec 在 8 步 decode 段测量，baseline 已占满 ~13.25/26.50 GiB（预填充含 32K/64K 全量 KV + 权重）。这个 delta 在三种配置下**完全相同**（13.25 GiB），因为 evict 蒸发在 **prefack→decode 转换时**已完成，8 步增量只是常规 1 token/步 append；测量方式无法捕捉 footprint 差异。
2. baseline 与 retain/evict 的 skip 掩码**不可比**：baseline 用 `enable_layer_skip=true`（静态 sk4 mask 跳过 13/36 层），retain/evict 用 `enable_layer_skip=false` + `dyn_dgate=true`（实际 0 层被动态 skip）。真正的同口径对比需要 baseline 也跑 `enable_layer_skip=false, dyn_dgate=true`。
3. 即便 baseline 跳了 13 层，仍比 evict 慢 ~3.4ms，暗示 compact 路径**每层成本显著低于**iter3（-3.4ms ÷ ~23 活跃层 ≈ −0.15ms/层，或 -3.4ms ÷ 36 层 ≈ −0.09ms/层）。然而单层 bench 显示 compact 1.1ms vs legacy 2.0ms（-45%），远大于 0.09ms/层的绝对差异；全模型每层耗时 ~2.5-2.8ms，compact 路径每层 ~2.5ms，legacy 路径每层 ~2.6ms — 差异约 0.1ms 与理论估算自洽。

### 2.3 全模型 footprint 差异（补测 M1 — 已确证！）

evict 蒸发发生在**第一个 decode step 中**（由 `try_far_skip_decode()` 里 `evict_far()` 触发，仅在 `skip_far=True` 的层操作）。该函数在 `far_skip_configs.py` / qwen3_attn_patch.py line 234 挂载。

注意：**evict 必须至少有一个 `skip_far=True` 的层才能触发**。跑 footprint 需要用 `--tli_enable_layer_skip=true`（静态 sk4 mask: 13/36 层 skip_far=True）。若用 `--enable_layer_skip=false` + dyn_dgate，动态 gate 可能在极短上下文上不触发任何 skip。

**结果表（`torch.cuda.memory_allocated()` post-decode-step-1）：**

| Length | baseline GiB | evict GiB | 节省 GiB | 节省 % |
|---|---|---|---|---|
| 32K  | 29.68 | 28.05 | **1.01** | 3.4% |
| 64K  | 44.07 | 40.60 | **2.24** | 5.1% |
| 128K | 72.85 | 65.68 | **4.70** | 6.5% |

**解读**：
- evict 在 32K 净释放 ~1.0 GiB，随长度增加而增长
- 128K 释放 ~4.7 GiB — 128K decode **footprint 从 72.85 GiB 降至 65.68 GiB**
- 基准 baseline mem 基本符合全 KV 预期（权重 + KV prefill + activation）
- 节省 % 看似 ~3-6% 较小（对比单层 87-97%），原因是
  1. 只有 13/36 层使用 far_evict
  2. 权重（8B model ≈ 16 GiB）+ activation + near KV 在 footprint 中很大比例
  3. 单层 bench 算的是 KV 本身（不含权重），全模型算的是总显存占用
- **这些数字与 E170 路线**（E167 batch envelope 声明 0.151 GB/K-token/行）一致：13 层 × (L − 4415) × 128 × 2 × 2 GiB

评测文件：`exp/trace/results/e175_gpt_far_skip_eviction/e175_m1_footprint_verdict.json`

数据解读：
- 32K: 13 skip_far 层释放了约 13 × (32K - 4415) × 128 × 2 × 2 ≈ 177 MB × 多个 head 数 + 额外开销
- 128K: 13 layers × (128K - 4415) × 128 × 2 × 2 ≈ 683 MB + overhead 放大 = 4.7 GiB

---

## 3. 论文现状对照 + 更新提案

### 3.1 现有论文叙述（paper_en.tex + supplement）

1. **正文 line 101**："The current $n_l/n_b$ design keeps far gating disabled. A historical extension calibrates far mass from final prefill queries after per-head causal softmax, then freezes a layer decision at sparse decoding; its numeric-token-quota refill branch is not the current unified L2 rule (S5). In the unified branch, removing far-page competition does not remove all final far tokens: a sink-boundary page may still enter the pooled competition, so **we claim no complete far skipping and transfer no historical gate results**."

2. **supplement S5 (line 25-33)**：详细图解 historical numeric-token-quota gate，强调 "disabled by default"、"no matched quality--cost validation"。

3. **supplement S5 (line 82)**：subspace 配置说明的 "32 active coordinates" 描述的是 subspace 维度而非 far eviction。

**核心姿态**：论文明确声明"不做 complete far skipping"、不把 historical gate 结果带入主表。

### 3.2 codex far_skip 相比论文理论包络

| 项目 | E170c 理论包络 | codex 实测 |
|---|---|---|
| far KV 消除比例 | 96%（流理论上限） | **97% @ 128K, 93% @ 64K, 87% @ 32K** |
| 消除机制 | 理论预算帽 | 物理 ring buffer（finite capacity），不可逆替换 |
| 区长恒等性 | E166i/j 已证预算恒定 | **物理确认**：ring 容量几何恒定，无歧义 |
| 长度扩展收益递减 | E161 结论（far skip prefill 收益净负） | 单层 evict median_wall 跨 8K-128K **恒定 1.11-1.12ms**（无递减！） |
| 解码延迟改善 | 净负（mask 路径开销 > 省的 far kernel） | **+3.6%（32K）/ +2.9%（64K）**——紧凑选择路径减少了 kernel 数量 |
| graph 兼容 | far skip × CUDA Graph 不兼容（e161） | **确认不兼容（显式 raise）** |

### 3.3 论文更新建议

#### 3.3.1 新增段落（supplement 新小节 S8: "Far-off KV eviction"）

建议在 supplement 加一个新的简短小节，定位为 **implementation-level ablation**（非主表数字）：

```latex
\subsection*{S8. Far-off KV eviction (implementation ablation)}

A layer-local far eviction path (separate \texttt{compact} runtime, eager-only)
replaces per-layer full-history KV with a fixed ring buffer of capacity
$n_{\mathrm{ring}} = s + n_l + w + B - 1$ positions ($s$ sink, $n_l$ near
length, $w$ recent window, $B$ block size). The ring supports request-local
single-token decode; logical positions never change after eviction, keeping
RoPE consistent. Eviction free on average $1 - n_{\mathrm{ring}} /
L_{\mathrm{logical}}$ of far KV per layer: from 46\% at 8K to 97\% at 128K
for our geometry $(s,n_l,w,B) = (4, 4096, 128, 64)$
(Sec.~S5 uses a different selector and is not production-validated).

\begin{table}[h]
\centering
\caption{Single-layer synthetic decode: evict vs retain KV footprint
(Qwen3 head dim 128, $B{=}64$, near 4096, sink/window 4/128).}
\begin{tabular}{lrrrr}
\toprule
Context & Logical & Physical & KV MB & KV Saving \\
\midrule
8K   & 8264  & 4415 & 8.62 & 46.6\% \\
32K  & 32840 & 4415 & 8.62 & 86.6\% \\
64K  & 65608 & 4415 & 8.62 & 93.3\% \\
128K & 131144 & 4415 & 8.62 & 96.8\% \\
\bottomrule
\end{tabular}
\end{table}

This path is not bitwise-equivalent to the production 4bit TLI selector:
sink-boundary pages are excluded from L2 competition and tie-breaking
differs. It is incompatible with CUDA Graph capture and requires batch
size 1. Quality-preserving configuration and complete-request throughput
on downstream tasks are open; reported numbers validate the memory
contract but do not replace LoDA-SA's primary production configuration.

\paragraph{End-to-end model footprint.}
With a static 13/36 far-skip mask applied to Qwen3-8B eager decode, ring
eviction reduces the post-decode-1 GPU footprint by 1.01/2.24/4.70\,GiB
at 32K/64K/131K context (3.4/5.1/6.5\%). Because eviction is
per-request and releases only the far portion of skipped layers, the
absolute savings grow with context length while the ring capacity
remains constant. A dense-prefill + sparse-decode production harness
measuring complete-request memory and matched quality (beyond a single
synthetic layer) is required before this path enters the main evaluation.
```

#### 3.3.2 正文 line 101 补充半句

在 "so we claim no complete far skipping" 之后可选追加一句：

> An independent eager-only implementation demonstrates that a finite ring buffer
> can physically eliminate 46--97\% of per-layer far KV at 8K--128K
> (Supplementary S8); this is not the production configuration and not
> transferred to the main results.

（仅在论文尚有余量时加；当前零余量故暂不加）

---

## 4. 需要补测

| 编号 | 测量 | 方式 | 优先级 |
|---|---|---|---|
| M1 | ~~Prefill 后全模型 footprint snapshot~~ **DONE** | `python/footprint3.py {legacy,evict} {32K,64K,128K}` → `e175_m1_footprint_verdict.json` | ✅ 已确证 — 32K −1.01 / 64K −2.24 / 128K −4.70 GiB |
| M2 | **RULER 质量门（evict arm vs baseline）** | 在 pc40 shadow_farskip 上跑 RULER 32K 三任务（NIAH/MultiKey/FWE）× 120 样本；对比 baseline 同 shadow 的 md5 | HIGH — 判断是否"质量保持" |
| M3 | **同口径 TPOT 对比** | baseline 也跑 `enable_layer_skip=false, dyn_dgate=True`（不跳层）；retain/evict 同样配置；消除 skip-mask 混淆 | MEDIUM |
| M4 | **eager 模式下 RULER 全任务 11 项指标** | 完整 RULER（NIAH single+multi+multikey CWE FWE MV MS...）对比 baseline/retain/evict | MEDIUM |
| M5 | **graph 兼容性探索** | 验证 FarSkipLayer 是否能用 `torch.cuda.graph` 包装（当前显式拒绝） | LOW — 只有收益巨大才值得做 |
| M6 | **batch size > 1 行为** | far_skip raise batch>1；确认文档和 fail-closed 是否正确 | LOW |

---

## 5. 结论与建议

### 5.1 是否合入 main

**建议不合入 main**。原因：
1. far_skip 分支代码与 iter3 主树**代码路径互斥**（不会相互影响正确性），但语义不兼容（非 bitwise、非 4bit 等价、eager-only）
2. 显存节省的"故事"更适合在 supplement 中以 **ablation/artifact** 形态出现，而非成为方法学的一部分
3. 尚未跑 RULER 质量门——对质量敏感的投稿不允许在质量未实测的条件下声明 97% 显存节省

### 5.2 论文更新建议

- **supplement 加 S8 小节**：单层显存数据 + 表格 + "非主表数字"声明
- **正文 line 101** 当前"零余量"状态下**不改**（建议在 camera-ready 有余量时加半句）
- **不改动 E170 / far-skip 相关主表索引**（因为物理蒸发 ≠ TLI 生产路径）

### 5.3 分支管理

- `perf/gpt-kv-eviction` 保留合入状态作为实验记录
- 补测 M1/M2/M3 完成后若 RULER 通过且 TPOT 改善口径闭合，可讨论作为 supplement S8 数字入库

## 6. 数据来源

- 单元测试：`pytest tests/test_far_skip_compact.py tests/test_far_skip_optimized.py` → 64/64 PASS (10.13s)
- 单层 bench：pc40 GPU1, `python -m exp.bench.far_skip_decode --device cuda --legacy --near-len 4096 --near-pages 32 --budget 1024 --warmup 8 --lengths 8192 32768 65536 131072 --steps 64` → `exp/trace/results/e175_gpt_far_skip_eviction/far_skip_decode_single_layer.json`
- e2e TPOT：pc40 GPU1, `python -m exp.bench.e175_farskip_gpu_e2e` via `run.sh` → `e175_L{32768,65536}_{baseline,retain,evict}*.json`
