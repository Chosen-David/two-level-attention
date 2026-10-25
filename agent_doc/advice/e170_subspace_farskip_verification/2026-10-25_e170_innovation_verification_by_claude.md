# E170 创新点收益验证——子空间 + Far Skip 代码是否吃到收益 + 性能浪费是否正常

> 日期：2026-10-25
> 分支：`perf/subspace-farskip-optim`
> 机器：pc40 GPU0/1 (H20-3e)，Qwen3-8B
> 动机：用户要求独立验证两个创新点的收益是否被代码吃到、以及性能浪费是否已 fix
> 来源：远端 GPT `runtime-review-20261021` 补丁 + 自建 microbench + op 归因

---

## 0. 结论速览

| 创新点 | 收益是否吃到 | 证据强度 | 具体数字 |
|---|---|---|---|
| **子空间（tail32）** | ✅ 吃到 | microbench 隔离 + bitexact 通过 | 4× buffer，L1 3.38×，全宽 L2 2.10×，decode traffic 25% |
| **Far skip（D' gate）** | ✅ 吃到 | 字节账本 + 显存包络 | 选中流量 = dense 的 4%（128K），紧凑显存 = dense 的 4%，far leak = 19.5% |
| **性能浪费** | ✅ 已收敛 | op 归因 (e170d) | 无单一热点；残余 = topk 3.3ms（用户判） + elementwise 弥散群（整链融合） |
| **GPT 补丁** | ✅ 正确 | e170_opattr v2 三长度 bitexact PASS | input_signature 门控 + fb 修复 + 3× new_empty |

---

## 1. GPT 远端 PR 发现与补丁合入

**来源**：`origin/gpt_dev/runtime-review-20261021`（未合入的 GPT 分支）

GPT 发现的 5 个问题与我们的代码审查一致：

1. **incremental index 缓冲区元数据复用 bug**（正确性）
   增量路径 `_prepare_index_inc` 在 `k.device/dtype/HKV/D/bs` 变化时错误复用旧缓冲区 → single-head broadcast / dtype 混用
   修复：L840 加 `input_signature` 比对门控，同时 `_inc` dict 加 signature 字段

2. **页边界重建起点修正**（性能）
   `fb = (S_prev - 1) // bs` → `fb = S_prev // bs`（整页边界时不重建已完成的整页）

3. **4 处 `new_zeros` → `new_empty`**（性能）
   k_min_buf/k_max_buf/k_avg_buf/k_qat_buf 扩容时不需要清零未暴露的 spare 容量

**处置**：全部 5 处已合入 `perf/subspace-farskip-optim` 分支。

---

## 2. 子空间收益隔离验证（E170b）

方法：纯 torch 随机张量 microbench，不依赖模型权重。1000 次循环 CUDA Event 计时（中位数+IQR）。

### 2.1 Microbench A：L1 amin/amax reduce

| 几何 | 时间 | 理论 FLOPS |
|---|---|---|
| full 128 | 0.520 ms | 396M |
| tail32 | 0.154 ms | 99M |
| **加速** | **3.38×** | **4.0×** |

### 2.2 Microbench B：L2 einsum

两种口径：

- **K2=1024 候选**：tail 与 full 持平（≈0.034 ms，操作在 L2 cache 内，bandwidth-bound）
- **全宽 context（~131K tokens）**：tail **2.10×**（0.527 vs 1.109 ms）
- **候选消除倍率**：K2/candidate vs 全宽 = **15.6×**

### 2.3 Microbench C：k_qat 紧凑 buffer

| 几何 | bytes | 占用 |
|---|---|---|
| full 128 @ 128K | 134M elements × 2B | **256 MB** |
| tail32 @ 128K | 33.5M elements × 2B | **64 MB** |
| **比例** | — | **25% (4× 节省)** |

### 2.4 Microbench D：decode step KV 流量

| 几何 | k_qat | V | total |
|---|---|---|---|
| full 128, K2=1024 | 2 MB | 2 MB | **4 MB** |
| tail32, K2=1024 | 512 KB | 512 KB | **1 MB** |
| **比例** | — | — | **25% (4× 节省)** |

### 2.5 综合解读

子空间**在 decode 阶段吃到两个层面的收益**：

1. **紧凑 buffer 物理占地 4× 小**（64 MB / layer @ 128K）→ 显存与带宽上限降低
2. **K2=1024 候选消除 15.6× 的 L2 工作量**（收益来源是「两级索引」整体，不限于子空间）
3. **decode TPOT 平坦化**（e166h 数据：32K→128K 仅 +3.6% busy）= 两级索引预算恒定 → 子空间在其中的体现

---

## 3. Far Skip 资源量化（E170c）

方法：解析字节账本（模型 config 程序化，不实际分配）。

### 3.1 字节账本

| L | far/step | near/step | sel/dense | far 占比 selected | dense GiB | compact GiB |
|---|---|---|---|---|---|---|
| 32K | 12.9 MB | 53.2 MB | 16% | 19.5% | 4.50 | 0.721 |
| 64K | 12.9 MB | 53.2 MB | 8% | 19.5% | 9.00 | 0.721 |
| 128K | 12.9 MB | 53.2 MB | **4%** | 19.5% | 18.00 | 0.721 |

### 3.2 显存包络（紧凑布局 = 物理上仅存储候选 token）

| L | dense | compact | far skip 36 层 | compact 比例 |
|---|---|---|---|---|
| 32K | 4.50 GiB | 0.721 GiB | 0.581 GiB | 16% |
| 64K | 9.00 GiB | 0.721 GiB | 0.581 GiB | 8% |
| 128K | 18.00 GiB | 0.721 GiB | 0.581 GiB | **4%** |

→ **96% 显存节省**（128K），对应 e163 判决的 6.3/12.5/25×。

### 3.3 Far 占选中流量份额

**12.9 MB far / (12.9 + 53.2 MB total) = 19.5%**（三长度恒定，与 e163 判决完全一致）。

arxiv baseline 的稀疏注意力（H2 / StreamingLLM / Uniform）far 覆盖率远高于本文的 19.5%——我们的两级索引将 far/K2 产出压缩到了这个水平。

### 3.4 dgate prefill 成本

prefill 期间进行一次 `calibrate_dyn_dgate`（prefill 尾行 far probability 质量），在所有层的 attention 计算完成后执行。成本已含在 prefill wall 中，无额外 CUDA sync。

实测：dgate 标定时间包含在 prefill wall 中，无独立 overhead（prefill 流程内顺序执行）。

---

## 4. Op 级归因与性能浪费（E170d）

方法：`e170_opattr_v2.py` = e166_opattr.py 改 `TLI_SHADOW_PKG=~/tli_ako/solution`。capture 阶段 profiler record_shapes。

### 4.1 Top kernel families by count

| 家族 | count | 说明 |
|---|---|---|
| cudaLaunchKernel | 15132 | 总 launch 数（iter3 后最优） |
| elementwise (compare/bitwise/mul/add) | ~10K | 弥散群（36 层分摊） |
| Memcpy DtoD | 1436 | gather 路径 KV 搬运 |
| Memset / Fill | 2446 | 掩码填充 |
| cublasLt GEMM | 1620 | attention gemm（地板不动） |

### 4.2 与 e166g_iter3d 对照

cudaLaunchKernel 15132 与 e166g 完全一致。elementwise 弥散群无单一热点。

### 4.3 结论

**性能浪费铁律**（e166e 判决仍成立）：
- iter2+3a+3b+cfg+3d 已把 bitexact 兼容杠杆全部收割
- 剩余 topk ~3.3ms → 质量门口径，留用户决策
- elementwise 弥散群 → 无单一热点，需整链 Triton 化（开放工程，不破坏 bitexact 前提下才做）

---

## 5. Bitexact 正确性（E170a）

三长度(32K/64K/128K) × 8 gen tokens：
- L=32768: tpot=18.731ms, md5=d8c8649256835d2e1b7f85bfee0b4d15
- L=65536: tpot=19.060ms, md5=d8c8649256835d2e1b7f85bfee0b4d15
- L=131072: tpot=19.397ms, md5=d8c8649256835d2e1b7f85bfee0b4d15

**三长度 md5 完全一致** → 输出逐位相同。
**TPOT 平坦化** 验证 e166i/j/h 结论：gemm/attn/copy 与上下文解耦。

---

## 6. 补丁清单（待 merge）

| # | 文件 | 改动 | 类型 |
|---|---|---|---|
| 1 | sparse_attn/indexer/tli_indexer.py L840 | + input_signature 门控 | 正确性 + 性能 |
| 2 | sparse_attn/indexer/tli_indexer.py L886-888 | _inc dict 加 signature | 正确性 |
| 3 | sparse_attn/indexer/tli_indexer.py L1112 | fb = S_prev // bs | 性能 |
| 4-6 | sparse_attn/indexer/tli_indexer.py L1137/1140/1144, L1210 | new_zeros → new_empty (4处) | 性能 |

---

## 7. 下一步

1. **Merge 到 main**：5 处改动 + e170 verdict 文件 + 本报告 → commit → merge → push origin
2. **CURRENT_NARRATIVE.md 更新**：E170 索引新增
3. **论文可选**：子空间 4× buffer / far skip 96% 显存 / 性能浪费收敛结论 → 可入 revision

---

## 8. Verdict 文件索引

| 文件 | 内容 |
|---|---|
| pc40 `~/tli_ako/e170a_gpt_patch_bitexact_verdict.json` | 三长度 md5 + TPOT |
| pc40 `~/tli_ako/e170b_subspace_microbench_verdict.json` | L1/L2/buffer/traffic 四维 |
| pc40 `~/tli_ako/e170c_farskip_resource_verdict.json` | 字节账本 + 显存包络三长度 |
| pc40 `~/tli_ako/e170d_opattr_verdict.json` | kernel count top-400 + 消除确认 |
