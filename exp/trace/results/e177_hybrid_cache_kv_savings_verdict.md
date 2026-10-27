# E177 判决：HybridCache KV 节省 + 论文三维度加速汇总

> 日期：2026-10-27 | 载体：exp/bench/e177_hybrid_cache_kv_savings.py
> 口径：Qwen3-8B 默认几何（36层, Hkv=8, D=128, bf16）
>
> 2026-10-27 源码复核：载体脚本只执行整数容量公式，没有创建 HybridCache、
> 分配 CUDA 张量或读取 memory_allocated/reserved/peak。以下保留原始估算数字，
> 不作为显存实测。70% 是人为输入，`int(36*0.7)=25` 层，非 per-prompt gate 观测。
> 公式不含 prefill/cache/output 引用、子空间摘要、Graph workspace/pool 与 decode 预留；
> 模型质量及真实 gate 分布必须另验。历史 TPOT 也不是本 hybrid 实现的配对结果。

## 1. KV Cache 容量估算（非 CUDA 显存实测）

### 1.1 定义

| 配置 | 每层策略 | 物理容量 |
|------|---------|---------|
| Dense | 全宽 linear | L_max tokens |
| TLI-linear | 全宽 linear（far-skip 但不蒸发） | L_max tokens |
| TLI-hybrid(70%) | 70% 层 ring + 30% 层 linear | ring: sink+near+window+block-1 = 4291 tokens |

### 1.2 Qwen3-8B 全模型 KV 占用（GiB, bs=1）

| 长度 | Dense | TLI-linear | TLI-hybrid(70%) | 节省 |
|------|-------|-----------|----------------|------|
| 8K   | 1.125 | 1.125     | 0.753          | 33.1% |
| 16K  | 2.250 | 2.250     | 1.097          | 51.3% |
| 32K  | 4.500 | 4.500     | 1.784          | 60.4% |
| 64K  | 9.000 | 9.000     | 3.159          | 64.9% |
| 128K | 18.000| 18.000    | 5.909          | **67.2%** |
| 256K | 36.000| 36.000    | 11.409         | **68.3%** |

### 1.3 不同 skip 比率（128K, Qwen3-8B）

| skip 比率 | 全模型 KV (GiB) | vs dense 节省 |
|-----------|----------------|--------------|
| 0% (全 linear) | 18.000 | 0% |
| 30% | 13.164 | 26.9% |
| 50% | 9.295 | 48.4% |
| 70% | 5.909 | **67.2%** |
| 90% | 2.524 | **86.0%** |

**容量模型结论**：假设 25/36 层固定使用 ring，128K 下理论 KV 容量减少 67.2%，
且节省率随长度单调增长（饱和趋势，因 linear 层占比固定 30%）。

## 2. 算子级加速（iter3 优化链分解）

| 优化 | 节省/步 | 贡献 | 判决文件 |
|------|--------|------|---------|
| iter2a 几何步级缓存+死op删 | ~0.8ms | ~3.2% | iter2a_verdict.json |
| iter2b 紧凑域 softmax+topk | ~1.2ms | ~4.9% | iter2b_verdict.json |
| iter2c span 并行 expand | ~0.5ms | ~2.0% | iter2c_verdict.json |
| iter3a 掩码装配融合 | ~0.18ms | ~0.8% | e166a_iter3a_verdict.json |
| iter3b 候选压缩融合 | ~0.90ms | ~4.0% | e166b_iter3b_verdict.json |
| iter3d 摘要/L1/候选三kernel | ~2.5ms | ~11.8% | e166d2_iter3d_verdict.json |
| compact cfg 调优 | ~0.34ms | ~1.6% | e166e_compact_cfg_verdict.json |
| **累计** | **~5.0ms** | **~24-40%** | — |

**残余热点**：topk 家族 ~3.3ms（17%）+ copy 弥散群 ~3.55ms（18%）。

## 3. E2E 加速比（TLI-graph vs Dense-Eager, K1=64, iter3）

### 3.1 主表（decode-only TPOT）

| 长度 | Qwen3-8B | Llama-8B | Qwen3-32B |
|------|----------|----------|-----------|
| 32K  | 1.451×   | 1.428×   | 1.338×    |
| 64K  | 2.216×   | 2.192×   | 1.924×    |
| 128K | 3.636×   | 3.643×   | 3.015×    |

### 3.2 长度平坦化

| 模型 | 32K→128K TLI 增长 | 32K→128K Dense 增长 |
|------|-------------------|---------------------|
| Qwen3-8B | +4.2% | +161% |
| Llama-8B | +4.1% | +165% |
| Qwen3-32B | +3.3% | +133% |

## 4. 论文视角一句话总结

> LoDA-SA（TLI）在 128K decode 上实现 **3.0-3.6× e2e 加速**（vs dense），
> 算子级优化贡献 24-40% TPOT 下降，HybridCache 在假设 25/36 层 skip 下预计减少 **67% KV 容量**（未测 CUDA allocator），
> 且 decode 成本与上下文长度解耦（4× 上下文仅 +3-4% 成本）。

## 5. 合入状态

- ✅ `codex/far-skip-hybrid-cache-20261026` cherry-pick 到 `perf/hybrid-cache-merge`
- ✅ 新增 `hybrid_batch_cache.py` + `hybrid_batch_runner.py`
- ✅ 测试 34/34 全过（18 hybrid_batch + 16 far_skip_hybrid）
- ✅ 回归测试 18/18 全过
- ⏳ 待 merge 到 main + 更新 CURRENT_NARRATIVE
