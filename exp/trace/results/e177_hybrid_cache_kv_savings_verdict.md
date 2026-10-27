# E177 判决：HybridCache KV 节省 + 论文三维度加速汇总

> 日期：2026-10-27 | 载体：exp/bench/e177_hybrid_cache_kv_savings.py
> 口径：Qwen3-8B 默认几何（36层, Hkv=8, D=128, bf16）
> 【2026-10-27 口径限定（审计补记）】本判决 §1 为**公式核算**（按 ring 容量
> 公式推 bytes），非设备实测；数值是**decode 稳态 KV 存储占用**口径——
> prefill 期完整 dense DynamicCache 峰值需求仍在，"节省 67%"**不可当作
> 总显存峰值降幅引用**。线性（非 skip）层按 capacity 全宽分配，实际节省
> 随 capacity 与请求长度的差值缩水。**70% skip 比率无任何 RULER 质量门
> 支撑**：skip 层 far 历史不可逆驱逐，质量边界未知——对外引用前必须先补
> 三方质量门（dense / 70% hybrid / 全层紧凑）。§2/§3 主表与加速数字为
> 独立口径（iter3 链 + E172-E174），与本节 HybridCache 无耦合。

## 1. KV Cache 物理占用（HybridCache 节省）

### 1.1 定义

| 配置 | 每层策略 | 物理容量 |
|------|---------|---------|
| Dense | 全宽 linear | L_max tokens |
| TLI-linear | 全宽 linear（far-skip 但不蒸发） | L_max tokens |
| TLI-hybrid(70%) | 70% 层 ring + 30% 层 linear | ring: sink+near+window+block-1 ≈ 4288 tokens |

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

**核心结论**：HybridCache 在 128K 下节省 67.2% KV 物理占用（70% skip 层），
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
> 算子级优化贡献 24-40% TPOT 下降，HybridCache 在 decode 稳态节省 **至多
> 67% KV 存储**（公式口径，70% skip 层、质量门未测），且 decode 成本与
> 上下文长度解耦（4× 上下文仅 +3-4% 成本）。
> 【引用纪律】"67% 稳态 KV 节省"带三个限定：稳态非峰值 / 公式非实测 /
> 无质量门。三者任一未闭合前不得进论文正文。

## 5. 合入状态与后续修复记录

- ✅ `codex/far-skip-hybrid-cache-20261026` cherry-pick 到 `perf/hybrid-cache-merge`，
  已随 `6c559c1` + `b0a378c` 合入 main
- ✅ 新增 `hybrid_batch_cache.py` + `hybrid_batch_runner.py`
- ✅ 测试 34/34 全过（18 hybrid_batch + 16 far_skip_hybrid）
- ✅ 回归测试 18/18 全过

### 5.1 2026-10-27 in-place 修复（本判决同日审计后落地）

- **构建期峰值显存**：`HybridBatchCacheBuilder.add()` 现逐层消费源
  DynamicCache（每层拷入目的地后立即置空源 KV），构建峰值从
  `Σ源+Σ目的` 压到逐层 `max(源,目的)`；新增测试断言 add() 返回时
  源 KV 已释放（不等调用方 del）。CPU 34/34 + 回归 58/178 subtests
  全绿；行为对拍无扰动。
- **GPU 验收欠账闭合（原 validation.md 未闭合项，本日补齐）**：
  验证基座 = **本修复的精确推送物**（17e56e8 + 本修复，隔离 worktree，
  tar 部署 pc02 `~/wt_hybrid_gpu`，双卡空闲窗口，不触碰宿主仓库/shadow；
  另在 pc31 上以同套用例先行复跑过一轮，结果一致）：
  - hybrid 全套含 CUDA 用例：**34/34 PASS**（含 batch∈{2,8,32}×
    fp32/fp16 真实整模型 capture、38 步 replay、EOS 冻结、第二请求
    重新 capture、graph/eager 逐 token 对拍 rtol≤3e-3）
  - E127 硬门（含 tli_layer_skip_mask.json 完整 079 门控）：
    **15/15 PASS**
  - tli_batch 硬门：**59/59 PASS**
  仍未闭合（如实声明）：真机大模型 TPOT/显存峰值实测、RULER 质量
  门——见下条。
- **已知未修（留用户/后续实验）**：①70% skip 比率质量门（最高优先，
  三方 RULER：dense/70%/全紧凑）；②批 decode 行串行（TPOT∝B，跨行
  融合=独立实验系列）；③每批重 capture 成本未测量；④70% 与 100%
  （e163 全紧凑 25×）之间的显存-质量边界未标定。
