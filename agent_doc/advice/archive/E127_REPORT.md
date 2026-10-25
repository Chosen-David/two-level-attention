# E127 最终报告 — TLI 稀疏注意力 kernel 优化（HF 路径 decode）

- **worktree**：/tmp/e127_kernel_wt（分支 e127-kernel-opt，HEAD = `0f542e4 [final] Restore iter8`）
- **约束遵守**：锚点 `sparse_attn/indexer/tli_indexer.py` 零改动（md5 `bf5e26c4d6123875b2786b9ed886da5f` 与主仓逐位一致）；未 push 任何 remote；所有优化路径 env 门控 `TLI_E127_FAST=1`，默认 0 走原路径（在飞链零扰动）。
- **形状口径**：Qwen3-8B decode —— D=128、GQA 32/8（g=4）、bs=64、K1=128 / K2=1024、S=8K/16K/32K/64K。

## 1. 定案数字

**S=64K（65536）每层每 decode step，median：**

| 口径 | baseline | iter8（final） | 加速比 |
|------|----------|----------|------|
| GPU busy sum | 8.29 ms | **1.52 ms** | **5.47×** |
| CPU wall | 9.62 ms | **2.02 ms** | **4.76×** |
| CUDA launch/step | 137 | 115 | — |

全档（vs baseline，normal 层）：16K 1.88×/1.94×、32K 3.07×/3.01×、64K skip 5.92×/5.06×、8K 1.16×/1.15×。

分阶段（64K normal）：prepare_index 2.89→~0.45、compute_score 2.80→0.35、compute_mask 0.30→0.36（原版被同步掩盖）、metrics 0.07→0.09、eager_decoding 2.23→0.19。

## 2. 优化路径（正结果，iter1-8）

| iter | 方向 | 64K GPU/wall | 要点 |
|------|------|------|------|
| 1 | metrics 惰性化 | 8.18/8.68 | 每层每步 2×.item() 同步 → 读取端一次；metrics CPU 3.72→0.09ms |
| 2 | eager gather | 6.62/7.12 | 消费端丢弃稀疏性对全 S 算 attention → gather 选中子集（~40× 带宽削减）；路由 T≥16K |
| 3 | gather 零同步 | 6.25/6.75 | 三处 host 同步消除：错误标志 GPU 化（fail-closed 语义守恒，raise 时机后移）+ W 上界化 + 长度零同步推导 |
| 4 | prepare_index 增量缓存 | 4.21/4.71 | decode 每步仅尾块在变；块统计/quant 逐 token 独立 → 缓存 ≡ 全量重算（**逐位一致数学保证**）；prepare 2.89→0.62ms |
| 5 | rep 缓存 + mask 同步消除 | 2.13/2.63 | b_k_qat 惰性增量缓存（1.6GB/步写带宽消除）+ skip 层 .item() 数学恒等消除；**锚点方法文本派生**（inspect.getsource 唯一替换，count≠1 拒载） |
| 6 | score_fine Triton | 2.15/2.65* | GQA K 复用 kernel，读带宽 4× 削减 + _rep 退役省 1.07GB/层显存；*GPU busy 实降 0.46ms/层（launch-gap 转移，详见 §4） |
| 7 | eager Triton 融合 | 1.73/2.23 | ~15 op → 2 kernel（mask 展开 + gather online-softmax）；eager 全档 0.55-0.60→0.186-0.191ms；launch 127→37 段实为 6 |
| 8 | score coarse Triton | **1.52/2.02** | minmax 双点积单 kernel（GQA k_min/k_max 块复用）+ avg 段复用 fine kernel；3× repeat 物化 + clamp 胶水 ~10 launches → 1；compute_score 27→13 launches |

结构性热点定位：baseline 单步 = 三大 O(S) 全量重算（prepare/score/eager 消费端）+ 72 次/步 host 同步。iter1-3 消同步与稀疏消费、iter4-5 增量化消除全量重算、iter6-8 Triton 化消除物化/胶水 launch。

## 3. 负结果（iter9-11，探索性保留）

**方向**：compute_mask 前段融合——coarse/fine kernel gmean 双输出前移（消 2 rearrange + 2 MeanOps）+ fused masked-softmax-gmean（六连 repeat+where+softmax+nan_to_num+mean+置位 → 2 kernel）。

**实测谱系**：16K normal +10%（-13μs）兑现；64K 主档 -3%（+43μs）回归；32K/8K 噪声内。

**A-B-A-B 背靠背交替对照定论**（iter11 已完全回退 = iter8 行为，排除时间漂移）：
- iter11 64K 稳定 +30μs vs iter8 同环境——残余差异是 host 侧路由判断/构造的纯 Python 开销（launch-gap 主导口径下 CPU 排队时间直接进 stage 计时）。
- 方向收益量级（10-20μs）< 本机环境方差（±20μs）< host 路由残余（30μs），全档不可稳定兑现。
- 结构性证据（profiler）：mask busy 194μs + gap 290μs、score busy 110μs + gap 260μs——kernel 本体已被融合方向证明可压，但 115 launches/step 的 CPU 排队（~800μs）才是剩余大头。

**开发中抓到的两个 bug（教训入库）**：
1. gmean 输出少一次 unsqueeze（3 维）→ 守卫恒假静默回退，门禁 gmean 断言抓出——**门禁须断言"优化路径确实生效"而非仅断言结果正确**。
2. Python 引用别名：直供 dict 键绑引用而非 clone，compute_mask 原地写（far 区 -inf / 末块 +inf）透传污染缓存键——单步无 mask 不复现，多步带 mask 才显形。**「读侧直供」须审原版写路径的别名结构**。

**处置**：stall 规则触发（3 iter 无稳定改善）→ `git checkout 226589a -- <files>` 逐字恢复 iter8，`[final]` commit `0f542e4`；恢复后 sanity：门禁 15/15 python±-O 双跑绿、64K stage bench 1.5201/2.0221 与定案吻合。

## 4. 数值口径（行为不变性）

- **逐位一致（==，非 allclose）**：index_dict + mask 全部档位/配置/多步增量；coarse/avg 分数（iter8 门禁实测 ULP=0/289408）；metrics 输出。
- **ULP 口径（差异比例 ≤1e-4 + 相对 ≤2%）**：score_fine（fp32 累加顺序 vs cublas，48/10.5M，maxrel 6.8e-3 = 1 bf16 ULP）——mask/topk 选择 4 配置 × normal/skip × 10 步实测逐位不变。
- **allclose(1e-2)**：attention 输出（eager gather/online-softmax，worst 1.95e-03）。
- **fail-closed**：所有 Triton 路径异常回退原版表达式（_triton_ok 标志）；契约校验（B7 空行 / W 超上界 / cu 破坏）GPU 化后语义守恒、同文案 raise。
- 门禁 `exp/bench/test_e127_gates.py` 15 项 python±-O 双跑；既有回归 7 套件（near_swa 3/3+4/4、C3 5/5、E122 6/6、E125 尾巴 10/10、E125 kmeans 守卫 5/5、E116f 12/12）env=1 全绿。

## 5. 交付物

- **代码**（全部 env 门控，默认零路径差异）：
  - `sparse_attn/indexer/tli_indexer_fast.py`（+532 行，iter4-8：增量缓存 + 文本派生 + Triton kernels）
  - `sparse_attn/ops/eager_decoding.py`（+416 行，iter2/3/7）
  - `sparse_attn/metrics.py`（iter1 惰性化）、`sparse_attn/patches/patch.py`（接线）
- **门禁/基准**：`exp/bench/test_e127_gates.py`（15 项）、`exp/bench/e127_stage_bench.py`、`exp/results_efficiency/e127_stage_bench_*.json`（baseline + iter1-11 + final）
- **日志**：`ITERATIONS.md`（12 条目完整记录，含负结果分析）
- **commit 链**：`41164af`（baseline）→ `6efd0e8..226589a`（iter1-8）→ `561a5aa/30f76c7/30b0ac5`（iter9-11 负结果）→ `0f542e4`（[final] 恢复 iter8）

## 6. 遗留方向（超出本任务范围，登记备查）

1. **launch 排队压缩（剩余大头）**：115 launches/step 的 CPU 排队 ~800μs > GPU busy 73%。方向：compute_mask 的 L2 分区 topk 段（4×topk + 3 排序 + softmax 链，35 launches）融合、prepare_index 33 launches 压缩。属大规模重构，非 iter9 融合方向的延伸。
2. **CUDA Graph**：decode 步形状固定，可整步 capture 消除全部 launch 开销——与增量缓存天然契合（GPU 指针稳定）。
3. **真实 Qwen3-8B e2e 对拍**（里程碑时）：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B` decode 验证 + 生成质量门禁。
4. **iter6 教训补充**：单层 event 同步口径测不出 36 层流水收益（CPU 跨层跑在 GPU 前面）——final 数字 5.47× 为保守下界，e2e 口径预期更高。
