# S-T025：E126 五臂 TLI 选择器分阶段计时 microbench（成本指纹）

来源：用户 2026-10-11 派单（E126）。任务性质：**优化前基线测量**——E127
优化验收合入后须同协议复测，优化前/后两组数字**成对**进 E128 综合表；
本报告所有数字单独不得表述为「优化后」或与任何未配对复测数字横比。

状态：**收官**（2026-10-11）。红绿测试全绿（python±-O），H20 正式计时
五臂全落袋，结果已入库 worktree 分支 `e126-stage-timing`（commit
eb308c7 harness + 6325bcf 结果 JSON；按隔离纪律**未 push 任何 remote**，
E127 验收后再议合入）。

## 1. 目标与口径

给五臂 TLI 选择器（E125 冻结配置）建立分阶段 GPU 成本指纹：

| 臂 | 配置（E125 派单口径） |
|---|---|
| mavg | far=minmax / near=avg，α=.25 β=.125 γ=.625（生产冠军） |
| aavg | far=avg / near=avg，(0,0) 单池 |
| mminmax | far=minmax / near=minmax |
| cavgsim | far=sim_greedy（--tli_sim 0.9）/ near=avg，kmeans on |
| cclustersim | far=sim_greedy / near=sim_greedy，kmeans on |

共同协议：B=1 / q_len=1（decode）/ H=32 / Hkv=8 / D=128 / K1=128
（page=64）/ K2=1024 / cmp_ratio=4 / tail32 子空间（dd=32）/ swa=128 /
sink 2 块 / tli_enable_layer_skip=False（microbench 单层无 D'）/
kv bfloat16 / greedy Triton kernel 开（SGLANG_TLI_GREEDY_KERNEL=1）。
S ∈ {4096, 8192, 16384, 32768, 65536}；每格 warm=3、REPS=10 ×
N_STEPS=128；structured + singleton 两数据形态（成本对分布敏感性的对照）。

阶段边界（S0-S8，行号锚定 tli_indexer.py md5 bf5e26c4 版本）：

| 阶段 | 内容 | 生产源位置 |
|---|---|---|
| S0 | pad + 子空间 prepare_index | L277-291 |
| S1 | L1 块统计 + 4bit 量化 | L291-349 |
| S2 | 建簇（仅 cavg 系；S2_far/S2_near 分段） | L368-409 |
| S3 | q 侧打分：S3_coarse 块分 + S3_avg 均值分 + S3_fine 细筛分 | L682-783 |
| S4 | 簇分数（far_tok_score，仅 cluster 臂显著） | L957 起 |
| S5 | L1 页级 topk | L865 起 |
| S6 | softmax p | L1070-1079 |
| S7 | L2 分区预算 token topk | L1149-1372 |
| S8 | eager_decoding_attn（mask 消费端，单列不计入选择器） | ops/eager_decoding.py |

## 2. 方法：零改动运行时 AST 插桩

**生产文件字节零改动**（tli_indexer.py 是 E125/E116b 在飞实验身份锚点，
md5 bf5e26c4d6123875b2786b9ed886da5f 全程 fail-closed 校验）：
harness 运行时读源文件 → 13 个 anchor 行号+行首内容硬校验（漂移即
SystemExit fail-closed）→ AST 在 anchor 前与函数 Return 前注入
`self._e126_mark(tag)`（cuda Event 对）→ 独立命名空间模块 exec 注入版。
计时开关 `TLI_STAGE_TIMING`（默认 0 = 全程零事件零开销路径）。

测量协议继承 e113（054/055/056）：阶段 span = 相邻 cuda Event
elapsed_time，**ms 原始样本按执行顺序持久化不排序不舍入，median 由持久化
样本派生**；选择器 wall = perf_counter_ns 整数纳秒；S8 独立 event 对；
e2e_128step = 计时关 1 rep 的 128 步粗粒度合并 wall（非服务级精度）。
fail-closed 正确性门每格运行（mask 无全 False 行 / 输出 finite /
选中 token 数=K2 区间核对，五臂全部 sel_tokens_mean=1024 精确命中）。
输出原子写 + sha256 sidecar + flock + attempt supersede + 身份闭包
（git SHA eb308c7 + tli_indexer.py sha256
2a0935cb5a7152f65ddbe63b3055f4a13dccb7e76abcf74dc7b2bba55f02bb03）。

红绿测试 `exp/trace/test_e126_stage_timing.py` 四组（python±-O 双跑全绿，
045 口径显式 check 不依赖 assert）：
- T1 anchor 硬校验 + 插桩构建 + 生产文件 md5 零改动 + 源文本零污染
- T2 计时开/关同输入逐位一致：五臂 mask+attention 输出 md5 逐位相等
  （基线非确定时 delta-delta 判定：timing-on 结果须落在基线噪声集合内）
- T3 mark 序列 == 各臂期望序列；span 非负；span 总和 ≤ wall×2+2ms
- T4 TLI_STAGE_TIMING=0 → 全程零事件

## 3. 五臂 × 阶段中位成本（structured，ms，H20）

### 3.1 mavg（生产冠军）

| S | S0 | S1 | S2 | S3c | S3a | S3f | S4 | S5 | S6 | S7 | 选择器Σ | wall | S8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 0.352 | 0.218 | 0.013 | 0.337 | 0.101 | 0.230 | 0.003 | 0.179 | 0.095 | 0.133 | 1.67 | 1.80 | 0.37 |
| 8192 | 0.390 | 0.210 | 0.003 | 0.330 | 0.101 | 0.394 | 0.003 | 0.060 | 0.051 | 0.133 | 1.68 | 1.80 | 0.44 |
| 16384 | 0.472 | 0.346 | 0.003 | 0.275 | 0.102 | 0.720 | 0.003 | 0.062 | 0.031 | 0.120 | 2.14 | 2.24 | 0.91 |
| 32768 | 0.634 | 0.634 | 0.003 | 0.277 | 0.103 | 1.370 | 0.003 | 0.066 | 0.042 | 0.127 | 3.27 | 3.36 | 1.62 |
| 65536 | 0.959 | 1.234 | 0.003 | 0.279 | 0.103 | 2.670 | 0.003 | 0.073 | 0.062 | 0.159 | 5.55 | 5.64 | 3.05 |

aavg 与 mavg 逐阶段几乎逐位一致（同为 avg 细筛路径，差值 <0.01ms）；
mminmax 因 near 无 avg 分支省 S3_avg（0.013ms），选择器 Σ 全尺寸最低。

### 3.2 cavg 系额外成本（S2+S4 相对 mavg 同格差值）

| S | cavgsim extra | cclustersim extra |
|---|---|---|
| 4096 | 0.060 | 0.134 |
| 8192 | 0.033 | 0.064 |
| 16384 | 0.039 | 0.055 |
| 32768 | 0.059 | 0.079 |
| 65536 | 0.092 | 0.116 |

**步均额外成本 0.03-0.13ms**：摊销后 cavg 系与 4bit 细筛臂的稳态选择器
成本差 <2%（cclustersim 65536 选择器 5.73 vs mavg 5.55）。

### 3.3 singleton 对照

五臂 singleton 与 structured 选择器成本几乎一致（例 S=65536：mavg
5.550 vs 5.552，cclustersim 5.720 vs 5.733）——**选择器成本对 KV 数据
分布不敏感**（差异体现在检索质量侧而非成本侧）；成本指纹可作为分布
无关的配置函数引用。

### 3.4 主要阶段随 S 的标度

- **S3_fine 细筛分是最大选择器成本项**，随 S 近线性（0.23→2.67ms，
  4K→64K ×11.6）：候选 token 全量 4bit 细筛，O(S)。
- **S1 块统计/量化随 S 线性**（0.21→1.23ms）。S0 pad/子空间 0.35→0.96ms。
- S3_coarse/S3_avg/S5/S6/S7 对 S 不敏感（候选域/预算恒定，0.1-0.3ms 级）。
- S8 eager attention 随 S 增长（0.37→3.05ms）由 K2 预算内读取 + 长度
  效应主导；选择器 Σ 与 S8 同量级，**选择器与 attention 消费端成本相当**，
  两级索引的索引开销不可忽略（E128 归因表用）。

## 4. 摊销曲线结论（R2c：S2 建簇步号 vs 累计耗时）

128 步逐步记录 S2（rep0 全步曲线持久化在 arm JSON `amortization` 段）：

| S | 冷建簇（step=S） | 块对齐重建步 | 步中位 |
|---|---|---|---|
| 4096 | cavgsim 11.2ms / cclustersim 10.6ms | 0.49 / 1.61ms | 0.031 / 0.046ms |
| 16384 | 136 / 121ms | 0.86 / 8.39ms | 0.006 / 0.009ms |
| 32768 | 489 / 445ms | 1.40 / 28.72ms | 0.006 / 0.009ms |
| 65536 | 1859 / 1680ms | 2.46 / 105.31ms | 0.006 / 0.009ms |

结论：
1. **块对齐缓存命中机制验证通过**——far_hi 块对齐（64 token 倍数）时
   才重建，128 步仅重建 2 次；步中位 0.006-0.009ms（命中路径只做指纹
   比对）。远低于 61s/step 反常阈值（sanity 锚点不触发）。
2. **冷建簇成本超线性**（S×16 时成本 ×150-167，~S^1.6+）：far_clusters=256
   × niter=10 的 kmeans 迭代主导。decode 期一次性建簇在 S=64K 需
   1.7-1.9s——**e2e 口径不可隐藏**（见 3.5 e2e128：cclustersim 64K
   3001ms vs mavg 986ms，差值 ≈ 建簇累计 1995ms 吻合）。
3. cclustersim 重建步比 cavgsim 贵 ~40×（105 vs 2.5ms @64K）：near 侧
   开 sim_greedy 后两路都重建。prefill 期建簇 + decode 期只读复用
   （ClusterKV 口径）是后续优化方向。

### 4.1 e2e_128step 粗粒度 wall（计时关，ms）

| S | mavg | aavg | mminmax | cavgsim | cclustersim |
|---|---|---|---|---|---|
| 4096 | 242.5 | 243.2 | 226.6 | 241.4 | 285.3 |
| 16384 | 364.3 | 363.6 | 349.8 | 493.1 | 526.4 |
| 65536 | 985.7 | 983.3 | 963.0 | 2838.9 | 3001.1 |

含建簇的 decode e2e 中 cavg 系在 S≥16K 显著劣化（建簇累计主导）；
选择器稳态差异（3.2 节）不足以解释 e2e 差——**成本归因必须分
「摊销稳态」与「一次性建簇」两层报告**，这正是本任务摊销曲线的用途。

## 5. H20 部署与拉回记录

- 正式计时机：H20（33.32.33.24:8416，用户 wangyuanshuo02），
  rsync 部署（--exclude=.git --exclude=__pycache__），单卡独占运行。
- 身份门：harness 启动即校验 `--git-sha eb308c7` +
  `--expect-tli-sha256 2a0935cb...`（H20 端 tli_indexer.py 与本地逐位
  一致才跑）；结果 JSON meta 段含 gpu 型号 "NVIDIA H20-3e"、全协议参数、
  臂 flags。
- 拉回：五份 arm JSON + summary + sha256 sidecar rsync 回本地 worktree
  `/tmp/e126_timing_wt/exp/trace/results/`，入库前 sidecar SHA 全量核验
  12/12 通过（git add -f 破 .gitignore exp/ 规则——E123 083 同坑，
  git check-ignore 逐文件核验后强制添加）。
- 本地 CUDA_VISIBLE_DEVICES=1 仅做 smoke（红绿 + 单臂小跑），正式数字
  全部出自 H20。

## 6. 交付清单

| 交付 | 位置 |
|---|---|
| 计时 harness | `exp/trace/e126_stage_timing.py`（worktree 分支 e126-stage-timing，commit eb308c7） |
| 红绿测试 | `exp/trace/test_e126_stage_timing.py`（同上） |
| 五臂结果 JSON + sidecar | `exp/trace/results/e126_stage_timing_{mavg,aavg,mminmax,cavgsim,cclustersim}.json{,.sha256}`（commit 6325bcf） |
| 汇总 | `exp/trace/results/e126_summary.json{,.sha256}`（同上） |
| 报告副本 | `/tmp/e126_timing_wt/E126_REPORT.md` |

## 7. 未覆盖项（如实清单）

1. **B>1 / q_len>1（prefill 形态）未测**：本任务只测 decode（B=1，
   q_len=1）；prefill 期 chunk 打分与建簇成本未覆盖（E127 若动 prefill
   路径须补）。
2. **D'/layer_skip 未开**：microbench 单层无 D' gate；多层场景的
   skip 分摊（多数层跳过选择器）未覆盖。
3. **e2e_128step 是粗粒度口径**：计时关 1 rep、单层合成数据，非服务级
   精度，只作选择器+S8+建簇的合并 sanity，不作论文 E2E 数字。
4. **kernel 路径 greedy_cluster_pass 计时**：Triton kernel 开（env=1）；
   Python 回退路径未单独计时（生产不使用）。
5. **E128 综合表**：本报告是优化前基线半边；E127 验收合入后须同协议
   同机复测，两组数字成对比较后进 E128，**严禁单边引用**。
