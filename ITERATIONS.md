# E127 迭代日志（ako4all 循环）

格式：每 iter = bench 运行 + 本日志追加 + git commit。基线 = 当前 main 的 TLI HF 路径（env 关）。

## Summary（随迭代更新）

| iter | 方向 | S=64K 全路径 GPU ms（median） | vs 基线 | 门禁 |
|------|------|------|------|------|
| baseline | 现状（无 env 开关） | 8.29 GPU / 9.62 CPU wall | 1.00× | — |
| iter1 | metrics 惰性化（去 2×.item() 同步/层步） | 8.18 GPU / 8.68 CPU wall | 1.01×GPU / 1.11×wall | 2/2 PASS |
| iter2 | eager_decoding gather（≥16K 稀疏走 gather） | 6.62 GPU / 7.12 CPU wall | 1.25×GPU / 1.35×wall | 3/3 PASS |
| iter3 | gather 路径零同步化（错误标志 GPU 化 + W 上界化） | 6.25 GPU / 6.75 CPU wall | 1.33×GPU / 1.43×wall | 9/9 PASS |
| iter4 | prepare_index 增量缓存子类（块统计/quant 逐位一致） | 4.21 GPU / 4.71 CPU wall | 1.97×GPU / 2.04×wall | 12/12 PASS |
| iter5 | compute_score rep 缓存 + compute_mask 同步消除（锚点文本派生） | 2.13 GPU / 2.63 CPU wall | 3.89×GPU / 3.66×wall | 15/15 PASS |
| iter6 | score_fine Triton GQA K 复用（_rep 退役） | 2.15 GPU / 2.65 CPU wall（bench 口径持平；GPU busy 实降 0.46ms/层） | 3.86×GPU / 3.63×wall | 15/15 PASS |
| iter7 | eager gather Triton 融合（~15 op → 2 kernel） | 1.73 GPU / 2.23 CPU wall | 4.79×GPU / 4.31×wall | 15/15 PASS |
| iter8 | compute_score coarse 段 Triton GQA K 复用（minmax 双点积 + avg 复用 fine kernel） | 1.52 GPU / 2.02 CPU wall | 5.47×GPU / 4.76×wall | 15/15 PASS |
| iter9 | compute_mask 前段融合（gmean 双输出 + fused mask-p，64K 净回归混合结果） | 1.56 GPU / 2.06 CPU wall（64K 主档 vs iter8 回归 -3%；最优维持 iter8 1.52/2.02） | 5.31×GPU / 4.67×wall | 15/15 PASS |
| iter10 | iter9 按档路由（T≤32768 启用融合）：16K 兑现 +3.7%，32K/64K 疑环境漂移待对照 | 16K 1.46（1.95×）其余档漂移疑点未定 | 16K 1.95×/1.98× | 15/15 PASS |

## Iterations

### [baseline] 初始化 bench 与分阶段计时
- 结果：exp/results_efficiency/e127_stage_bench_baseline.json（4 档 S × normal/skip 层）
- 分析：S=64K 每层每步 GPU 8.29ms（prepare_index 2.89 / compute_score 2.80 /
  mask 0.30 / metrics 0.07 / eager 2.23），×36 层 ≈ 300ms/token。单步 137 CUDA
  launch。三大 O(S) 全量重算 + 72 次/步 host 同步是结构性热点。

### [iter 1] metrics 惰性化
- 改动：sparse_attn/metrics.py —— add_select_result 增 env 门（TLI_E127_FAST=1）
  惰性路径：GPU 端存 (where(cond, sum, -1), bs, H) 三元组，get_select_tokens
  读取端一次性逐条 .item() 后做与旧路径完全相同的 Python float64 算术。
- 原理：旧路径每层每 decode step 做 q_ids[i].item() + b_mask.sum().item() 两次
  host 同步（36 层 × 每 token = 72 次），每次同步 = 排空整个 launch 队列，
  CPU 无法跨层跑在 GPU 前面。惰性化把同步推迟到生成结束后的唯一一次读取。
- 数字（S=64K normal 层）：metrics 阶段 CPU 3.72ms → 0.09ms；全步 CPU wall
  9.62 → 8.68ms（-10%）；GPU busy 基本不变（符合预期——本项优化的是停顿不是
  计算）。drain 停顿现转移至 eager_decoding 的 .item()（iter 3 消除）。
- 门禁：exp/bench/test_e127_gates.py metrics 2/2 PASS
  （get_select_tokens 逐位一致 old=21887.578125 == lazy；sync_debug_mode=error
  下 20 次调用零同步异常）。
- commit：见 git log

### [iter 2] eager_decoding_attn gather 路径（稀疏 token gather 后 attention）
- 改动：sparse_attn/ops/eager_decoding.py —— 原实现整体保留为
  _eager_decoding_attn_dense；TLI_E127_FAST=1 时入口走 _gather：
  counts→一次读回（min==0 复用 B7 报错，同步次数与原 any().all() 持平）→
  topk 取每行选中位置 → gather K/V（~W×D per 行）→ 小尺寸 attention。
  路由：T<16384 直接稠密（零同步，8K 档与 baseline 同路径）；W*8>T 回退稠密。
- 原理：原消费端对全序列 S 算 fp32 分数再 mask——k.to(float32) 物化 +
  全长 einsum + 全长 softmax = O(S·Hkv·D) 带宽，稀疏性在消费端被完全丢弃。
  gather 后带宽 O(R·W·D)，W ≈ K2+sink+swa ≈ 1280-1600，S=64K 时 ~40× 带宽削减。
  softmax(全 S 行、未选位 -inf) ≡ softmax(选中子集)（exp(-inf)=0 精确），仅归约
  顺序不同 → 输出 ULP 级漂移。
- 数字：S=64K eager 阶段 GPU 2.23→~0.6ms（stage 归因含一次读回的 drain 停顿）；
  全步 GPU sum 8.29→6.62（1.25×），wall 9.62→7.12（1.35×）。S=8K 不回归
  （路由稠密，2.22 vs 2.15 baseline，噪声内）。隔离 microbench S=64K：
  dense 0.86ms vs gather 0.49ms（gather 组件：topk 0.10 + gather/einsum/softmax
  ~15 个小 op，其余为 launch 开销——后续 Triton 化方向）。
- 门禁：eager 3/3 PASS——allclose(1e-2) 全配置（S=4097/8192/16384/65523 ×
  shared/per_qh）worst_abs_diff=1.53e-05；B7 全 False 行守卫双口径一致 raise。
- commit：见 git log

### [iter 3] gather 路径零 host 同步化（错误标志 GPU 化 + W 上界化）
- 改动：sparse_attn/ops/eager_decoding.py —— 三处同步全消：
  ① stats 读回（torch.stack([max,min]).cpu() 取 W+B7 检查）→ W 改用
  host 已知契约上界 TLI_E127_W_UPPER（默认 2048；indexer 契约 K2(1024)+
  sink(128)+swa(128)+当前块(64)≈1344，50% 安全垫），B7 空行/W 超上界/
  cu 契约破坏三类违规 GPU 端原地点并全局标志（copy_+maximum，code=1/2/3），
  生成结束 eager_decoding_check_errors() 一次性读回校验——fail-closed
  语义守恒（同文案 raise、整次生成作废），仅 raise 时机后移（与 iter1
  metrics 惰性化同精神）。topk k=2048 时补齐位由既有 valid 掩码处理，
  数学路径与精确 W 完全一致。
  ② bos_k/eos_k 的隐式 .item() 切片（b_mask 裁剪 + KV 切段，dense 亦有的
  存量同步）→ 用 host 元数据判等零同步推导：mask 块宽×bs == k 序列长 →
  cu 恒 [0,S]（prepare_cu_seqlens 构造保证、B=1 生产路径恒成立）直接算；
  判等失败（投影 padding）回退 dense（同步次数与 env=0 持平）。
  ③ 路由 W*8>T 用 W_upper 判定（host 已知量）——生产 W≈1344 与
  W_upper=2048 在 16K/32K/64K 档路由同向，行为零变化。
- 开发中修的 bug：k.shape[1] 误用（k 布局 [seq,head,dim]，序列维是
  shape[0]）——判等恒假致 gather 全程未生效，maxdiff=0.0 暴露。
- 数字（S=64K normal 层）：eager 阶段 GPU 0.58→0.25ms（真实 GPU 计算
  浮出，drain 停顿从 stage 归因消失）；全步 GPU sum 6.62→6.25（vs
  baseline 1.33×），CPU wall 7.12→6.75（1.43×）。全档首测：16K normal
  1.23×/1.28×、32K normal 1.29×/1.38×、64K skip 1.24×/1.33×、16K/32K
  skip 1.06×-1.14×；8K 档路由 dense 零回归（2.19 vs 2.15，噪声内）。
  生产收益大于 bench 所示：bench 每步 cuda event 计时强制同步，跨层
  流水收益测不出——真实 decode 36 层/token 的每层同步全消后 CPU 可
  跨层跑在 GPU 前面。
- 门禁：eager 9/9 PASS——allclose(1e-2) 全配置（4 档 S × shared/per_qh，
  worst 1.22e-04）+ B7 小序列双路径当场 raise（双口径）+ B7 大序列
  gather 尾部 raise + W 超上界/cu 破坏尾部 raise + gather 主路径零
  host 同步（sync_debug_mode=error 下 40 次调用零异常）。metrics 2/2
  保持绿。既有回归 7 套件 env=0 全绿（near_swa 3/3+4/4、C3 5/5、E122
  6/6、E125 尾巴 10/10、E125 kmeans 守卫 5/5、E116f 12/12）；env=1
  子集复跑见 commit 后记录。
- commit：见 git log

### [iter 4] prepare_index 增量缓存子类（TLIIndexerFast，锚点文件零改动）
- 改动：新文件 sparse_attn/indexer/tli_indexer_fast.py + patch.py 接线
  （TLI_E127_FAST=1 时 tli 臂实例化 TLIIndexerFast，snapshot 注入同语义；
  默认 0 走原类，在飞链零扰动）+ bench 接线。
- 原理：原 prepare_index 每次 decode step 对全序列 O(S) 重算
  （S=64K 每层步 2.89ms：pad_tensor 全量 134MB 复制 + gather + 全量
  quant + zeros_like 134MB + amax/amin/mean 全块归约），但 decode 每步
  只追加 1 token——只有尾块统计在变。增量版：k_min/k_max/k_avg 复用
  块 [0, fb) 缓存 + 尾段重算（pad 语义与全量版一致）；k_qat 预分配
  buffer 步进写（尾段 zero_+scatter，容量翻倍摊销扩容）。
  **逐位一致为数学保证**：amax/amin/mean(dim=2) 是块内独立归约、
  quant 逐 token 独立，块间无交叉 → 缓存值（原 kernel 历史输出）≡
  本次全量重算值。契约 fail-closed：S_now ≤ S_prev（回退/新请求）全量
  重建；clear() 失效缓存（patch 每次 prefill 前调用）；投影基/moba/
  子空间 key 变化一律 super() 全量。k append-only 契约（HF
  DynamicCache）文档化。
- 数字（S=64K normal 层）：prepare_index 2.89→0.62ms（4.7×）；全步
  GPU sum 6.25→4.21（vs baseline **1.97×**），CPU wall 6.75→4.71
  （**2.04×**）。32K normal 1.64×、16K normal 1.28×、64K skip 1.76×。
  8K 档复测持平零回归（2.16/2.26 vs baseline 2.15/2.25；iter4 run 中
  8K skip 2.66 为单次环境噪声，复测两次均 2.26）。
- 已知 trade-off（如实记录）：compute_score 2.80→3.02ms（+0.22）——
  原版 prepare 每步全量重写 k_qat（134MB）使 compute_score 的 repeat
  读取命中 L2（temporal locality 红利）；增量版只写尾块，score 的
  全量读取退回 HBM。净 prepare+score 合计 5.69→3.64ms（-2.05ms），
  总账大赚；根本解法是消除 compute_score 的 1GB repeat 物化（锚点
  文件，见报告建议节）。
- 门禁：12/12 PASS——增量 12 步 decode（块边界 S=4093 起点/多 token
  +3/跨块大步 +63）normal+skip 层 index_dict+mask 逐位一致（==，
  非 allclose）+ 64K 大 S 15 步 mask 逐位 + S 回退全量重建/后续增量
  正确。既有回归 env=1 复跑 5 套件全绿（near_swa 3/3、C3 5/5、E122
  6/6、E125 尾巴 10/10、E125 kmeans 5/5）；env=0 套件 iter3 时已全绿
  （patch.py 改动 env=0 零路径差异）。
- commit：见 git log

### [iter 5] compute_score rep 缓存 + compute_mask 同步消除（锚点方法文本派生）
- 改动：sparse_attn/indexer/tli_indexer_fast.py（锚点文件依旧零改动）+ 门禁扩充
  exp/bench/test_e127_gates.py（+3 项至 15 项）。三块：
  ① compute_score 的 b_k_qat 物化（repeat(k_qat,"t h d -> t (h g) d").to(fp32)
  每步 [S,H,128]，S=64K 合计 ~1.6GB/步写带宽，profile 实测占该阶段 78%：
  拷贝/转换 2.23ms vs einsum 本体 0.58ms）→ _fast_b_k_qat 惰性增量缓存
  （尾段步进写 + 容量翻倍摊销扩容）。逐位等价：repeat/fp32 转换均为逐元素
  确定性算子，k append-only 前缀 token 的 k_qat 值不变（同 iter4 归纳）→
  缓存行 ≡ 全量 repeat+转换输出 → einsum 输入逐位同 → score_fine 逐位一致。
  fail-closed：k_qat 与增量缓存 data_ptr host 判同（零同步）才复用前缀；
  投影基/moba/外部合成 k_qat/维度不符一律全量重建（值同原版路径）。
  ② compute_mask 的 skip 层两处 n_valid 统计 .item() host 同步（skip 层
  compute_mask CPU 3.0ms 停顿主源）数学恒等消除：site1（e64 分支）far
  区间刚被整段置 -inf → 计数恒 0（host 常量）；site2（单池分支）sc_pool
  除 far 区外全有限 → 计数 = kt − far 区块数（host 可算）。另消两处
  far_range tensor 高级索引赋值（index_put_ 触发同步）→ 连续区间切片赋值
  （数学恒等，区间空时同为空操作）。
  ③ _qat_indices 缓存（原每步 list→tensor pageable H2D → GPU 端 arange
  cat 构造零 H2D）+ _prepare_index_inc 的 pad_cu 向量化（原对 GPU 张量
  逐元素 Python 迭代 = 隐式 .item() 同步）。
- 实现方式（杜绝手工拷贝 500+ 行方法漂移）：_derive_anchor_method ——
  inspect.getsource 读锚点方法文本 → 唯一文本替换（count≠1 fail-closed
  raise 拒载）→ dedent → exec 挂到 TLIIndexerFast。除替换点外与锚点逐字节
  同；锚点文本变更时拒绝加载（E127 派生契约 fail-closed）。super() 零参
  在 exec 函数中无 __class__ cell → 显式改写 TLIIndexer.compute_score
  （Fast 的 MRO 下语义等价）。
- 数字（S=64K normal 层）：compute_score 3.02→0.99ms（3.05×）；全步
  GPU sum 4.21→2.13（vs baseline **3.89×**），CPU wall 4.71→2.63
  （**3.66×**）。64K skip 层 GPU 1.93→1.02（4.16×）、wall 2.16→0.56
  （3.86×）。32K normal 2.75×GPU/2.62×wall、16K normal 2.62×/2.41×。
  8K 档复测持平零回归（dense 路由，噪声内）。
- 如实记录（运行方差，非行为变化）：本 iter 未触碰 eager_decoding 代码，
  mask 逐位一致门禁证明行为零变化，但 16K/32K/64K normal 层 eager 阶段
  GPU +0.1~0.3ms（0.25→0.37 量级）。64K 复测（GPU 空闲态）主 run 一致
  （2.13/2.63），eager 仍 ~0.37ms——判为环境/时钟状态方差（iter4 run 时
  卡上无并发负载），全步口径不受影响（3.89× 为复测确认值）。
- 门禁：15/15 PASS——原 12 项保持绿 + 新增 score_mask_derived（4 配置
  矩阵 champion_e64/singlepool_00/aavg_e64/aavg_00 × normal/skip × 10 步
  score_dict+mask 对 TLIIndexer 逐位一致（==）+ S 回退 rep 全量重建后续
  增量正确）+ pipeline_zero_sync（20 步 prepare+score+mask 全管线
  sync_debug_mode=error 零同步，normal+skip 双层）。既有回归 env=1
  复跑 7 套件全绿（near_swa 3/3+4/4、C3 5/5、E122 6/6、E125 尾巴 10/10、
  E125 kmeans 守卫 5/5、E116f 12/12）。
- commit：见 git log

### [iter 6] score_fine Triton GQA K 复用（compute_score 派生替换点②改写）
- 改动：sparse_attn/indexer/tli_indexer_fast.py——新 Triton kernel
  _score_fine_kernel（grid = (T/64, Hkv)，每 program 载 K 块 [64,128]
  bf16→fp32，寄存器内循环 g=4 个 q 头逐头点积，K 读带宽 [T,H,D]→
  [T,Hkv,D] 4× 削减）+ _fast_score_fine 方法（BT=64/num_warps=4，
  参数扫描实测最优；Triton 编译失败置 _triton_ok=False 回退原版
  repeat+einsum 表达式 fail-closed）。派生 _SCORE_REPL 替换点②扩为
  4 行整段（b_k_qat 物化 + 双分支 einsum → Triton 调用 + 三路回退）。
- iter5 _rep 增量缓存整体退役：Triton 直接读 k_qat 无需物化 [S,H,128]
  fp32，**省 1.07GB/层常驻显存**（S=64K，36 层生产 = 38GB）。
- 数字：score 阶段 64K 0.99→0.53ms（3.05×→5.7×，且全档平坦化：
  8K-64K 全部 0.48-0.53——microbench 原版 repeat+einsum 2.63ms vs
  Triton 0.106ms，einsum 本体 0.568 vs 0.106）。
- 【如实记录：launch gap 转移】全步 bench 口径持平（64K normal
  2.133→2.150）——score -0.46 被 mask（0.29→0.49）+ eager（0.37→0.60）
  +0.43 吃掉。隔离实验（同代码 _triton_ok on/off）+ profiler 实证：
  mask/eager 的 kernel 本体时间不变（0.250/0.254ms，47 kernel 完全
  同），差异全部是 launch 空转 gap——原版 3ms 大 gemv 给 CPU 提前
  排队的 lead time 掩盖了后续阶段 launch 开销；Triton 后 CPU/GPU
  并跑无余量，gap 浮出。**GPU busy（纯 kernel 时间）实降 0.46ms/层**；
  真实 36 层 decode 流水（CPU 每层 ~0.6ms << GPU kernel 每层 ~1.2ms，
  CPU 跨层跑在 GPU 前面）收益真实存在，单层 event 同步口径测不出。
  launch gap 是 iter7 主攻方向（mask 47 kernel + eager 15 kernel 融合）。
- 数值口径（allclose 任务容内，如实分级记录）：score_coarse/score_
  coarse_avg 未动逐位保持；score_fine fp32 累加顺序与 cublas 不同 →
  ULP 漂移，bf16 舍入后差异 48/10.5M（4.6e-6 比例，maxrel 6.8e-3 =
  1 bf16 ULP）；**mask/topk 选择 4 配置 × normal/skip × 10 步实测
  逐位不变**（经验验证口径非数学保证）；attention 输出 allclose(1e-2)
  门禁保持绿（worst 1.22e-04）。
- 门禁：15/15 PASS（score_fine 口径改 ULP 容差 ≤1e-4 比例 + 相对
  ≤2% + mask 仍逐位；S 回退测试同步改 ULP 口径）。既有回归 env=1
  复跑 7 套件全绿（near_swa 3/3+4/4、C3 5/5、E122 6/6、E125 尾巴
  10/10、E125 kmeans 5/5、E116f 12/12）。
- commit：见 git log

### [iter 7] eager gather Triton 融合（mask 展开 + gather online-softmax attention，~15 op → 2 kernel）
- 改动：sparse_attn/ops/eager_decoding.py（原型 exp/bench/_scratch_triton_eager.py
  6 配置验证后集成）——两个 Triton kernel 替代 torch gather 路径主段：
  ① _eager_expand_kernel（grid=(R,)，TP=1024/num_warps=8）：token 级
  bool mask → 每行紧凑选中索引 IDX [R,W_upper] + 计数 COUNT，chunk 循环 +
  running cumsum；iter3 三类契约校验（B7 空行=1 / W 超上界=2 / cu 破坏=3）
  原地点并入 kernel（atomic_max 全局错误标志），文案/语义逐码一致。
  ② _eager_attn_kernel（grid=(R*G,)，BW=256/num_warps=4）：program =
  (行×组内头) 展平，单头标量 online-softmax + acc[D]（规避 Triton 块内
  标量索引赋值），gather K/V 块循环；cnt clamp 到 W（与 topk 只取 W_upper
  同语义，漏选 flag=2 由 kernel ① 记录）。
  路由条件全 host 已知量（零同步判走）：T≥16K + block_size==1（TLI 生产
  mask 即 token 级）+ bool mask + D 2 幂 + 尾维 stride==1 + cu int32 +
  W_upper*8≤T；任一不满足或 Triton 异常 → 回退原 torch gather 路径
  （_EAGER_TRITON_OK 全局标志 fail-closed，本次即回退，值同 iter3 路径）。
  triton 缺失时模块仍可 import（try/except，env=0 生产零扰动）。
- 数学语义：softmax(全 S 行、未选位 -inf) ≡ softmax(选中子集)（exp(-inf)=0
  精确）；online-softmax 与 torch 全长 softmax 仅归约顺序不同 → ULP 级
  漂移，allclose(1e-2) 门禁（eager 组 worst 1.95e-03，容差内）。
- 数字【如实记录】：本 run 全步 bench 被外部容器负载污染无效——GPU0 被
  跨容器进程占满（82→95GB / 100% util，本容器 ps 不可见），所有阶段 GPU
  时间被排队延迟主导（prepare_index 8K-64K 全档恒 2.50ms vs iter6 增量
  0.46ms = 明显失真；8K eager 4.96 vs 2.2 = 3.5× 偏差）。干净卡隔离
  microbench（原型，S=16K-131072 × shared/per_qh）：Triton 融合
  0.10-0.29ms vs torch gather 0.41-0.66ms（2.7-4.6×）。
  **开发中抓到并修复的路由 bug（如实记录）**：初次集成的路由条件要求
  cu int32，而生产 prepare_cu_seqlens 的 `.cumsum(0)` 把 int32 tensor
  整数提升为 int64 → 生产/主 bench 全程 Triton 路径未生效（eager 阶段
  launch 实测 37 = torch gather；污染跑的 eager 0.24-0.26ms 实为 torch
  gather 数字，此前条目误读为「Triton 快 2.3×」——已更正）。gates 全绿
  是因测试用手工 int32 cu，掩盖了该 dtype 分歧（教训：**门禁输入形态须
  含生产构造路径**，已补生产形态 int64 cu 对拍：allclose maxdiff
  9.77e-04）。修复：helper 内确定性 cast int64→int32（2 元素 tiny
  kernel，值逐位同）+ 路由放宽接受 int64。修复后 eager 阶段 launch
  37→**6**（cu cast/full/qs mul/qs to/kernel A/kernel B）。
- 数字（GPU 空闲态复测定案，watcher 02:24 自动重跑覆写
  e127_stage_bench_iter7.json）：S=64K normal 全步 **1.73 GPU / 2.23 wall
  （vs baseline 4.79× / 4.31×）**；64K skip 5.18×/4.60×、32K normal
  2.69×/2.69×、16K normal 1.70×/1.75×。eager 阶段 16K-64K 全档
  0.55-0.60 → **0.186-0.191ms（3.2×）**且全档平坦（gather 本
  O(R·W·D) 与 S 无关，torch 路径的 launch 空转被 2-kernel 消除后真实
  kernel 时间浮出）；8K 档 dense 路由零回归（eager 0.516 不变、全步
  2.01 vs baseline 2.15 噪声内）。单步 CUDA launch 137（baseline）→
  **127**（eager 段 37→6）。vs iter5 全步口径 2.13→1.73（eager 单点
  -0.40ms 兑现）。
- 附带发现（只读分析，iter8 输入）：增量 prepare_index 真实口径 33
  launches 零 H2D/DtoH——此前探针测得 45+5 memcpy 系固定-S 重复调用
  误走全量路径的口径错误（与 iter6 profiler 教训同型）；全量版 anchors
  每步 torch.tensor(list) 构造 indices 的 pageable H2D 仅 prefill/回退
  触发。剩余 launch 大头：compute_mask 35（4×topk+3 排序+3 MeanOps+
  softmax 链）、compute_score 27、cu 构造 4（含 prepare_cu_seqlens 的
  pageable H2D ×1，gates 注释既知）、metrics 7。
- 门禁：15/15 PASS——eager 7/7（allclose 全配置 worst 1.95e-03 + B7 小
  序列双路径当场 raise + B7 大序列 gather 尾部 raise + W 超上界/cu 破坏
  尾部 raise + 零 host 同步 40 次调用零异常，python±-O 双跑同绿）；其余
  8 项（metrics 2 / indexer 3 / score_mask_derived 2 / pipeline_zero_sync
  1）保持绿。fail-closed 专项验证：注入 Triton 异常 → 回退 torch gather
  输出 vs dense allclose（maxdiff 4.88e-04）+ 标志持续 False（exp/bench/
  _scratch_fallback_check.py）+ 生产形态 int64 cu 对拍 allclose（maxdiff
  9.77e-04）。既有回归 env=1 复跑 7 套件全绿（near_swa
  3/3+4/4、C3 5/5、E122 6/6、E125 尾巴 10/10、E125 kmeans 守卫 5/5、
  E116f 12/12）。
- commit：见 git log

### [iter 8] compute_score coarse 段 Triton GQA K 复用（替换点③④：minmax 双点积单 kernel + avg 段复用 fine kernel）
- 改动：sparse_attn/indexer/tli_indexer_fast.py（锚点文件零改动）——新
  Triton kernel _score_coarse_kernel（grid=(kt/64, Hkv)，每 program 载
  k_min/k_max 块 [64,d']（min/max 独立 stride），循环 g=4 个 q 头做
  min(q,0)·k_min + max(q,0)·k_max 双点积——负部配 k_min、正部配 k_max，
  与原版 repeat 后 clamp 双 einsum 数学等价）+ _fast_score_coarse 方法；
  _SCORE_REPL 扩至 4 处替换（③ coarse minmax 段、④ score_coarse_avg 段
  ——einsum(b_q, repeat(k_avg)) 与 _score_fine_kernel 同构直接复用
  iter6 kernel）。门禁 test_score_mask_derived 口径更新：coarse/avg
  从逐位改 ULP（同 fine：比例 ≤1e-4 + 相对 ≤2%），mask 仍逐位。
- 原理：iter7 附带发现 compute_score 残余 27 launches 中 coarse 段
  gemv 本体仅 ~32μs，repeat(k_min/k_max/k_avg) 物化 + clamp + copy 胶水
  ~120μs 主导；且 coarse 维度是块级 kt=S/bs（64K 档仅 1024 行）——
  计算量极小、launch/物化开销占比更高的典型胶水瓶颈。GQA 组内
  k_min/k_max 块寄存器复用（读带宽 /g），[kt,H,d'] fp32 repeat 物化
  ×3（min/max/avg）全消。
- 数字（GPU 空闲态）：原型 microbench（kt=1024, d'=32）：torch
  coarse+avg 0.173ms → Triton 0.062ms。stage 64K normal
  compute_score 0.536→0.346ms（-0.19）；全步 1.73→**1.52 GPU /
  2.02 wall（vs baseline 5.47× / 4.76×）**。全档改善：16K normal
  1.88×/1.94×、32K 3.07×/3.01×、64K skip 5.92×/5.06×；**8K 档首次
  真提升 1.16×/1.15×**（2.01→1.86——8K eager 走 dense 路由但
  coarse Triton 不受 S 门槛限制，coarse 胶水 8K 亦在消费）。单步
  CUDA launch 127→**115**（compute_score 27→13）。
- 数值口径（如实分级）：门禁 4 配置 × normal/skip × 10 步
  score_coarse/coarse_avg **实测 ULP=0/289408（逐位一致，优于 ULP
  预期口径——块级 kt 小、fp32 树形归约与 cublas 在该尺度常同舍入）**；
  fine 48/10.5M 保持；mask 逐位不变；S 回退全量重建 ULP 口径绿。
  ULP 口径作为保守上界保留（更大 kt/不同配置可能出 1 ULP 漂移，
  原型隔离实测 D=128 大 kt 时 ~5e-5 比例差 1 ULP）。
- 下一步输入：compute_mask 现为最大阶段（64K normal 0.467ms、
  35 launches：4×topk + 3 排序 + 3 MeanOps + softmax 链）→ iter9
  主攻方向。
- 门禁：15/15 PASS（python±-O 双跑）；既有回归 env=1 复跑 7 套件
  全绿（near_swa 3/3+4/4、C3 5/5、E122 6/6、E125 尾巴 10/10、
  E125 kmeans 守卫 5/5、E116f 12/12）。
- commit：见 git log

### [iter 9] compute_mask 前段融合（gmean 双输出前移 + fused masked-softmax-gmean）
- 改动：sparse_attn/indexer/tli_indexer_fast.py（锚点文件零改动）——两方向：
  ① coarse/fine kernel gmean 双输出（_SCORE_REPL ③④ 扩 want_g=True：
  _score_coarse_kernel/_score_fine_kernel 加 OUTG+HAS_G constexpr，s_out
  先舍入 OUT dtype 再 fp32 顺序累加 /G = torch mean opmath 口径；⑤ 新
  替换点透传 score_coarse_g/score_coarse_avg_g 键）+ _MASK_REPL M-A/M-B
  消费点直供（形状守卫，无键/不符回退现场 rearrange+mean）——消
  compute_mask 内 2 处 rearrange 物化 + 2 MeanOps kernel。
  ② M-E fused mask-p（_mask_p_stat_kernel grid=(H,NC) 单头单 chunk
  chunk-local (m_c,l_c) + _mask_p_out_kernel grid=(Hkv,NC) 跨 chunk
  合并 + 循环 g 头重算，p = Σ_g exp(s−m_g)/l_g /G；swa 区直接置 1.0）
  替代共享口径六连（repeat+where+softmax+nan_to_num+mean+置位：6 kernel
  + 3 次全宽 [H,T] 物化 → 2 kernel）。exp 用 libdevice.exp（__nv_expf，
  与 PyTorch softmax 内部 expf 同实现）；mean 前 bf16 舍入口径对齐。
  路由条件全 host 已知量（NC≤64 → T≤64K），异常 _triton_ok=False 回退
  原段 fail-closed。
- 开发中抓到的两个 bug（如实记录）：
  ① ⑤ 替换点 _sc_g 只 unsqueeze 一次（3 维），M-A 守卫 dim()==4 恒假
  → gmean 全程回退现场 mean（mask 门禁照过掩盖了未生效），门禁新增
  gmean 对拍断言抓出（100% 形状失配）→ 补两次 unsqueeze 到 [1,1,Hkv,kt]。
  ② M-A/M-B 绑引用（非 clone）：compute_mask 下方对 score_coarse 的
  原地写（skip_far far 区 -inf / 末块置 +inf 排除）透传进 dict 的
  score_coarse_g 键——原版 mean 产物是独立新张量，dict 键永不被 mask
  阶段污染；直供版破坏该契约（最末块全 h_kv +inf，多步/带 mask 运行
  才显形，单步无 mask 不复现）。修复：消费点 clone（[1,1,Hkv,kt] tiny
  copy，换掉 rearrange 物化 + MeanOps 仍净赚）。教训：**Python 引用
  语义下「读侧直供」须审原版写路径的别名结构**。
- 数字（GPU 空闲态，复测一致）：64K normal 全步 1.516→1.560 GPU /
  2.02→2.06 wall（**vs iter8 净回归 -3%**；vs baseline 5.31×/4.67×）；
  分解：compute_score 0.346→0.370（gmean 双输出 +0.024 复测一致）、
  compute_mask 0.467→0.486（fused 生效但净 +0.019，launches 47→42）。
  16K normal 1.88×→**1.98×**（mask 0.468→0.423）；32K normal
  3.09×→3.10×；8K/skip 全档持平。**混合结果：16K 改善 / 64K 主档
  回归 / 32K/8K 噪声内——最优 iter 维持 iter8（1.52 GPU / 2.02 wall，
  5.47×）**。
- 分析（如实）：原型隔离 microbench fused p 0.085→0.066ms（-19μs）未
  传导到 64K 生产 stage 口径——launch-gap 主导下六连 kernel 本体被
  队列中其他 kernel 执行时间掩盖（overlap），融合省的是已被掩盖的
  gap；而 gmean 双输出的 fine/coarse kernel 特化（HAS_G 分支 +
  OUTG store）开销在 score 阶段显形。iter6「launch gap 转移」教训的
  反向情形：融合方向在小 S 档（16K）有效（CPU 排队压力下 gap 不被
  掩盖），大 S 档 kernel 本体占比高时融合收益被特化开销吃掉。
- 数值口径（门禁实测）：gmean 双输出 4 配置 × normal/skip × 10 步
  对 fast 分数现场 mean **逐位 0 差异**（0/289408 级）；fused p 原型
  口径对齐档 0 ULP / padding 档 4/524K≈7.6e-6；mask 全档逐位一致；
  fine 48/10.5M 保持。S 回退 ULP 口径绿。
- 门禁：15/15 PASS（python±-O 双跑；新增 gmean 断言进
  score_mask_derived）；既有回归 env=1 复跑 7 套件全绿（near_swa
  3/3+4/4、C3 5/5、E122 6/6、E125 尾巴 10/10、E125 kmeans 5/5、
  E116f 12/12）。
- commit：见 git log

### [iter 10] iter9 按档路由：gmean 双输出 + fused mask-p 仅 T ≤ 32768 启用
- 改动：sparse_attn/indexer/tli_indexer_fast.py——③④ want_g 改条件路由
  （kt*bs ≤ 32768 才双输出 gmean，否则单输出 = iter8 行为；_fast_score_*
  want_g=False 返回单值，调用处分支解包）+ M-E fused 路由上界 65536 → 32768。
  M-A/M-B 无需改（键不写 → get None → 回退现场 rearrange+mean）。
- 依据：iter9 实测收益谱系——16K normal +10%（CPU launch 排队压力下 gap
  不被掩盖，融合赚）/ 64K 主档 -3%（大 S 档 kernel 本体占比高，特化 +
  clone + Triton host launch 开销超收益）→ 按档取最优。
- 数字（如实，含未解疑点）：16K normal 1.511→1.457（vs i8 +3.7%、
  vs base 1.95×GPU/1.98×wall）兑现路由收益；**但 32K/64K 档回退路径
  后仍 -1.9~-2.8% vs iter8 定案**（64K 分解 score +17μs / mask +20μs，
  与 iter9 时 +43μs 几乎相同——回退 gmean/fused 后理论上应消失）。
  8K 持平（-1%）。**疑似环境/时钟漂移**：iter8 定案 1.516 为数日前
  GPU 空闲态测量，本晚 iter9/10 两次独立复测 1.556/1.560 一致——
  下一步用 iter8 代码原样复跑对照定论。
- 门禁：15/15 PASS（python±-O 双跑）；路由冒烟：16K gmean 键在 /
  64K 键无 + 双档 mask 对 TLIIndexer 逐位一致。
- commit：见 git log
