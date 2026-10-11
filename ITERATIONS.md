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
