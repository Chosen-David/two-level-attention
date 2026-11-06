# NFA 64-bit Top-K 优化记录

## Baseline

- 源码：`origin/main=fe941027c95c8ca3646baebf1c37aedbdd8e04f8`。
- 生产形状：B1、HKV=8、R=8452、K=1024、36 层；capacity=65728 时使用 int64 mono Top-K。
- 原测试把 BF16 logits 误传给 packed-int32 partial 路径；先修正测试输入，再参数化 16384/65728/131520，与保留的 mono kernel 对拍。
- 修正后的 baseline：16K 通过，64K/128K 因尚未实现 int64 tiled 输入而按预期失败。

## Iter 1 — int64 partial + merge

- 修改：fine-score 直接生成 int32/int64 packed key；两种 token 域统一使用 2048-wide partial 与精确 merge；mono kernel 保留为 reference。
- 正确性：16K/64K/128K 三档 indices/counts 与 mono 逐元素一致，覆盖同分、同 token、NaN/Inf、quota=0 和补洞，3/3 通过。
- microbench：capacity=65728，36 次调用/repeat，10 repeats，3 warmup。
  - mono median：12.277328 ms。
  - tiled median：1.924848 ms。
  - mono/tiled：6.378336×。
- 口径限制：该计时隔离 Top-K，candidate 输入 packed key 已物化；不包含 fine-score pack 的增量成本，不是完整 selector 或 decode TPOT verdict。
- 下一步：完整 selector、Graph/eager token identity 与生产 decode 配对验收。

## Iter 2 — strict-skip 上移植 Graph-only L1 分块 Top-K

- 日期：2026-11-07。
- 基线：`eb4677a2b817bb18f53b2edd57f7d54673cac10f`（strict-skip）。
- 候选来源：`8807738f..e1e58733`，仅移植 selector 与 CUDA 对拍测试。
- 修改：`KT > 512` 且并行 candidate expansion 时，以 512-wide partial Top-K 加精确归并替代整行单 CTA Top-K；eager 保留原 fused kernel。原始 BF16 分数与 int32 partial 使用独立指针。
- 正确性：pc40 GPU1，`tests/ops/test_l1_topk_tiled_cuda.py -v` 2/2 通过，7 组索引子测试及 fused candidate 全输出逐位一致；覆盖 ties、NaN、±Inf、非整块、K=0/1。
- 性能：B1、原生 RULER `niah_multikey_1` sample 0、512 new tokens、3 warmup、独立进程 ABBA；baseline=`eb4677a2`，candidate=`53ec0888`，8 runs/格均完成且 `gpu_hygiene=clean`，repeat 与 baseline/candidate token identity 全部通过。
  - 32K eager：26.868000 → 26.739632 ms，1.004801×（far=2）。
  - 32K Graph：9.856648 → 9.953856 ms，0.990234×（far=2，负面格子）。
  - 64K eager：26.448864 → 26.605808 ms，0.994101×（near=3，负面格子）。
  - 64K Graph：10.725512 → 10.407408 ms，1.030565×（near=3）。
  - 128K eager：26.667720 → 26.345000 ms，1.012250×（near=13）。
  - 128K Graph：12.055552 → 10.441184 ms，1.154615×（near=13）。
- 口径：128K Graph 收益在当前 strict 基线复现；32K Graph 与 64K eager 为负面结果，不宣称全长度加速。
- 下一步：与 L2 load 前过滤、L1 按需统计组合后重新做身份门禁与完整 decode ABBA。

## Iter 3 — L2 middle 候选在物理页与低维 K 读取前过滤

- 日期：2026-11-07。
- 基线：`53ec08882091ccfb52d3a4d9f1b50f562c89e6ab`。
- 修改：将已有的 `unique + LOW/HIGH + quota + 页表有效` 条件合并为 `load_r`，前移为物理页表和低维 K 的 load mask；protected、越界、非 owner 与 quota=0 候选不再先读 K 再丢弃。NaN 仍在打分后按原规则排除，BF16 舍入、packed key、同分顺序和补洞逻辑不变。
- 正确性：pc40 GPU1，`tests/ops/test_fine_scores_preload_mask_cuda.py -v` 1/1 通过；修改前独立 oracle 与候选 packed logits 逐元素一致。将被丢弃候选的物理页改为 `0x3FFFFFF` 后输出不变，quota=0 且整表非法时也不读 K。首次运行发现测试把 CPU keys 地址写入设备指针表，修正为直接分配设备 keys 后通过；该失败不是候选 kernel 回归。
- CPU 回归：候选修改后 strict 套件 99 passed、5 subtests passed。
- 性能：本轮尚未测量，不宣称加速。
- 下一步：真实模型 eager/Graph 身份门禁、纯算子与完整 decode ABBA。

## Iter 4 — L1 按实际候选池读取 Min/Max/Sum

- 日期：2026-11-07。
- 基线：`8db999fc07b74916db5965c298509744f572325e`。
- 修改：Min/Max 仅由使用 MinMax 的 far/near 候选池读取，Sum 仅由使用 Avg 的候选池读取；skip、bypass 和保护区在 load 前排除。两种区域都为 Avg 时在编译期删除 MinMax load 与点积，其余组合保持原 BF16 中间舍入和 GQA 累加顺序。
- 正确性：pc31 GPU1，`tests/ops/test_l1_scores_demand_load_cuda.py -v` 1/1 通过；四种 far/near 方法、四态 skip、非对齐 sink/SWA、不等长和非连续 request slot 的全部输出逐位一致，覆盖 NaN/Inf。both-Avg 使用单元素 Min/Max 存储配合原大 stride，未发生越界且输出不变。
- CPU 回归：`test_strict_skip_receipt.py` 7/7 通过。
- 性能：本轮尚未测量，不宣称加速。
- 下一步：合并候选做真实模型身份、纯算子和完整 decode ABBA。

## Iter 5 — skipped L1 池跳过 Top-K 局部排序与归并

- 日期：2026-11-07。
- 基线：`f313e725d2ac1d46e60450c5462554bc37185967`（L1 tiled + L2 load 前过滤 + L1 按需统计）。
- 修改：从设备 skip flags 推导每行 far/near gate；skipped pool 的 partial kernel 不读取 score、不执行局部排序，candidate Top-K 不读取 partial、不执行归并，直接生成与原全 `-inf` packed-key Top-K 相同的 `0..K-1` 索引。eager fused 与 Graph parallel prepare 共用 `_l1_candidate_topk_body`；非 skip 路径、候选展开及最终集合语义不变。
- 正确性：CPU strict/cache 回归 18 passed、5 subtests passed；pc40 GPU1 `tests/ops/test_l1_topk_tiled_cuda.py -v` 最终 5/5 通过，覆盖四态 gate、tiled/non-tiled、非连续 request slot、非零 layer offset、skipped K=0/1/K=KT、ties/NaN/±Inf、单元素 skipped score/partial 不读取，以及 fused 和 prepare+生产 expansion 的全输出逐位一致。独立静态审查未发现可达正确性问题。
- 真实身份：B1/B4 eager/Graph strict oracle 均 `valid=true`、`violations=[]`、`gpu_hygiene=clean`；B1 token SHA256=`c8e3b877…64c204`，B4=`3ac6583a…b5738`，候选与 `f313e725` 基线逐 token 一致，且 eager/Graph 一致。
- 纯算子性能：pc40 GPU1，B1、原生 RULER 128K `niah_multikey_1` sample 0、3 warmup、5 repeats、独立进程 ABBA；8/8 run returncode=0，`gpu_hygiene=clean`，baseline/candidate source receipt 分别为 `f313e725`/`7f18437a` 且 diff SHA256 均为空。输入 SHA256=`4078e45e…24eaec15dab5`，两臂均为 13 个 near-skip 层、0 个 far-skip/both-skip 层。数值为两臂 median 的 median，单位 ms/step：
  - eager L1：2.559130 → 2.264135，1.130291×；L2：2.239222 → 2.234864，1.001950×；main Attention：1.027152 → 1.027873，0.999298×；exactly-once 合计：5.825114 → 5.526965，1.053945×。
  - Graph L1：0.754053 → 0.737982，1.021776×；L2：2.193801 → 2.197059，0.998517×；main Attention：1.019745 → 1.023396，0.996433×；exactly-once 合计：3.968891 → 3.958040，1.002741×。
  - eager `_topk_candidate_batched_kernel`：2.189653 → 1.895574 ms，减少 0.294080 ms；Graph `_l1_topk_partial_kernel`：0.164160 → 0.156481 ms，`_topk_candidate_prepare_kernel`：0.154544 → 0.145777 ms。Graph 总收益较小，不宣称全 selector 或完整 decode 同比例加速。
- 静态成本：tiled partial workspace 为 `m*HKV*NTILES*(K_F_PAD+K_N_PAD)*4` bytes；B1、HKV=8、K1=128、near=32 时，64K 为 15360 bytes、128K 为 25600 bytes，B4 分别为 61440/102400 bytes，16K 不启用 tiled。Graph 每层较原 parallel prepare+expand 多一次 partial launch，eager launch 数不变；均为形状推导，不是独立显存实测。
- 口径：仅报告 L1/L2/main Attention 纯 GPU 算子，不包含其他模型工作，不是完整 TPOT verdict；L2 与 Attention 的微小变化不作收益宣称。
- 下一步：非 skip near Top-K 仍扫描全 KT；改为只对动态 near 页区间做固定宽度精确 Top-K，并保留全局 `-inf`/NaN/tie 补洞语义，随后重做身份、CUDA 对拍和完整 decode ABBA。

## Iter 6 — 非 skip near 固定区间 Top-K

- 日期：2026-11-07。
- 基线：`333af802`（Iter 5 正式算子结果）。
- 修改：从 `LENGTHS` 计算每行动态 near 页区间，仅对该固定配置宽度区间做 Top-K；再与原 near 池外的前 `K` 个全 `-inf` 位置做精确二路归并。全局 packed index、NaN/±Inf/signed zero、同分逆序索引和补洞规则不变。`near_start<K` 或固定区间未完整覆盖 near 池时逐行回退原整行/tiled 路径。eager 直接输出索引；Graph 仅在 tile 0 物化 near 最终 packed run，其余 near tiles 不再排序，far 路径不变。
- 正确性：pc40 GPU1，`tests/ops/test_l1_topk_tiled_cuda.py -v` 5/5 通过。四态 gate 用例提升为 KT=2055、K1=128、near=32 的 128K 生产形状，覆盖非连续 request slot、非零 layer offset、长行紧缩、短行回退、ties、正/负 NaN、±Inf、signed zero、全 `-inf` 补洞，以及 eager、tiled、prepare+生产 expansion 全输出逐位一致。通用非生产掩码用例显式保留原路径，避免以池外随机分数违反紧凑算法前提。
- CPU 回归：strict/cache 18 passed、5 subtests passed；扩展 selector/strict 套件 99 passed、5 subtests passed。
- 性能：本轮尚未测量，不宣称加速。
- 下一步：独立静态复核、B1/B4 eager/Graph 真实 token identity、128K 纯算子 ABBA 与 32K/64K/128K 完整 decode ABBA。
