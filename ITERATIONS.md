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

## Iter 2 — 128K L1 512-wide partial + exact merge

- 源码基点：`origin/main=8807738ffd6e3fc32385a7c6da4e223de9ee1a99`；正式基线测量快照为 `11695f98d203cb553ce3a78597f365d0ed64d72a`。
- 问题：真实 Qwen3-8B、RULER 128K `niah_multikey_1` sample 0 中，原整行 L1 Top-K 在 36 层均稳定回退；Graph/eager L1 分别为 2.380720/2.568628 ms，NFA 纯算子总耗时为 5.624544/5.882530 ms。
- 修改：`KT>512` 时按 512 项并行生成 signed packed-int32 局部 Top-K，再以有序 run merge 精确保留全局 Top-K 和 tie 顺序；短上下文保留原 mono kernel，far/near 两路仍共用 launch。
- CUDA 对拍：`tests/ops/test_l1_topk_tiled_cuda.py` 覆盖 7 组形状、同分、NaN/±Inf、非整块长度及 K=0/1 边界，far/near 索引与顺序逐元素一致，1/1 通过。
- 真实 Qwen niah 128K 纯 kernel（3 warmup、5 repeat，Graph/eager 顺序 direct run）：
  - Graph：L1 0.764162 ms，L2 2.231276 ms，Attention 1.036390 ms，总计 4.031538 ms；相对基线总耗时加速 1.395×，同批 FlashInfer 基线 10.875715 ms，对应 2.698×。
  - eager：L1 0.902237 ms，L2 2.305446 ms，Attention 1.023746 ms，总计 4.229929 ms；相对基线总耗时加速 1.391×，同批 FlashInfer 基线 10.876654 ms，对应 2.571×。
- 口径限制：候选 direct run 无旁路监控，不冒充 hygiene-clean 正式 ABBA 臂；纯 operator chain 不是完整 decode TPOT。真实 selected IDs、生成 token identity、其余长度回归与干净 ABBA 尚未验收。
- 后续硬门：真实 128K 512-token 验收中 Graph 与基线完全一致，但 eager 在第 36 个生成位置首次分叉；代码审查确认 fused eager 将 int32 partial 指针误传给后续 BF16 candidate completion，Graph 拆分路径不受此影响。

## Iter 3 — 分离 eager Top-K partial 与原始分数指针

- 修改：`_topk_candidate_batched_kernel` 显式接收 `FAR_TOPK/NEAR_TOPK` 与 `FAR_SCORES/NEAR_SCORES` 两组指针；前者用于 mono/tiled Top-K，后者仅供 candidate completion 读取，非 tiled 路径仍传同一 BF16 缓冲。
- 回归：新增直接调用 fused candidate kernel 的 mono/tiled 对拍，除 `i_f/i_n` 外覆盖 candidate IDs、valid mask、lower/upper/quota、owners 和 generations；该用例覆盖 Iter 2 独立 merge 测试遗漏的指针语义。
- CUDA 结果：`tests/ops/test_l1_topk_tiled_cuda.py` 2/2 通过（含原 7 组索引/顺序用例与新 fused 用例）。
- 下一步：重跑真实 128K eager/Graph 512-token 四方身份与分叉步 selected IDs，再测修复后的纯 kernel 性能；Iter 2 性能数字不得直接当作 Iter 3 正式 verdict。
