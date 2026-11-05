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
- 正式 128K 512-token 四方 ABBA：Graph candidate/baseline=10.438656/12.083000 ms，1.157525×；eager=26.715592/26.196760 ms，0.980579×。全部 repeat/cross/eager-Graph token identity 通过，gpu_hygiene=clean；eager 回退另由 host/device profile 定位为每层多一次 launch，而非 device kernel 变慢。

## Iter 4 — eager 保留单 kernel，Graph 保留 tiled partial + merge

- 修改：`_candidate_pack_batched` 仅在 `kt > 512 and parallel_expand` 时启用 tiled Top-K；eager 恢复原单 fused kernel，Graph 继续并行 partial + exact merge。packed key、K、tie 规则和选中集合不变。
- CUDA 对拍：pc31 GPU1，`tests/ops/test_l1_topk_tiled_cuda.py` 2/2 通过，覆盖 7 组索引/顺序形状及 fused candidate 全输出对拍。
- 真实 Qwen3-8B、原生 RULER 128K `niah_multikey_1` sample 0、B1、512-token、3 warmup、完整 `GraphRunner.step` 四方 ABBA，pc31 GPU1，八腿均 clean：
  - Graph：candidate 两臂 10.516752/10.508272 ms，baseline 12.159392/12.163376 ms；臂中位数再取中位数为 10.512512/12.161384 ms，baseline/candidate=**1.156849×**。
  - eager：candidate 26.528576/25.945504 ms，baseline 29.247456/26.468400 ms；对应中位数 26.237040/27.857928 ms，baseline/candidate=**1.061779×**。baseline b1 明显偏高，故该比值如实保留但不宣称稳定 6.2% 收益；候选两臂及典型 baseline 约 26 ms，Iter 3 的 eager 回退已消除。
  - Graph/eager 的 candidate repeat、baseline repeat、candidate/baseline cross、两 variant eager/Graph token identity 全部为 true；输入 SHA256=`4078e45e1b76f5fd31a92884ab6d1c20bacd5647464e2e5bed7c24eaec15dab5`，自然触发 13 层 near skip。
- 证据：pc31 `/tmp/nfa-l1-iter4-e2e-results/summary.json`、`source_identities.json` 与八腿原始 JSON/log；候选 HEAD=`9a78d812ccd5fd3f61a891d8dccd6b6f0afdcb3f`，工作 diff SHA256=`bc39cf4a06da99e4c55843ed9d9b84cd4aca7ca1052f6afb4ac87864ee0fc456`。
- 口径限制：这是相对 Iter 3 基线的 candidate/baseline 验收，不是相对 FA2 的 2× verdict；Graph 的 device 收益不得记作 eager device 收益。
