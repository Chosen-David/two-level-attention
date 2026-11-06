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
- 性能：本轮尚未测量，不引用旧分支加速比。
- 下一步：strict 回归、真实模型 eager/Graph 身份门禁与 32K/64K/128K ABBA。
