# E141 验收器有限值门与 aggregate far 修正

范围：在 dev `12ffa8e5e01bcebff6c8886b9b55031d0f8456a6` 上只修验收器和
E141 指标命名，不改算法，不执行实际采集，不合 main。

确认缺陷：`max(previous, float(NaN))` 可以保留 previous=0，使概率误差看似
为零。现在先检查 got/ref 两个概率 tensor 全部有限，再计算并显式检查每行
绝对误差有限，最后才归并 max；最终 parity 对所有误差同时要求 finite 和
阈值内。检查采用显式 ValueError/math.isfinite，在 `python -O` 下不消失。

E141 把 per-head 概率先平均再传入 E140 O1，只有一个聚合行，不是 MLA 最坏
head。因此 E141 输出改用 `e141-o1-head-mean-v2` schema，移除
`far_max_kv_group_mass`，使用 `aggregate_far_mass` 并注明聚合口径。其值等于
已有 `mass_far`；E140 多 KV-group 接口及旧历史输出均不改写。

验证：普通/-O 各通过 8 项 E141 测试与 8 项 E140 accounting 测试。
其中 got/ref 各注入 NaN、+Inf、-Inf，共六个分支经过实际 verifier main，
均退出 1 且不输出 parity=true；有限 surrogate 仍退出 0。另两个子进程直接
测试 helper：有限 fp32 相减溢出为 Inf，以及隔离注入 NaN 行误差，均退出 1。
这些行误差测试不冒称完整 main 注入。测试驱动可重复构造反例，不依赖临时
mutation 文件。独立只读审查见证据目录 review.md。

证据：`exp/paper/evidence/e141-finite-gate-20261015/`；测试入口：

```bash
PYTHONPATH=/tmp/tla-cpu-deps OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_e141*.py' -v
PYTHONPATH=/tmp/tla-cpu-deps OMP_NUM_THREADS=1 python -O -m unittest discover -s tests -p 'test_e141*.py' -v
```

测试沿用已有本地 `040246…` 替代源码。旧有限 surrogate 数字和原始证据保持
有效范围，不因这次非有限输入门缺陷而作废；新正例再次确认其有限误差结果。
官方 `7d8e…` 源码、YaRN、完整模型、bf16、预训练身份及 GPU 仍未验收。本轮
没有新增下载、GPU、付费、Claude、SGLang、E142 或 MLA 算法研发。
