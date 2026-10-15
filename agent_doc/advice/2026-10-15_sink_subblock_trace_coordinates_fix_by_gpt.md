# 子块 sink 的 trace 页坐标修正

## 范围与结论

- 基线 main：`2014eeb479c748e2a430ad0cafe17bbfb2b12f53`；相较上次 `366fd384ebd18c9caf294f9114be144bb2040fb4`，新增 sink4 构造许可、E135 扩展臂和 E148b mass 探针。实验分支 `e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，只读对照，未合并。
- **TL-TRACE-SINK-PAGE-ORIGIN-20261015 / P2 / confirmed → fixed，CPU rechecked**：两个 trace 分析器的 L1 页区间判断与实际 token gather 使用不同起点，影响非块对齐 sink。修复实验实现，不修改生产 indexer、kernel、预算、chunk2048、guide 或历史 JSON。
- sink4 生产许可并不使 `batch_select._gate_ok` 误派发：其 `sink_tokens % bs == 0` 条件仍会回退。这个假设已排除，不作为缺陷报告。

## 根因与实际触发

位置：`exp/trace/analyze_e135_sinkswa_grid.py` 的 `eval_layer → eval_sinkswa → select_pages`（修后约 111–128 行）；同样代码存在于 `exp/trace/analyze_e148_sink4_grid.py:116–133`。`analyze_e135_sink_ext.py` 调用前者，因此继承问题。

页分数切片从 `floor(sink / BS)` 页开始，实际 gather 也以 `floor(sink / BS) * BS` 为起点。但修前重叠判断使用 `sink + bi * BS`，把每页向右平移 `sink % BS` 个 token。块对齐时差为零，所以 E98 的 sink128 锚点通过不能验证 sink4/16。

最小例：`BS=64, sink=4, mid_len=4096, far=[0,3072), n_pages=1`，2 heads、80 页，score 为页号、页48改为1000，无并列分数。真实页48绝对区间 `[3072,3136)` 与 far 绝对区间 `[4,3076)` 相交4个 token，应选 mid 坐标 `3068..3071`。旧判断把页48误认为起于3076，剔除该页，转选页47的 mid 坐标 `3004..3067`。这不是允许浮点误差或 topk ties，而是候选集合语义错误。

修复：明确 `page_start = (pool_lo_blk + bi) * BS`，以半开区间重叠条件 `page_start < sink+hi` 且 `page_start+BS > sink+lo` 筛选页。其余量化、评分、L2、预算、补位逻辑保持原样。本测试不证明这些其他逻辑全部正确。

## 复现与验证

新增 `exp/trace/test_sink_page_coordinates.py`：AST 提取仓库实际嵌套函数，不是重写被测算法；参考实现逐个枚举绝对 token，再独立按无并列页分数选页。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python exp/trace/test_sink_page_coordinates.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O exp/trace/test_sink_page_coordinates.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests
```

执行环境附加 `PYTHONPATH=/tmp/twolevel-cpu-deps:.`，Python3.12.14、x86_64、PyTorch2.8.0+cpu；`torch.cuda.is_available() == False`，CUDA/驱动/Triton/kernel 配置不适用。未使用远端训练资源。`/tmp/trace/qwen3-8b` 不存在。

原始短日志：

```text
baseline: Ran 2 tests; FAILED (failures=54)
fixed:    Ran 2 tests in 0.220s; OK
fixed -O: Ran 2 tests in 0.344s; OK
ops:      Ran 34 tests in 2.225s; OK
ops -O:   Ran 34 tests in 1.287s; OK
aligned_legacy_parity_cases=192
```

两项新测试共258个子项，包含 sink0/4/16/63/64/65/127/128、far/near、空区间、单 token 和跨页尾块、2 heads、minmax/avg 路径。另从 `git show HEAD` 提取修前函数，比较 sink0/64/128、4个区间、n_pages0/1/2/8、两种评分路径，192例逐位相同。输入分数张量 float32 连续字节 SHA256：`fd95fc31f29dc3a6200b739d2f56f47db96ba3cc0d5a44392507298522f4e5ff`；其余输入参数如上。测试耗时不是性能 benchmark。

## 历史数据与重跑边界

保留 `e135_sinkswa_grid.json`、`e135_sink_ext.json`、`e148_sink4_massgrid.json` 原始字节；重跑默认写入对应 `_v2.json`，记录 `page_coordinate_protocol=absolute_page_origin_v2`。三个输出路径已实际 import 核验，未运行真实 trace、未产生替代分数。

优先重跑所有 sink4/16 等非对齐臂，及其 sink128 对照和 sink128/SWA1024 锚点；使用相同样本、行集、预算、输入 hash。纯 sinkswa_only 归因掩码不经过该函数，未直接受此坐标错误影响；champion mass 及二者差值需要核验。不能继续仅凭已通过的对齐锚点认定旧 sink4 数值可信，也不能据本次合成反例宣称真实质量一定降低多少。旧“-0.28pt”等数值仍是历史输出，不是修正后结果。

E148 的 SWA124 是 trace 探针，当前生产构造仍要求 SWA 块对齐。本修复未把该 trace 协议等同生产 K1/K2/GQA/L2 语义，也未验证模型质量、prefill/decode 状态交接或端到端速度。

## 后续

补充真实 trace 的 v2 对照，确认 sink4 采用决策的证据。生产 GPU 性能、compact-block 分支、overlap 和 MLA decode padding/bias 契约仍未完成；本轮不声称提速、不合入 GPU 未验证改动。实现优化技能本轮用于限定可验证的 CPU 实验修复，并把运行时间与性能证据分开。
