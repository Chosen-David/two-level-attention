# TL-E155D-SCORE-UNITS-20261020 验证

日期：2026-10-20。基线：4ea101c9eca2ef9c4a852db7a5dfbd04db061c6b。
独立干净 worktree，仓库原检出中的无关未跟踪字体临时文件保留不动。
远端 main 重新 fetch；实验分支 e138-ablation=2ed6a1f，prefill compact
分支=93e4a364，仅理解/对照，未合并其代码或引用其结果为 main 验证。

## 契约与计划

沿用 research-implement-optimize 技能的先复现、最小修改、差分验证流程。
本次有界工作属于根 TASK T2：CI 输出应采用 AVG5 分制，保留随机采样和
统计判决；不放宽配额，不触及 GPU。成功标准为最小反例修复且合法采样
结果仅发生预期单位换算。测试先于生产源码修改编写、执行。
原论文/官方上游 eval.py 网页查询返回 DisabledError，未据其作新增论断；
直接以本库实际执行的 eval.py:240–255 作为量纲参照，不宣称调研最新算法。

## 修前实际短日志

`python -m unittest discover -s tests -p test_e155d_score_units.py -v`

```text
test_constant_negative_difference_in_score_points ... FAIL (-0.5 != -50)
test_constant_positive_difference_in_score_points ... FAIL (0.25 != 25)
test_main_verdict_ranking_and_ci_share_scale ... FAIL (0.25 != 25)
test_mixed_differences_interval_still_includes_zero ... ok
test_scale_before_rounding_preserves_small_difference ... FAIL (0.0 != 0.004)
test_unequal_task_sizes_keep_macro_not_micro_average ... FAIL (0.2 != 20)
test_zero_interval_not_significant ... ok
Ran 7 tests in 0.009s
FAILED (failures=5)
```

## 修后实际短日志

```text
python -m unittest discover -s tests -p 'test_e155d*.py'
Ran 32 tests in 0.039s
OK
python -O -m unittest discover -s tests -p 'test_e155d*.py'
Ran 32 tests in 0.031s
OK
python -m unittest discover -s tests -p 'test_explicit*.py'
Ran 17 tests in 0.149s
OK
python tests/check_e155d_score_units.py
PASS: 24 baseline differential cases; actual aggregate scorer scale
python -O tests/check_e155d_score_units.py
PASS: 24 baseline differential cases; actual aggregate scorer scale
git diff --check
[exit 0]
```

检查脚本从固定基线 git 对象读取旧实现，只截取舍入前输出，然后核对新
输出的量纲转换；4 种样本规模 × 3 个 seed × 两种 ID 类型共 24 组，含
任务间不同样本数、反向字典顺序。不是另一个 Agent 的独立上下文复核。
真实 aggregate scorer AST 用数值 metric 替身，与合成恒定差用例对齐。
main 测试的收集与评分是替身，不等于真实 LongBench 全链路通过。

## 环境与身份

Python 3.12.14，Linux 6.18.55 x86_64 glibc 2.39；可见 CPU 型号
AMD EPYC 9V74（9 vCPU）。无 Torch、Triton、nvidia-smi；无 GPU、驱动、
CUDA 编译器或 kernel 配置的运行证据。仅标准库；耗时是测试时长，不是性能基准。

输入生成器/校验 SHA256：

- tests/test_e155d_score_units.py：`6f77cd90a007ce74cf1d7d17b1eaad594654da58ec66f9464da6db30d1965424`
- tests/check_e155d_score_units.py：`66d44da290e452f056c5a9ba19f1d902169e346f299f91b03235a3137273f8a2`
- 修改后 exp/trace/analyze_e155d_grid.py：`8ecb84747d918fff2ddcbfb46fa9072b1ff1261f6aca46368e9ac818446ef647`

历史预测/JSON 未修改。新单位元数据和数值是有意接口修正；外部消费者应
读取单位，旧无标签产物不得默认为新量纲。GPU/性能/质量与真实 T2 重评未验。
