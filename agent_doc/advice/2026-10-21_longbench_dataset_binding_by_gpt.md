# LongBench评分数据集与sidecar绑定

问题ID：`TL-LB-DATASET-BINDING-20261021`；基线
`1342fe330d4d01b6c109a7053837cbf1040f6d67`。

触发：例如文件为`hotpotqa-tli-test.jsonl`，sidecar的dataset为qasper，
但result_file、SHA及行数均自洽。修前`eval.py`按文件名前缀选择scorer，
按sidecar的dataset分组输入身份，未核对两者，可能错集评分并绕开同集对比门。
不是SHA篡改：合法构建器也能生成这种错配sidecar。

修复：`benchmark/LongBench/eval.py:94-103`在读取并验证sidecar后、返回评分
元数据前核对producer的`dataset.split("-")[0]`命名契约；错配报
`DATASET-BINDING`。保留`repobench-p → repobench`兼容，无sidecar的显式legacy
流程仍按原规则处理。

测试执行`eval.py`中原样AST函数与真实标准库sidecar读写，排除numpy/评分器
等可选导入；没有mock身份校验或重写consumer。覆盖错集、legacy_allowed不能
绕过、预计算SHA/行数不能绕过及合法qasper/repobench命名。

```text
修前 python -m unittest tests.test_lbv1_dataset_binding -v
Ran 4 tests; FAILED (failures=3): AssertionError: SystemExit not raised
修后 python [-O] -m unittest tests.test_lbv1_dataset_binding \
  tests.test_lbv1_input_uniformity tests.test_lbv1_input_identity_v2 \
  tests.test_backfill_identity_format
普通 Ran 26 tests in 0.064s — OK
-O   Ran 26 tests in 0.150s — OK
compileall、git diff --check 通过
```

环境：Python 3.12.14，Linux 6.18.55 x86_64，CPU；无torch/jieba/GPU及相应
驱动/编译器/kernel配置。fixture脚本SHA-256：
`4689110098e01ac13312a7e046c43c2277bcb5ca00820237d10d06cafb0b46a5`。
没有完整评分CLI、真实模型质量/TTFT/显存/kernel数/速度测量，不主张性能提升。
历史预测不改写；需在生产依赖可用后重评错配目录及同数据集混挂对比，
当前无法判断是否影响历史87格或冠军排序。
