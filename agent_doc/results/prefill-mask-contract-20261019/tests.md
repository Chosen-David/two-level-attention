# CPU 红绿验证摘要

基线：3b0cfea1dac5598b44bfa28818b020c5a7e61227。每条命令由 timeout 120 限制，OMP_NUM_THREADS=1。此处耗时仅原始 unittest 日志，不是修前修后性能基准。

## 原始短输出

```text
BEFORE (new test before implementation, 17 invalid cases x 3 routes):
Ran 3 tests in 1.275s
FAILED (failures=1, errors=51)

AFTER (18 invalid cases x 3 routes, 24 valid subcases, actual singleton case):
Ran 3 tests in 0.778s
OK

FINAL OPS (normal; includes thread-state cleanup in new test):
Ran 84 tests in 2.800s
OK
FINAL OPS (-O):
Ran 84 tests in 2.289s
OK

ADJACENT (normal):
Ran 23 tests in 77.160s
OK
ADJACENT (-O):
Ran 23 tests in 83.425s
OK
```

完整命令：

```sh
OMP_NUM_THREADS=1 timeout 120 python -m unittest sparse_attn.ops.tests.test_prefill_mask_contract -v
OMP_NUM_THREADS=1 timeout 120 python -m unittest discover -s sparse_attn/ops/tests -v
OMP_NUM_THREADS=1 timeout 120 python -O -m unittest discover -s sparse_attn/ops/tests -v
OMP_NUM_THREADS=1 timeout 120 python -m unittest tests.test_e151c_prefill_routing tests.test_e139_graph_contract tests.test_e139_shared_gqa_indices -v
OMP_NUM_THREADS=1 timeout 120 python -O -m unittest tests.test_e151c_prefill_routing tests.test_e139_graph_contract tests.test_e139_shared_gqa_indices -v
```

## SHA256（原始日志留本地检查点）

| 文件 | SHA256 |
|---|---|
| before.log | c9a93820e4d8dbe8e467ae88238ec1fa8d122a88702f025b5310556e2c02bd03 |
| after.log | 1398c9f5103287b9fcac2bd53723d872132e59250fd78e8a90013286377f0772 |
| ops.log | c6be65640214a45be5ae1bd25ce6cca6281b647484bf94f22b2b376e0f145a45 |
| ops_optimized.log | d135b371b005739307c6598b22258c78187a65baadbbcee0be2302e5c2cb7dd9 |
| adjacent.log | e89340473ede81fbabcf9eb1a85fc4c5ccefe725ebdbfc9c14d5847cb5c10fa8 |
| adjacent_optimized.log | 3fcedf44a8eb6eb18de5f4940bd3aa91d027010a8c32406bb884bd3cf80fce7a |

新增测试源码SHA256：0fb806f1dd9202bdf450f03a0c735c381059c60d748d8057eefcb986ae85b977。测试用人工构造张量，无模型/私密原始数据；独立随机输入构造、输入hash和oracle见 verify.py / independent_review.md。

环境：Python3.12.14、torch2.8.0+cpu、Linux6.18.55、Xeon Platinum8573C、CUDA不可用。相邻E139测试自身打印的CPU allocation proxy是历史功能回归输出，不是本补丁的性能收益。
