# TL-E155D-PAIRING-20261020 验证记录

基线：a46ebc32443d9a14259b46a4047155622193d14e。2026-10-20。

## 计划与边界

按 research-implement-optimize 的复现→修复→回归→独立复核流程，验证固定配额
T2 top-N 配对样本身份。只改汇总器，不改 nb/nt/nl、推理、kernel、历史预测或
人类指南。CLI正常合法输入保持既有bootstrap算法/seed/输出量纲；非法样本
不得以行号或子集替代。当前无Torch/nvidia-smi，使用Python3.12.14标准库。

## 修前短日志（最初9项测试，生产源码未修改时）

命令：`python -m unittest discover -s tests -p test_e155d_pairing.py -v`

```text
test_disjoint_sets_rejected_with_diagnostic ... ERROR
test_duplicate_id_rejected_instead_of_overwrite ... FAIL
test_empty_sets_rejected_with_diagnostic ... ERROR
test_matching_ids_preserve_seeded_result_and_order_invariance ... ok
test_missing_id_rejected_instead_of_row_number ... FAIL
test_null_id_rejected ... FAIL
test_partial_overlap_is_rejected_not_silently_intersected ... FAIL
test_single_task_mismatch_rejected ... FAIL
test_valid_row_order_does_not_change_identity ... ok
Ran 9 tests in 0.007s
FAILED (failures=5, errors=2)
```

两处ERROR是ZeroDivisionError，而非可定位的身份拒绝。最小样本每任务
A={shared:0.5,a:0}，B={shared:0,b:1}：完整均值差-0.25，旧交集差+0.5。
这是合成输入反例，不是某个实际网格冠军的重判。

## 修后短日志

```text
python -m unittest discover -s tests -p 'test_e155d*.py'
Ran 25 tests in 0.025s
OK
python -O -m unittest discover -s tests -p 'test_e155d*.py'
Ran 25 tests in 0.026s
OK
python -m unittest discover -s tests -p 'test_explicit*.py'
Ran 17 tests in 0.140s
OK
git diff --check
[exit 0, no output]
```

覆盖12项配对测试和13项收集测试：部分重叠/完全不交/空集合/单任务错配，
缺ID/重复ID/null/布尔/对象/数组/浮点/空字符串，整数与字符串身份不混淆，
合法行顺序不影响结果。main测试真实执行配对拒绝，确认没有新verdict写入；
上游收集/评分为替身，不代表LongBench真实打分验收。

合成输入完整生成器：tests/test_e155d_pairing.py，SHA256
`25e4c9e268bbf00a75d06330765aa3137826f2edffb8ae4eaad32153403be4c1`。
实现SHA256：`5cd576fffd61dd53fd93da74570ddef2b1527b0ee0b2e930272bca1891748290`。
设备：Linux x86_64 CPU；无GPU/驱动/框架/kernel配置实测。
以上耗时只是测试时长，不是优化基准，不报告速度/显存/质量提升。

## 未验证边界

独立上下文复核（twolevel_pairing88_review）：12项新增测试通过；字符串/整数
ID × 3种样本数 × 4个seed，共24组新旧完整返回值一致。10种异常 × 新输出
不存在/旧verdict存在，共20组main失败测试全部拒绝，不创建新verdict，旧文件
字节不变；第三名单任务异常也阻止发布。审查结论：有界修复无阻塞。
复核仅使用收集/打分替身验证控制流，不代表实际LongBench scorer通过。

未访问真实收集快照，未跑实际LongBench scorer。相同ID不自动证明answers、
输入内容或cap一致；冻结staging与CI重读源文件的快照绑定仍须后续专项处理。
CI数值沿用0–1逐例分数差，AVG5沿用0–100，未在本补丁中改量纲或算法。
失败不写新verdict，但已有旧verdict文件不会被此补丁删除，必须检查命令退出码。
GPU待验证候选及T7待办保持原状态。
