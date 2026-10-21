# LBv1 同数据集输入身份聚合门禁

问题ID：`TL-LBV1-INPUT-UNIFORMITY-20261021`。修复基线：
`ca1b8c15a0e8c31e91dd50f98c1dfd5676292457`。

## 触发条件与根因

同一打分目录含同一 dataset 的两个或更多 strict method 产物，其中
`source_fingerprint`、`prompt_template_sha256` 或
`actual_input_ids_ordered_sha256` 至少一项不同，或 v2 与旧 strict sidecar 混挂。
基线的 `benchmark/LongBench/eval.py` 会逐文件验证并记录这些字段，却没有跨
method 比较；因此不同问题正文、模板或实际送模 token 的分数仍能进入同一
`result.json`，形成不可证明的公平对照。

修前对基线源码执行入口 AST 检查，原始短日志为：

```text
baseline_gate_calls= 0
AssertionError: same-dataset input identity uniformity gate is missing before result publication
```

## 修复

- `benchmark/LongBench/eval.py:133-156` 从已验证 sidecar 返回实际 dataset；
  `:445-448` 在任何 `result.json` 写入前调用输入身份单一性门禁。
- `benchmark/LongBench/run_cap_manifest.py:187-220` 按 dataset 分组。同一 dataset
  的多个 v2 strict 产物必须对源行、模板、实际 token 三轴完全相同；缺闭包、
  任一轴漂移及 v2/legacy strict 混挂均以
  `LBV1-INPUT-UNIFORMITY` fail-closed。不同 dataset 独立比较；全 legacy strict
  仍按历史口径可读，但不获得 v2 证明。
- `tests/test_lbv1_input_uniformity.py` 固定门禁必须位于发布前，并覆盖三轴漂移、
  缺字段、v2/legacy 混挂、跨 dataset 允许和相同身份通过。

## 实际验证

环境：CPython 3.12.14，Linux 6.18.55 x86_64；无 `torch`、`jieba` 与
`nvidia-smi`，无GPU/驱动/框架/kernel配置。专项输入文件
`tests/test_lbv1_input_uniformity.py` SHA-256：
`b99c94e111e87e96567c3531dd4ebb984d394de46d42dbfd3ac1411b98e6aa59`。

```text
python -m unittest tests.test_lbv1_input_uniformity -v
Ran 7 tests in 0.005s — OK
python -O -m unittest tests.test_lbv1_input_uniformity -v
Ran 7 tests in 0.004s — OK

python -m unittest tests.test_lbv1_input_identity_v2 tests.test_backfill_identity_format -v
Ran 15 tests in 0.042s — OK
python -O -m unittest tests.test_lbv1_input_identity_v2 tests.test_backfill_identity_format -v
Ran 15 tests in 0.131s — OK
```

`python -m compileall -q benchmark/LongBench ...` 与 `git diff --check` 通过。

## 质量、速度、内存及重测范围

这是结果可比性正确性修复：错误混挂由“可发布”变为“发布前拒绝”，合法同输入
分数计算不变。当前没有真实 LongBench 数据、tokenizer/model、GPU或可导入的
完整评分依赖，未重评历史目录，也未测质量、TTFT、吞吐、显存、kernel数或
host同步；不主张性能变化。历史 legacy sidecar 不含完整输入闭包，本修复不回填
伪证据；其是否存在污染及对冠军排序的影响仍需在有生产资源时按真实源/模板/
token身份重验。
