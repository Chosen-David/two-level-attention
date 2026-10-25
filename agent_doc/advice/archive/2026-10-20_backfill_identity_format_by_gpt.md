# 回填身份检查不能由首行豁免

问题ID：TL-BACKFILL-ID-FORMAT-20261020；基线
`5b91ab69311101752733f98bb1fe9585e9392118`。

`exp/trace/backfill_run_cap_manifest.py:315–338`：首行没有复合ID就直接
返回旧格式回退，跳过后续行。首行无ID、第二行含错指纹/错任务复合ID时，
真实CLI仍可按匹配行数写出`backfilled-from-stdout`和
`sample_set.complete=true`，违反既有混行拒绝契约。

修复在回退前检查全行ID存在性一致、无复合格式混入；保留原首行诊断、
合法复合ID验证、全无ID旧v1和官方ID回退。新增7行守卫，不迁移生产ID、
不改评分/计数政策，不回写历史产物。

复现与回归：[test_backfill_identity_format.py](../../tests/test_backfill_identity_format.py)。
执行`python [-O] -m unittest discover -s tests -p test_backfill_identity_format.py -v`。
修前8个测试方法产生8个失败（含subtest），真实CLI日志：

```text
[096-backfill] provenance=backfilled-from-stdout effective_cap=16384 protocol_cap=31500 sample_set_complete=True
Ran 8 tests ... FAILED (failures=8)
```

修后普通/-O各8方法通过，CLI非零退出且不创建sidecar。独立运行旧脚本
`exp/trace/test_e096_closure_fix_101_106.py`的`test_105_backfill_binding()`，
两模式各`RESULT 10 0 0`；覆盖缺行、乱序、日志绑定及合法旧格式计数回退。
完整旧脚本因首项导入Torch失败，未冒称全套通过。

同一两行CLI输入JSONL字节SHA256：
`c46beb4cc0099abe1276c4db7b0f848ab5b0c75d5180c06b884473f3c0247b2a`；
普通CPU，CPython3.12.14、Linux6.18.55 x86_64/glibc2.39；无Torch/CUDA设备、
架构/驱动/编译器/kernel计时不适用，无速度/质量/显存提升主张。
源码SHA256 `6d4c8c53825c42d62c662bcfabab7aba71b5a6bd11d560e4fdb226a8012ec1c9`；
测试SHA256 `e9331e46466b52a939021da0dfc8ae21c6e9fada1c54926a8f07dc5b488b7c07`。

独立上下文复核同源码/测试SHA：新专项普通/-O各8通过，旧105专项10通过。
另以200行hotpotqa默认计数、无`--expect-count`实际CLI验证：基线返回0并标
完整，修后返回1且不生成sidecar；两行与200行各正反序均拒绝；拒绝
`--overwrite-backfill`混行输入后既有sidecar字节不变。200行输入SHA256：
`74c4e6f5b561679d17c4bd8429279b3f73c1a397529fbe9ada0b4b52ab964c82`。
独审原始简报SHA256：
`66f59f8de438bc97de39f4a26ffc5b0f442e62f06d3f0d85bad150591754f906`。

历史是否存在这种混行尚未知；重测范围是需重新回填的旧JSONL。此修复不认证
问题/context或实际token，不解决TL-LBV1-INPUT-IDENTITY-20261020；真实输入、
tokenizer和新旧协议端到端验证仍待完成。
