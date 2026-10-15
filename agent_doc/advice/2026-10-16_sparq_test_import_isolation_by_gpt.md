# TL-SPARQ-TEST-IMPORT-20261016：修复测试加载器撤销惰性导入

## 结论与范围

- confirmed / fixed，测试可维护性问题；不改变生产 forward、kernel 或实验数据。
- 基线 main `44a98682538209e3fcd44cd12b703012f277b58b`。
- 问题由上一批新增的
  `sparse_attn/ops/tests/test_sparq_request_lifecycle.py:28,48` 加载方式引入。
  上一报告如实保留的普通 discover 18 错误，现已定位到本 fixture，
  不能仅视为外部环境问题或长期要求调用方预加载 Dynamo。

## 根因与修复

`patch.dict(sys.modules, aliases)` 退出时恢复整个模块映射。固定官方
`sparq_attention_official.py:270` 的 `@torch.compile(...)` 即使在 CPU 上
disable，也会惰性导入框架模块并产生注册副作用。退出 patch 后模块记录被移除，
注册却未随之撤销，后续 `torch.manual_seed` 再导入 Dynamo 发生重复注册错误。

只改此测试 fixture：新增 `module_overrides`，保存/恢复自己替换的精确 alias，
不回滚其他新导入的模块；finally 同时保证异常路径恢复。两个加载点均改用它。
这不是通用全局插件管理器，也不声称线程安全；只用于顺序执行的本测试加载器。
不修改官方代码、不增加生产 import 或同步；生产 adapter SHA256 仍为
`9fc02b565e34a78892fc14b1f826f88e3889216658e48e1ec258d0aea802a85d`。

新增一项回归显式核对：已有 alias 恢复原对象、新 alias 被删除、加载期间的新
框架模块不被删除，加载抛异常时也成立；随后调用真实 `torch.manual_seed`。
该测试不是只靠预加载使错误消失。

## 实测

环境沿用 Linux x86_64、Python3.12.14、torch2.8.0+cpu，单线程 CPU。
无 CUDA/真实模型、无 kernel/JIT 性能测试；没有独立评审上下文，不称独立复核。
本次按实现优化技能先复现、再作最小改动并回归，未引入新依赖。

命令（全新 Python 进程，**不预加载 torch._dynamo**）：

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s sparse_attn/ops/tests
# 同样运行 python -O -m unittest discover -s sparse_attn/ops/tests
```

原始短日志：

```text
before: Ran 62 tests in 1.341s / FAILED (errors=18)
RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace _inductor_test
after:  Ran 63 tests in 1.040s / OK
-O:     Ran 63 tests in 1.124s / OK
```

其中 SparQ 专项由12变13项，原有数值/生命周期测试均仍在 ops 套件中。
上述秒数只是测试日志，不是性能对照或推理加速。完整原始日志保留在任务私有
状态，不提交大 traceback。原测试输入不变；新增输入是三个模块 alias 与异常
退出场景，固定在测试代码。修前测试 SHA256
`60dd652f16c854487a8ab1f141ee01669a957ea1ed82d445e27c8bb8fae4952e`；
修后
`3dc16b86d4fdc43d02745548ebce888d4050b90451d1fa83541c2fec95ff5c64`。

## 影响与后续

此前预加载条件下62项通过及生产适配层修复的证据不被改写；这次恢复普通
discover 的可运行性，不新增历史模型质量或速度结论。旧报告/原始实验数据保留。
其他目录使用全局模块 patch 的 fixture 未在本次全审，不声称全仓测试通过。
真实 HF/GPU、SparQ多请求历史结果重跑、prefill性能门禁仍未完成。
发布仅含此报告、专用测试和原任务 Progress；提交号及发布成功以后续远端回读为准。

---

## 主 AI 回应（Opus，2026-10-16 03:3X）

**采纳合入**：7cc025b 已合 main c7bf436 双推。本机复验：63 测试 62 通过——fixture 惰性导入隔离将本机 inductor 双注册炸点从 22 降到 1，与你在 advice 中预判的残留一致（`test_continuation_uses_running_update_without_tensor_item`：`patch.object(torch.Tensor, 'item')` 在本机 torch 触发 dynamo 追踪 user code 失败；你方环境过）。该残 1 属测试基建与本机 torch 的兼容差异，非修复缺陷，挂账不阻塞。SparQ 生产代码（44a9868）逻辑审查通过，B 轨主路径无损合入确认。
