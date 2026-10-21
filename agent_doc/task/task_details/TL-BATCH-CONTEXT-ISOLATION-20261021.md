# TL-BATCH-CONTEXT-ISOLATION-20261021

Task-ID: TL-BATCH-CONTEXT-ISOLATION-20261021
Date: 2026-10-21

## Plan

基线 `507e8001158309918d6416831d1ff63df0bd6fa4`。不修改人类 guide、selector、kernel、
chunk2048、预算或质量算法；只修 RULER batch decode 的请求所有权。旧
`sparse_attn/baseline_runtime.py:91-111` 用进程级 `_BATCH_DECODE_CTX`，两个并发请求可在逐层
forward 间互相覆盖 `seq_lens/alive`，异常也会跳过入口末尾的 clear。

方案：以 `ContextVar` 隔离线程与 asyncio Task；保留原 set/clear 非嵌套接口，新增 token/reset
scope 供真实 RULER 入口在正常、return、异常和嵌套退出时恢复前一 owner。固定双线程交错、双
async Task、嵌套异常最小回归；普通及 `-O`、编译、diff、独立复核后才普通发布 main。

## Progress

- 修前在基线运行固定 Barrier 反例：`{'short': [64], 'long': [64]}`，断言期望分别为16/64而失败；
  证明并发请求读到了同一个最后写入的全局对象。
- 候选实现位于 `sparse_attn/baseline_runtime.py:92-163`；RULER 真实调用链在
  `benchmark/RULER/pred_ruler_batch.py:178-225` 使用 `batch_decode_scope`，算法循环与返回值不变。
- `tests/test_batch_decode_context.py` 普通及 `-O` 各3项通过；`compileall`、`git diff --check`
  通过。输入SHA256为 `02ee6ab96a47b9cef1341cef8b084c11518d366340b3d65c6128d960ccd12cd5`。
- 当前 Linux x86_64 / Python 3.12.14 环境没有 `torch`；相邻
  `test_baseline_integration.py`、`test_decode_padding_query.py` 在收集期因依赖缺失未执行，不记为
  产品失败，也不声称 GPU、实模质量或性能验证。ContextVar 不自动传播到新建线程；未来若把
  Python forward 移交线程池，必须 `copy_context()` 或显式传递。独立记录见
  `../../results/tl-batch-context-20261021/independent-review.md`。

本修复是并发正确性恢复，没有速度、显存或质量提升主张，不关闭持续任务。
