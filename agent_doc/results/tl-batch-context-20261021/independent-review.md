# TL-BATCH-CONTEXT 独立复核

- 基线：`507e8001158309918d6416831d1ff63df0bd6fa4`。
- 结论：无阻塞项。`ContextVar` 能隔离并发线程与 asyncio Task；`batch_decode_scope` 的 token/reset 可在嵌套及异常退出后恢复上一 owner。RULER 成功路径的生成循环、状态更新与返回值未改变，异常路径新增可靠清理。
- 修前证据：双线程均完成 `set` 后再读取、读取后才允许 `clear`；旧进程全局变量实现稳定出现一方读到另一请求的 `seq_lens`（`[64] != [16]`）。
- 修后验证：
  - `python -m unittest -v tests.test_batch_decode_context`：3/3 通过；
  - `python -O -m unittest -v tests.test_batch_decode_context`：3/3 通过；
  - 正式 asyncio 双任务交错用例：分别保持 `[24]`、`[96]`，退出后主上下文为 `None`；
  - `python -m compileall -q ...` 与 `git diff --check` 通过。
- 未验证边界：当前环境无 PyTorch/CUDA，未执行真实 Qwen3、GPU kernel、批量 RULER 或性能测试；若未来把 Python forward 显式移交新线程，需用 `copy_context()` 或显式传递上下文。
