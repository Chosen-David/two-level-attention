# TL-GRAPH-REQUEST-ROWS-20261101

## 问题与范围

基线 main `f5ed4cc3193765bcdc53a9b3046bce721a0b0f97`，cache blob
`4fc4847a751414b8a320f5a4991067218491a7b1`。
CPU staging / block_table_buf 按持久 request ID 编址，但两个
`_store_gather_slots*` helper 按 batch 行读取页表。请求 0 EOS 后，
请求 1 压紧到 batch 行 0，会把请求 1 的 logical IDs 映射到请求 0 的页。
重排及槽位复用同样受影响；逐请求与 both-skip 批量路径均覆盖。

修复显式传递 request ID 映射，gather 输出仍按 batch 行写入。
保留 request-keyed 页表，因此 `project_visible_varlen` 的索引语义不变。
批量 advanced indexing 只生成选中页的 `[B,W]` 结果，不物化重排后的
`[B,max_blocks]` 页表，也不增加逐请求 Python 循环。
相较新增 Triton kernel，沿用 Torch indexing 的维护成本较小；这是正确性
修复，不是实测加速。批量 helper 新增小型 request-ID H2D tensor 及索引操作，
其开销待整步 GPU 配对实验；不宣称单 kernel 或零开销。

## 去重与依赖

PR10 `2449f569` 无此 cache 实现；PR23/24/25 的 cache 仍为旧 main blob
`330b9f24`，PR20 为另一 batch cache 的重复请求检查（`7eb07dde`）。
这些开放 PR 不修复此映射问题。其余未变分支复用既有 diff/任务/实验覆盖；
本改动直接基于上述 main，无未合并依赖，不搬运或覆盖他人分支。

## 已执行验证（主机）

`tests/test_graph_ready_request_rows.py` 从生产源 AST 提取实际 helper，
NumPy doubles 执行地址运算；独立字典 oracle 检查物理页。涵盖页大小
3/4/64、异构非连续页、EOS 压紧、重排、padding、槽位复用，以及 selected
和 interval 两条 metadata caller 路径。

- 修前同一套测试：26 个断言失败；修后：4 个主机测试通过。
- 普通与 `python -O` 均通过；2 个原生 CPU/CUDA 测试因缺少 Torch 跳过。
- 生产源、benchmark caller、测试 `py_compile` 通过。
- 无 GPU、无 Torch/Triton；以上不是 GPU 数值/显存/速度或模型质量实测。

## 待远端验证

在完整仓库及既有 Torch/Triton GPU 环境运行：

```bash
PYTHONPATH=. python tests/test_graph_ready_request_rows.py -v
PYTHONPATH=. python -m pytest tests/test_graph_ready_paged_region_cache.py -q
PYTHONPATH=. compute-sanitizer --tool memcheck python tests/test_graph_ready_request_rows.py RequestRowsNativeTest.test_cuda_gather_replay_after_eos_and_replacement -v
```

原生 CUDA 用例 capture 实际 sparse gather attention，变更 metadata 后 replay，
检查重排、EOS 压紧、替换请求输出是否仍对应独立常量 V 标签。
它不等同完整模型 GraphRunner 验收：当前 Q/K→selector 的时序、逐 KV-head
选择集合、padding query ABI、所有 gate 状态及反复 ring wrap 仍需独立验证。
GPU 配对计时应分别测 helper 与整个 prepare/selector/attention step，并记录
硬件、dtype、B/W/max_blocks、eager/Graph；新 main 的 FA2 benchmark 只计
固定前 K token gather attention，不能证明本修复或整步 NFA 加速。

基线复现：从上述 main 提取 cache 到临时路径，再设置
`BASELINE_SOURCE=/tmp/baseline/paged_region_cache.py` 运行同一测试。
