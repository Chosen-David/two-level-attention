# TL-GRAPH-NEAR-RESIDENCY-20261102

基线 main f5ed4cc3193765bcdc53a9b3046bce721a0b0f97，cache blob
4fc4847a751414b8a320f5a4991067218491a7b1。

## 机制与修复

`freeze` / `_sync_block_table` 使用 `_resident_page_ranges_for_length`，后者
原先复用计算可见区间。near-only 因而释放暂不参与 attention 的 near 页；
当这些旧 token 进入 far，sync 只重新分配页，decode 仅写当前新 token，
无法恢复旧 KV。主机复现：prompt96/page4/near32/window7，长度99时 token56
成为 far 可见，但原值56已丢失。混合 near-only/both-skip 同样受影响。

修复：只要任一层保留 far，公共页表保留全部历史。各层计算区间仍按原 gate
屏蔽 near；所有层均跳过 far 时，继续按原有可见页并集回收。共享物理页号的
架构本来就需要满足所有层的驻留需求；不在此 PR 改成逐层独立页表。
现有 `_max_resident_pages_for_layer` 对保留 far 的层已按完整历史预留容量，
所以无需增加容量公式或 decode 全局分配。此修改不会减少实际 arena 分配，
不是 GPU 显存节省或加速声明；near-only 保留更多有效 KV 是必要正确性成本。

与已有实现比较：较早 B1 paged cache 的 near-only 保留语义应继续遵守；
PR10 hybrid ring/linear 布局不是此 GraphReady 公共页表的替代补丁。PR18
处理旧 paged reclaim no-op，PR26处理请求行映射，均不修复此问题。
直接基于 main，无未合并依赖，不覆盖其他分支。

## 已执行与待验

源码 AST 执行实际 freeze、resident-range、sync 方法，主机存储 double 在释放
时清除旧内容，独立 token 标签 oracle 检查所有可见 KV；不是 CUDA 执行。
四态及两种混合层 × page4/8/16，共18种配置，每种覆盖192步 decode，
同时检查物理页不重复、decode 不做全局分配。修前6组失败，修后18组通过；
普通及 `python -O` 均通过。原生 Torch CPU/CUDA 两测试已添加、因环境缺失
跳过；py_compile通过。真实模型、CUDA Graph、显存、吞吐和选择质量仍待验。

```bash
PYTHONPATH=. python tests/test_graph_ready_near_residency.py -v
PYTHONPATH=. python -m pytest tests/test_graph_ready_paged_region_cache.py -q
PYTHONPATH=. compute-sanitizer --tool memcheck python tests/test_graph_ready_near_residency.py NearResidencyNativeTest.test_cuda_original_kv_survives_near_to_far -v
```

修前复现：将基线 cache 文件保存到 /tmp/baseline/paged_region_cache.py，设置
`BASELINE_SOURCE=/tmp/baseline/paged_region_cache.py` 运行同一测试。
远端性能配对应包含 freeze、metadata 和完整 decode，固定输入/gate/选集，
分别记录 near-only 与 all-far-skipped；不要把少保留了应存在的数据视为基线优势。
当前 query/selector 时序、seed 页策略、逐 KV-head 选集等问题独立存在；本 PR
不声称完成整个 GraphReady 模型正确性验收。
