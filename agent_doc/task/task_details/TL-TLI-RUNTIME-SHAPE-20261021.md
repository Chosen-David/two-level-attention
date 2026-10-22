# TL-TLI-RUNTIME-SHAPE-20261021

## 范围与基线

- 基线：`0c500b061831ddf00730e96ac3513a84f469141c`（当时的 `origin/main`）。
- 生产入口：`sparse_attn/indexer/tli_indexer.py` 的 `TLIIndexer.prepare_index`、`compute_score` 与 `_moba_mask`。
- 触发条件：直接向单请求 TLI indexer 传入 `B>1`；向公共入口传入不符合子空间契约的 K 宽度、多个 query token；或在 MoBA 路径缺少 `score_moba`。
- 本项只修公共运行时契约，不改 selector、预算、候选集、CUDA/Triton kernel 或计时口径。

## 根因与修复

`prepare_index` 先判断是否走增量路径，旧的批次和 K 宽度 `assert` 却只在后续全量路径。已有 B=1 缓存时，增量实现读取 `k[0]`，因此 B>1 输入会静默丢弃其余请求。`compute_score` 的查询形状约束也在快捷返回之后或依赖父类 `assert`；`_moba_mask` 同样以 `assert` 检查必需分数。所有这些断言在 `python -O` 下会消失。

修复把约束放到各公共入口、位于任何分流之前，并改用带实际值的 `ValueError`：

- `prepare_index` 仅接受 B=1；普通子空间要求 K 末维128，`mla_rope` 要求192。
- `compute_score` 仅接受 B=1、单 query token。
- `_moba_mask` 明确要求 `score_moba`。
- 删除全量路径中已被公共入口覆盖的重复断言；合法输入继续使用原增量、全量或父类快捷路径。

## 修前证据

新增回归先在未改生产源码的基线上运行：

```text
python -m unittest tests.test_tli_runtime_shape_contract
Ran 4 tests ... FAILED (failures=4)

python -O -m unittest tests.test_tli_runtime_shape_contract
Ran 4 tests ... FAILED (failures=3, errors=1)
```

普通模式下四个非法契约均未给出预期稳定异常；优化模式还会接受 B>1、错误 K 宽度和多 query，缺失 MoBA 分数则进入后续代码并产生非契约错误。

## 修后验证

环境：Linux 6.18.55 x86_64 容器，Python 3.12.14。输入身份由测试源码与基线 SHA 共同固定。最终 SHA-256：生产文件 `38ab196ad79db74c07c990cfc3ec5e51ba07b18283069f26eee70b639598198e`，专项测试 `9a21bacc1da8e39a8d17b4bc4aa95d174a547214ee4db90a5fc876fae42ae8ee`。

```text
python -m unittest \
  tests.test_tli_runtime_shape_contract \
  tests.test_tli_debug_contract \
  tests.test_explicit_near_page_budget
Ran 13 tests ... OK

python -O -m unittest \
  tests.test_tli_runtime_shape_contract \
  tests.test_tli_debug_contract \
  tests.test_explicit_near_page_budget
Ran 13 tests ... OK
```

专项测试覆盖非法增量批次、错误 K 宽度、多 query、缺失 MoBA 分数，以及合法128/192维的增量/全量派发和父类快捷返回。相邻回归覆盖调试契约与显式 near 页预算。

## 影响与未验证边界

- 质量：非法输入由静默错误或不稳定错误改为立即失败；没有真实模型质量数据。
- 速度/显存：未测试，不能报告提升；合法路径新增常数级形状检查。
- 环境未安装 `torch`，因此依赖 Torch 的相邻套件无法收集；没有可确认的 CUDA/GPU，生产张量、kernel、端到端请求、峰值显存和性能均未验证。
- 历史上直接复用同一 `TLIIndexer` 并传入 B>1，或依赖非法 K/query 形状的结果不可信，应按正确的逐请求/批量适配入口重跑。合法 B=1、单 query、128维及 `mla_rope` 192维合同不要求历史重跑。
