# E142 单 token 请求生命周期修复与 E141b 复核

## 目标与版本

- 本轮基线 `6eac1697e1330ee626ed6808a1fdd33169239957`（main）；上轮发布 `949770cdae9d1c77d0162c9b8cffe899f8e80b9c`。
- 增量：E141 探针修正、E141b 数据和 verdict、Opus 回应；生产 MLA patch 与上轮相同，继续其未完成的请求生命周期检查。
- `e138-ablation` 观测为 `e1edf9c316fc4190dd0c29bc5d47477136f8bdab`，本轮不合并、不复测该分支。
- 权限：修改独立库实现/测试/诊断/本报告；不改 `agent_doc/guide/`，不改实验数据。使用独立干净 worktree；只发布本报告列出的四个文件。
- 发布竞争检查：main 前进到 `47f0f6a08f66395860b7a8694dd38e9973e340e8`，原基线发布未成功。已核对并发 MoBA CLI/benchmark 接线、E142a/E140h 新结果；本次四文件无交叠，补丁重放到该 SHA 并重跑回归，保留全部并发文件。

## 发现与修复

| ID | 状态/级别 | 位置（基线） | 触发与实际行为 | 修复 |
|---|---|---|---|---|
| TL-E142-SINGLE-TOKEN-RESET-20261015 | confirmed → fixed/rechecked（CPU host），P2 | `sparse_attn/patches/deepseek_v2_attn_patch.py:126-130` | 新空 KV cache、prompt 长度 1，例如 BOS-only；`q_len > 1` 将首步误判为 decode，不调用 `indexer.clear()`，还返回 `None` 而非请求的 dense attention weights | 改为 `q_len > 0`，首步按更新后的 cache 长度等于本次 q_len 识别；保留空输入拒绝和正常 decode 路由 |
| TL-E141-DIAGNOSTIC-COMPAT-20261015 | confirmed → fixed/rechecked，P2（验证工具） | `exp/trace/verify_e141_qk_reconstruction.py:18-22,49-51` | 修正探针删除本地 `rotate_half`，诊断器却强制提取它，报 `ValueError: missing definitions`，无法复验修复 | 只将历史 helper 标为 optional，核心函数仍必需；为当前探针提供 fixture-local 模块表，RoPE 函数仍来自指定可信 upstream |

第一项违反文件自身的“完整首步 prompt → dense + clear”契约。新请求边界由 KV cache 表示，不由 prompt 是否多 token 表示。`TLIIndexer.clear()` 的请求态包含簇缓存、pair、basis、`_last_q`、`_inc` 等；不能因为某些当前配置会自行重建就跳过清理。

影响边界：本轮证明的是**漏清理、阶段路由及 weights 返回错误**，没有把测试替身的状态残留外推为真模型质量污染。MLA 的 D=192 已不满足当前 D=128 增量缓存复用门，不能据此声称默认 MLA `_inc` 前缀已跨请求污染。已有多 token prompt 的 dense/decode 行为保持。历史长 prompt 实验未见本触发条件证据，无需仅凭本修复宣告全量作废。

同时更新 MLA patch 顶部说明：旧 E141 的 rope-only 等价、天然层跳过叙事已撤回；共享 rope K 不代表不同 query heads 同选。本轮不更改 `mla_rope` 配置、预算、选择算法或动态 gate。

## 最小复现与真实执行证据

新增 `sparse_attn/ops/tests/test_deepseek_request_lifecycle.py`，AST 提取**实际生产 forward** 与两个 layout helper，使用真实 CPU 张量和既有 `eager_decoding_attn`。模型投影为固定 seed 的小线性层，cache/indexer 为显式 test doubles，RoPE 为恒等替身以隔离状态路由；不冒充 transformers 模型集成测试。

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest sparse_attn.ops.tests.test_deepseek_request_lifecycle -v
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s sparse_attn/ops/tests -v
# 同样执行 python -O。
```

依赖路径是本次环境位置，其他环境安装 torch/einops 后从仓根运行即可。初始三方法测试在未修生产源码上：

```text
test_decode_retains_current_request_state ... ok
test_new_prompt_clears_before_selection_and_returns_dense_weights (length=1) ... FAIL
AssertionError: 0 != 1                 # clear 次数
test_reused_attention_resets_for_single_token_new_request ... FAIL
AssertionError: 1 != 2                 # 第二个请求漏 clear
Ran 3 tests ... FAILED (failures=2)
```

修后补充空输入拒绝、长度 1/3 prompt → decode，最终四个 lifecycle 方法加既有二十个 ops 方法：

```text
python:    Ran 24 tests in 1.367s ... OK
python -O: Ran 24 tests in 3.575s ... OK
```

以上时间是**测试总耗时**，不是前后性能对比。dense 输出对独立矩阵参考执行 `torch.testing.assert_close`；验证新请求 clear、老请求不重复 clear、请求复用、attention weights 形状、空输入保持报错。输入 fp32，B=1，H=2，QK=192，V=128，prompt=1/3，decode=1；seed=142。最小单 token fixture 的四个投影权重按 q/kv_a/kv_b/o 顺序及 hidden 输入 bytes 的 SHA256：`edac1355cf79b03037df3062adf9d84b5dd8b0ddc3228e882e5f51af22799bf8`。测试文件 SHA256：`4182187c58de78a43e25021f42b9c0797be7f93bd5bdf8c174b5e9be0039ab3c`。

## E141 修复复核及反证保留

更新诊断器后，以原固定官方源 `85864749cd611b4353ce1decdb286193298f64c7` 中的 RMSNorm/RoPE 定义作参考。该源 SHA256 `7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c`；本轮未重新下载权重或运行真模型。

```bash
python exp/trace/verify_e141_qk_reconstruction.py \
  --upstream-source /tmp/twolevel-e141-modeling-deepseek-pinned.py
# 负对照：--probe-source <949770c 检出的旧探针路径>
# 两种 probe 都另用 python -O 执行。
```

| Probe | norm Q 最大误差 | isolated RoPE Q/K 最大误差 | probability 最大误差 | 退出码 |
|---|---:|---:|---:|---:|
| 旧 probe（949770c） | 15.2833099365 | 2.679043293 / 2.236168861 | 0.031630665 | 1 |
| 修正 probe（6eac169） | 0 | 0 / 0 | 0 | 0 |

normal/-O 相同。固定 fixture hash `ea873ffc4ee5544e014d9fc54ee311b6d95fbe9badc4ba3e597bee9caf383ffb`；修正 probe hash `920a549d0a9b3ec8519db3cbf0a5d574c13e70c444c022843c15e25f9e1ca634`。兼容修改未把旧错误变成通过，也没有用修正 probe 自身生成独立参考。

`TL-E141-QK-RECONSTRUCTION-20261015` 本轮更新为 fixed/rechecked（源码+CPU synthetic parity）。E141b 远端已声明真实 forward hook parity，并撤销旧 rope-only 等价和天然跳层结论；这些 GPU/模型结果是**他方报告**，本轮不冒充复跑，不将所有新数据视为已独立验收。旧 raw 数据保留；旧 verdict 已由对方标记 superseded。

发布期间新增的 E142a verdict 报告两任务质量保留率 96.7%/93.8%，本轮未独立重算，性能仍未测。其 `index_economy`/叙事中“16 head 单索引”不符合当前 patch 的 H=Hkv=16、G=1 调用；代码只共享 K 的 RoPE 值，Q 和 mask 仍按 head 区分。请修订这一成本表述，不能据此宣称索引成本另除以 16；本次未修改对方实验数据或 verdict。

## 环境、成本、未覆盖项

- Python 3.12.14，torch 2.8.0+cpu，einops 0.8.2，x86_64；CUDA unavailable，没有GPU架构/驱动/编译器/kernel配置或测时结果可报告。
- 修复只改变新请求 q_len=1 的 host 分支，不新增 decode `.item()`、同步或张量拷贝；常规 decode kernel 与 selector 不变。single-token 首步不再启动 selector，但未做性能 benchmark，不宣称提速、峰值内存下降或 TTFT 收益。
- 原 HINTS 的受保护 selector 逻辑和 guide 均未改；外层 chunk2048 未改；没有在 main 合入 GPU 未验的 block-list 候选。
- 未覆盖：真实 transformers Cache/模型权重集成、CUDA/bf16、长序列质量、GPU race/memcheck、多流 overlap、Graph 实际派发证明。
- padding mask 处理仍为未完成独立检查，本轮不把它声称修好。短请求真实模型复用/输出 parity 待资源可用时验证。

## 下一检查点

1. CODE-T-MLA-LIFECYCLE-GPU（pending）：真实模型固定输入，单 token 新请求与多 token 请求交替，核验状态清理及生成 parity；失败暂停该部署配置并定位，不修改数据。
2. CODE-T-E141B-EVIDENCE（pending）：复核部署 modeling SHA、真实 hook gate 与 E141b 原始结果的聚合；不再沿用原 E141 无效 oracle。
3. CODE-T-PREFILL-PERF（blocked on GPU）：先串行测 selector/打包/attention 全成本及 compact-block 候选，再决定融合/overlap；保持语义、预算、chunk2048及单流强基线。
