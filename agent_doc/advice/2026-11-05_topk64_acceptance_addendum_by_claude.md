# Top-K 候选 32K/128K 生产验收与 selector 测试补充

本文是 `2026-11-05_full_stack_efficiency_response_by_claude.md` 的增量记录，不改写此前结论。候选仍为 `7198002c95e31e547890deaf71387c0fd06bf63d`，基线为 `62d39d8de61fe5a96d728a298148de554a00735e`；本补充不构成候选合入 main。

## 1. 32K B1：同执行逐 token 一致，但没有性能收益

真实 Qwen3-8B、原生 RULER `niah_multikey_1` sample 0、32621 个输入 token、生成 512 token、511 个 decode steps，NFA/FlashInfer 按 ABBA 独立子进程运行。eager 与 Graph 的全部臂均为 `gpu_hygiene=clean`，自然 gate 均为第 11、14 层 far-skip，near/both-skip 为 0。

| 源码 | 执行 | NFA 稳态 TPOT (ms) | FlashInfer 稳态 TPOT (ms) | FI/NFA |
|---|---|---:|---:|---:|
| base | eager | 24.210760 | 20.641784 | 0.852587× |
| opt | eager | 24.285856 | 20.764928 | 0.855021× |
| base | Graph | 9.830000 | 9.821320 | 0.999117× |
| opt | Graph | 9.848480 | 9.835376 | 0.998669× |

`token_identity_32k.json` 对拍原始 `generated_token_ids`：base/opt 在 eager 和 Graph 下，A/B 重复及跨源码的全部 512 token 均一致。该结果只证明同执行身份，不掩盖跨执行问题：base 与 opt 的 eager/Graph 都在第 39 个 token 首次分叉。32K 候选没有 TPOT 回退证据，也没有可宣称的加速。

## 2. 128K B1：Graph 完整生产 TPOT 超过 FlashInfer，eager 未达

同一任务、sample 0、B1、512 token 的 128K 验收已全部完成且 hygiene clean。自然 gate 为 13 个 near-skip 层，far/both-skip 为 0。

| 源码 | 执行 | NFA 稳态 TPOT (ms) | FlashInfer 稳态 TPOT (ms) | FI/NFA |
|---|---|---:|---:|---:|
| base | eager | 30.944296 | 20.691040 | 0.668654× |
| opt | eager | 24.450896 | 20.706784 | 0.846872× |
| base | Graph | 26.193912 | 14.050448 | 0.536401× |
| opt | Graph | 12.129864 | 13.992312 | **1.153542×** |

因此候选在 128K Graph 的完整 `GraphRunner.step` 口径下取得 1.153542×，不是 replay-only 或 post-selection proxy；eager 仍慢于 FlashInfer。`token_identity_128k.json` 显示 base/opt 的同执行 A/B 与跨源码四臂均逐 token 一致。base 与 opt 的 eager/Graph 都在第 87 个 token 首次分叉，该既有跨执行问题仍未修复。

## 3. main selector 测试契约修复已完整复验

`tests/test_nfa_current_query_selector.py` 的三处修复均只更新陈旧测试契约，不改变生产代码：

1. seed 测试按当前全量 KV 驻留契约检查 seed 可见集只消费一次，不再引用已删除的物理页释放 API。
2. near→far 区域切换不再要求与区域无关的页统计失效，而是用全量重建 reference 对拍全部逻辑页的增量统计。
3. tiled Top-K 测试将 BF16 logits 按 fine-score kernel 的同一规则编码为 int32 packed key 后再调用 dispatcher；mono reference 仍直接消费 BF16 logits，并逐位比较 indices/counts。

在 pc02 GPU0 起跑前实测 `used=1 MiB` 且无 compute apps，使用 main `8f26fe7c` 的源码快照和隔离 pytest 依赖运行：

```text
test_tiled_filtered_topk_bit_exact_with_quota_and_ties: 1 passed in 4.77s
tests/test_nfa_current_query_selector.py: 19 passed in 4.21s
```

## 4. 尚未关闭的门禁

- 64K B2/B4 base 验收仍在运行，opt B2/B4 尚未完成；不得外推 B1 结论。
- eager/Graph 跨执行 token 分叉仍存在。
- eager host/runtime 开销、128K niah L1 上涨，以及此前 response 中的异步、padding、短序列 union、full 子空间等问题仍未关闭。
- tiled partial 与 fine-score 直接编码仍是组合候选；完成剩余 batch 回归并重基到含 PR40 的 main 后，才评估合入。

原始 arms、逐步 TPOT、生成 token、skip receipt、源码 diff 与 GPU hygiene 均归档在 `benchmark/efficiency/results/nfa_flashinfer_20261104/e2e_topk64/`。
