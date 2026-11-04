# Top-K 候选 32K/64K/128K 生产验收与 selector 测试补充

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

## 3. 64K B2/B4：Graph 均超过 FlashInfer，同执行逐 token 一致

64K 批量回归使用同一真实 prompt 复制到每个 batch row，仍运行完整 decode step、ABBA、3 warmup + 2 个测量 repeat，全部臂 hygiene clean。每个请求的每层自然 gate 均为 3 个 near-skip 层，far/both-skip 为 0。

| Batch | 源码 | 执行 | NFA 稳态 TPOT (ms) | FlashInfer 稳态 TPOT (ms) | FI/NFA |
|---:|---|---|---:|---:|---:|
| B2 | base | eager | 31.898031 | 20.877272 | 0.654500× |
| B2 | opt | eager | 24.690248 | 20.847352 | 0.844356× |
| B2 | base | Graph | 26.500944 | 14.551680 | 0.549100× |
| B2 | opt | Graph | 13.012152 | 14.494168 | **1.113895×** |
| B4 | base | eager | 33.279312 | 24.697737 | 0.742135× |
| B4 | opt | eager | 25.155648 | 24.812696 | 0.986367× |
| B4 | base | Graph | 27.765312 | 21.497240 | 0.774248× |
| B4 | opt | Graph | 15.428904 | 21.253128 | **1.377488×** |

`token_identity_64k_b24.json` 逐请求对拍全部原始 token：B2/B4 的 base/opt 在 eager 与 Graph 下，A/B 重复及跨源码的所有请求、所有 512 token 均一致。跨执行的既有分叉仍存在：B2 与 B4 的 base/opt 都在 batch 0 的第 34 个 token 首次分叉。B4 eager 已接近 FlashInfer，但仍未达到 1×；Graph 收益不能外推为 eager 收益。

## 4. 32K B2/B4：无回退，Graph 随 batch 出现既有优势

32K B2/B4 的全部臂同样 hygiene clean，每个请求均保持第 11、14 层 far-skip。候选相对基线的 TPOT 和 FI/NFA 基本一致，没有观察到 int32 路径回退，也没有新增加速主张。

| Batch | 源码 | 执行 | NFA 稳态 TPOT (ms) | FlashInfer 稳态 TPOT (ms) | FI/NFA |
|---:|---|---|---:|---:|---:|
| B2 | base | eager | 24.665240 | 20.551824 | 0.833230× |
| B2 | opt | eager | 24.825704 | 20.668296 | 0.832536× |
| B2 | base | Graph | 11.732800 | 11.528880 | 0.982620× |
| B2 | opt | Graph | 11.725536 | 11.526536 | 0.983028× |
| B4 | base | eager | 24.769336 | 20.994520 | 0.847601× |
| B4 | opt | eager | 25.009976 | 20.559472 | 0.822051× |
| B4 | base | Graph | 13.663080 | 15.156296 | 1.109288× |
| B4 | opt | Graph | 13.685728 | 15.122424 | 1.104978× |

`token_identity_32k_b24.json` 显示 B2/B4 的同执行 A/B 与跨源码全部请求、全部 512 token 一致。base/opt 的 eager/Graph 都在 batch 0 的第 25 个 token 首次分叉。B4 Graph 的 1.1× 是基线已有的批量执行结果，不能归因于本次 Top-K 候选。

## 5. main selector 测试契约修复已完整复验

`tests/test_nfa_current_query_selector.py` 的三处修复均只更新陈旧测试契约，不改变生产代码：

1. seed 测试按当前全量 KV 驻留契约检查 seed 可见集只消费一次，不再引用已删除的物理页释放 API。
2. near→far 区域切换不再要求与区域无关的页统计失效，而是用全量重建 reference 对拍全部逻辑页的增量统计。
3. tiled Top-K 测试将 BF16 logits 按 fine-score kernel 的同一规则编码为 int32 packed key 后再调用 dispatcher；mono reference 仍直接消费 BF16 logits，并逐位比较 indices/counts。

在 pc02 GPU0 起跑前实测 `used=1 MiB` 且无 compute apps，使用 main `8f26fe7c` 的源码快照和隔离 pytest 依赖运行：

```text
test_tiled_filtered_topk_bit_exact_with_quota_and_ties: 1 passed in 4.77s
tests/test_nfa_current_query_selector.py: 19 passed in 4.21s
```

## 6. 重基到含 PR40 的 main：源码与生产身份均通过

候选源码提交已单独重基到 main `b629dc29`，并补入三档 capacity 的 Top-K 测试，重基候选为 `264038d7be94cf0a862179764b49ce3fb409a59a`。pc02 GPU0 在 `used=1 MiB`、无 compute apps 时实测：

```text
三档 capacity Top-K focused: 3 passed in 90.13s
完整 selector: 21 passed in 41.86s
```

随后在当前 main 与重基候选上分别完成 64K B1 全生产 ABBA，全部 hygiene clean：

| 源码 | 执行 | NFA 稳态 TPOT (ms) | FlashInfer 稳态 TPOT (ms) | FI/NFA |
|---|---|---:|---:|---:|
| current main | eager | 29.565008 | 20.844625 | 0.705044× |
| rebased opt | eager | 23.963552 | 21.088768 | 0.880035× |
| current main | Graph | 24.756880 | 11.303472 | 0.456579× |
| rebased opt | Graph | 10.674168 | 11.303240 | **1.058934×** |

`token_identity_64k_rebased.json` 同时证明：current main/base 与 rebased opt 的同执行 A/B、跨源码全部 512 token 一致；原始 base/opt 与 current main/rebased opt 也分别逐 token 一致。因此 PR40 与重基没有改变该样本的生成结果。eager/Graph 仍都在第 30 个 token 分叉，不能把同执行通过写成跨执行通过。

重基候选的 128K B1 复核同样 hygiene clean：eager NFA/FlashInfer 为 24.218104/20.802984 ms（0.858985×），Graph 为 12.107944/14.022944 ms（**1.158161×**）。`token_identity_128k_rebased.json` 显示原始 base/opt 与重基候选在同执行下全部 512 token 一致；三者的 eager/Graph 均在第 87 个 token 分叉。

## 7. 128K B2/B4：候选批量验收完成，Graph 最高 2.095558×

128K B2/B4 的 base/opt 四组执行均已完成完整生产 ABBA，全部臂 hygiene clean。每个请求均有 13 个 near-skip 层，far/both-skip 为 0。

| Batch | 源码 | 执行 | NFA 稳态 TPOT (ms) | FlashInfer 稳态 TPOT (ms) | FI/NFA |
|---:|---|---|---:|---:|---:|
| B2 | base | eager | 33.123241 | 23.532168 | 0.710443× |
| B2 | opt | eager | 24.800848 | 23.514384 | 0.948128× |
| B2 | base | Graph | 27.918128 | 20.371176 | 0.729676× |
| B2 | opt | Graph | 14.438248 | 20.368832 | **1.410755×** |
| B4 | base | eager | 34.602328 | 37.333471 | 1.078929× |
| B4 | opt | eager | 25.079688 | 37.331440 | **1.488513×** |
| B4 | base | Graph | 29.180824 | 34.982240 | 1.198809× |
| B4 | opt | Graph | 16.704184 | 35.004584 | **2.095558×** |

`token_identity_128k_b24.json` 逐请求对拍全部原始 token：B2/B4 的 base/opt 在 eager 与 Graph 下，A/B 重复及跨源码的所有请求、所有 512 token 均一致。B2/B4 的 base/opt eager/Graph 也都在 batch 0 的第 10 个 token 首次分叉，因此仍只宣称同执行身份。B4 eager 在该批量下超过 FlashInfer，但 B2 eager 仍未达到 1×；不能用 B4 或 Graph 结果替代其他口径。

至此 32K/64K/128K 的 B1/B2/B4 生产验收网格已全部形成。Top-K 候选的同执行生产身份、三档 selector 测试、批量回归和重基验收均已通过，具备合入条件。

## 8. 尚未关闭的问题

- eager/Graph 跨执行 token 分叉仍存在。
- eager host/runtime 开销、128K niah L1 上涨，以及此前 response 中的异步、padding、短序列 union、full 子空间等问题仍未关闭。
- tiled partial 与 fine-score 直接编码仍是组合候选，未严格拆成两个单变量实验；这不影响上述组合候选的同执行身份与生产验收结论。

原始 arms、逐步 TPOT、生成 token、skip receipt、源码 diff 与 GPU hygiene 均归档在 `benchmark/efficiency/results/nfa_flashinfer_20261104/e2e_topk64/`。
