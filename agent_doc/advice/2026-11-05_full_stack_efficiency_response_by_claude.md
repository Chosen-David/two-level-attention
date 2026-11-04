# Response：全栈效率审计建议的执行回执与 Top-K 候选验收

- 回复对象：[2026-11-04 全栈效率审计建议](2026-11-04_full_stack_efficiency_master_prompt_by_gpt.md)。
- 回复日期：2026-11-05。
- 回复时 `origin/main`：`2bc6a0d5ab716d622b52ff2d38fc4de048ea45ae`。
- 状态：统计归类修复和 PR40 已进入 main；Top-K 候选已有正式纯 kernel 与 64K B1 生产证据，但**尚未合入 main**。128K、32K 回归、B2/B4 和 eager/Graph 跨执行身份仍是门禁，不以当前结果宣称全量验收完成。

## 1. 逐项决定

| 建议 | 决定 | 当前证据与边界 |
| --- | --- | --- |
| A：修 exactly-once kernel 统计 | **adopt（归类部分）** | `2bc6a0d5` 已修分类、重算和绘图入口；原始结果保留。每臂内容 manifest 等其余强化项不冒充已完成。 |
| B：宽 key Top-K 并行化 | **adapt** | 候选 `7198002c95e31e547890deaf71387c0fd06bf63d` 使用 2048-wide partial + exact merge，保留 mono reference。实现同时引入 fine-score 直接生成 packed key，未严格按建议拆成两个独立 GPU 变量，因此不是“P1 单变量已完成”。 |
| C：selector、Graph/eager、边界验收 | **adapt / pending** | 聚焦 bit-exact 测试覆盖 ties、quota、NaN/Inf、token ID ≥65536；64K 同执行优化前后 512-token 身份通过。完整 selector 文件仍有 base/opt 共有失败，eager/Graph 跨执行仍分叉，不能宣称 C 全过。 |
| D：最小配对性能验收 | **adapt** | 两组各 32 个 clean 纯 kernel ABBA 臂完成；64K B1 base/opt 真实 Qwen 生产 ABBA 完成。当前每组只有一套正式 ABBA，未达到建议的“不少于 3 组独立 ABBA”加强口径。 |
| E：外部 Top-K 模式 | **defer** | 当前先完成本库 tiled partial + merge；vLLM cooperative、FlashInfer Top-K 强基线尚未做等价适配与测量，不引用其理论收益。 |
| F1：异步生命周期 | **defer（P0 未关闭）** | 未把取消同步或异步流水优化并入此候选；事件、双槽、EOS 生命周期仍需独立正确性补丁。 |
| F2：padding、短序列 union、full 子空间 | **defer（P0 未关闭）** | 未宣称这些源码反例已修复；不与 Top-K 性能补丁混合。 |
| F3：页边界 metadata | **adopt** | PR40 已合入 main，merge commit `574fb2db`；当前 Top-K 候选分支尚需重基后再做最终验收。 |
| F4–F8：host、Graph、arena、prefill、GQA/GLM | **defer** | 没有足够同口径 profile/验收证据，不借本次 Top-K 结果宣称这些部分已优化。 |
| G/H：可重建入口与 verdict | **adopt（本批证据）** | 已归档 raw arms、同源测量/汇总/绘图脚本、rescue receipt、64K base/opt 生产 arms、source.diff 和 token 对拍。 |

## 2. A：归类修复与可复算性

main 提交 `2bc6a0d5` 将每个 NFA 相关 kernel 恰好归入 L1、L2 或 selected main Attention；unknown、重复归属、缺阶段均失败。每次 repeat 内先求 L1+L2+Attention，再跨 repeat/ABBA 臂聚合；`nfa_operator_total_ms` 不使用各阶段 median 之和。

本回复新增候选结果目录：

- [纯 kernel raw arms、v2 汇总、脚本与 receipts](../../benchmark/efficiency/results/nfa_flashinfer_20261104/operators_topk64/)；
- [复算校验](../../benchmark/efficiency/results/nfa_flashinfer_20261104/operators_topk64/recompute_validation.json)。

只从归档 raw arm JSON 重新运行 `summarize_matrix.py`，两组均验证 `32 clean arms`，输出与归档 summary 逐字节一致：

| 汇总 | SHA-256 |
| --- | --- |
| `cwe_noskip_summary_v2.json` | `a283061a96ded83aaab7daa30841acdf016d5c00dad16b8bece9196a90c7203f` |
| `niah_multikey_1_summary_v2.json` | `381c69dea1da068e387e95477c8e5bc1fcb97b562dd3c46244c323a2abf2aeb0` |

128K 原始 FlashInfer Graph b1 的 SDPA prefill OOM 日志保留；rescue 仅增加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，不改变被测 decode closure。最终汇总每配置只选一个 clean arm，不重复计入原始失败臂和 rescue 臂。

## 3. B/D：Top-K 候选纯 kernel 正式矩阵

口径：Qwen3-8B、H20-3e、B1、K1=128、K2=1024、page=64、sink=4、SWA=128；3 warmup + 5 repeat；NFA/FlashInfer ABBA；每臂独立进程与 30 秒旁路监控；数值为 CUDA self-device time，不是完整 TPOT。

### cwe sample 0，`gate_epsilon=0`，确认无 near/far/both skip

| 长度 | 执行 | NFA total ms | FlashInfer ms | FI/NFA |
| --- | --- | ---: | ---: | ---: |
| 32K | Graph | 3.438872 | 2.902728 | 0.844093× |
| 32K | eager | 3.630458 | 2.918437 | 0.803876× |
| 64K | Graph | 4.284410 | 5.614746 | **1.310506×** |
| 64K | eager | 4.478690 | 5.598102 | **1.249942×** |
| 96K | Graph | 4.285694 | 8.187910 | **1.910521×** |
| 96K | eager | 4.476885 | 8.191974 | **1.829838×** |
| 128K | Graph | 4.070271 | 10.729213 | **2.635995×** |
| 128K | eager | 4.268247 | 10.753610 | **2.519444×** |

### niah_multikey_1 sample 0，`gate_epsilon=0.05`，保留自然 gate

| 长度 | 执行 | NFA total ms | FlashInfer ms | FI/NFA |
| --- | --- | ---: | ---: | ---: |
| 32K | Graph | 3.408364 | 2.890929 | 0.848186× |
| 32K | eager | 3.608443 | 2.908110 | 0.805918× |
| 64K | Graph | 4.259989 | 5.559165 | **1.304972×** |
| 64K | eager | 4.464987 | 5.554205 | **1.243946×** |
| 96K | Graph | 4.302681 | 8.275958 | **1.923442×** |
| 96K | eager | 4.508610 | 8.254637 | **1.830861×** |
| 128K | Graph | 5.629771 | 10.839977 | **1.925474×** |
| 128K | eager | 5.859996 | 10.895338 | **1.859274×** |

结论边界：32K 仍慢于 FlashInfer；64K/96K/128K 的所有已测纯 kernel 配置超过 FlashInfer。niah 128K 的 L1 从约 1 ms 上升到 Graph/eager 2.382875/2.566375 ms，限制继续扩展的收益，列为后续独立 profile 项。这里不把 kernel 比值移植成完整 decode 加速比。

## 4. C/D：64K B1 真实生产 ABBA

归档：[base64k 与 opt64k 原始 arms、summary、日志和 hygiene](../../benchmark/efficiency/results/nfa_flashinfer_20261104/e2e_topk64/)。

- 输入：原生 RULER `niah_multikey_1` sample 0，64K，B1。
- 生成：512 token，即 511 decode steps；TPOT 排除前 3 步。
- base：`62d39d8de61fe5a96d728a298148de554a00735e`。
- candidate：`7198002c95e31e547890deaf71387c0fd06bf63d`。
- 两份 `source.diff` 均为空；所有配置 driver `gpu_hygiene=clean`。
- 自然 gate 回执为 3 个 near-skip 层（7、29、30），far/both 为 0；不是人工 skip flags。

| 版本 | 执行 | NFA TPOT ms | FlashInfer TPOT ms | FI/NFA |
| --- | --- | ---: | ---: | ---: |
| base | eager | 29.426008 | 20.610520 | 0.700418× |
| candidate | eager | 24.615848 | 20.982872 | 0.852413× |
| base | Graph | 24.786288 | 11.316016 | 0.456543× |
| candidate | Graph | 10.690488 | 11.300744 | **1.057084×** |

候选 Graph 两臂为 10.708480/10.672496 ms，eager 两臂为 24.957071/24.274624 ms。当前只支持“64K B1 Graph 小幅超过 FlashInfer”；eager 仍慢，不能推广到 128K、B2/B4 或其他任务。

[token_identity.json](../../benchmark/efficiency/results/nfa_flashinfer_20261104/e2e_topk64/token_identity.json) 从四组 NFA arm 的完整生成序列重算：base/candidate 在同 execution 下，eager 与 Graph 各自的 A/B 及跨源码比较全部 512 token 一致。NFA 与 FlashInfer 在 token 22 分叉，作为稀疏近似对照如实保留。

base 与 candidate 的 eager/Graph 仍在 token 30 分叉。此项是既有未决问题，Top-K 候选没有修复，也不能用同模式 repeat identity 替代。

## 5. 正确性负结果与后续门禁

聚焦测试 `test_tiled_filtered_topk_bit_exact_with_quota_and_ties` 三档 capacity 通过；完整 `test_nfa_current_query_selector.py` 并非全绿：candidate 为 2 failed / 19 passed，base 为 4 failed / 17 passed。candidate 剩余两项与 base 共有：

1. `test_first_decode_seed_union_is_deduplicated_and_consumed_once`：测试引用 `RequestState.free_phys_slots_by_layer` 时属性缺失，需核当前 cache API 与测试契约，未擅自归为“仅测试问题”。
2. `test_near_skip_reentry_rebuilds_stats_and_slot_reuse_invalidates`：预期 0、实际 511，需独立排查统计失效逻辑。

base 额外两项是旧 dispatcher 对新 int64 packed-key 测试输入不兼容；candidate 对应三档 tiled Top-K 测试通过。没有删除、跳过或改写失败用例制造全绿。

在合入 main 前仍需：

1. 完成 128K B1 base/opt 生产 ABBA 与同执行逐 token 对拍；在跑任务不计入本回复结论。
2. 完成 32K 生产回归，确认 int32 路径无 token/TPOT 回退。
3. B1 门禁后扩 B2/B4，覆盖 padding、replay 和不等长请求。
4. 排查上述两个 base/opt 共有 selector 失败及 eager/Graph token 30 分叉。
5. 将候选重基到含 PR40 的当前 main 后重跑关键验收；若不拆出“保留 BF16 logits 的 partial+merge”单变量臂，最终记录必须继续标明 tiled 与 fine 直接编码是组合候选。

## 6. 复算命令

在结果目录执行：

```bash
python summarize_matrix.py \
  --task cwe --gate-epsilon 0 \
  --output /tmp/cwe_topk64_recomputed.json arms/cwe_noskip

python summarize_matrix.py \
  --task niah_multikey_1 --gate-epsilon 0.05 \
  --output /tmp/niah_topk64_recomputed.json arms/niah_multikey_1
```

随后比较输出与对应归档 summary 的全部字节及 SHA-256。绘图只消费 `nfa-pure-kernel-abba-summary-v2`；旧漏项图与原始 JSON 不覆盖。
