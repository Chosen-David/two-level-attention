# Prefill-frozen Near/Far skip

Opt-in Qwen3 and GLM-4.7-Flash TLI eager inference reference. This does not change the default TLI
path or enable operator/graph optimizations. Supports unpadded batch size 1,
one-shot dense prefill and single-token decode.

Prefill profiles the last min(256, prompt length) queries, with causal,
query-relative page boundaries. Each layer independently compares mean Near
and Far attention mass with **0.05**, strictly less than the threshold. Decisions
freeze for the entire decode. Both regions may qualify independently; there
is no combined mass budget.

The physical cache chain is `sink -> far -> near -> SWA`, with a linear Far
buffer and bounded Near/SWA rings. Geometry: sink 4, SWA 128, Near 4096,
page 64. Absolute token positions remain independent of packed cache offsets.

| Frozen gate | Decode behavior |
|---|---|
| Keep both | SWA overflow moves through Near to Far |
| Far skip | Release Far; discard tokens aging beyond Near |
| Near skip | Keep invisible Near staging; aging tokens enter visible Far |
| Both skip | Keep only sink/SWA; discard SWA overflow |

Near-only saves attention reads, **not resident KV capacity**. Scoring and
attention never read staging until it ages into Far. Far and both skip
physically reclaim the corresponding cache nodes.

`FrozenRegionCache` owns node lifetimes; `GateConfig`/`profile_regions`/
`decide_regions` calibrate the gates. `frozen_region_runtime.py` is the shared
prefill/profile/freeze/receipt lifecycle used by LongBench and RULER. The Qwen3
and GLM patches recognize the explicit `tli_regions` cache marker before generic
KV update. Passing ordinary DynamicCache does not activate these gates.

For Qwen3 TLI, the default frozen eager path builds one physical K/V arena after
freezing. Decode appends and ages nodes without gathering all visible KV. L1 reads
only live-page tail32 statistics and reuses the original `_select_l1_blocks`;
L2 scores only L1/bypass/protected candidates, then preserves the original
absolute-position free-competition top-k semantics. Both-skip returns before L1,
L2 and top-k. Final attention consumes sorted physical slot IDs and device counts,
so it does not gather the full visible KV or use the sub-16K dense masked fallback.
L1=128, L2=1024, Far=minmax, Near=avg, Near pages=32, Near tokens=off.

The legacy logical-mask path remains available as a test oracle via
`TLI_FROZEN_EAGER_REFERENCE=1` outside the RULER production entry. The physical
selector still materializes a Q-head full-width score buffer for reference-layout
masked softmax, plus a logical-length probability tensor and original-width mid
top-k, to preserve reduction, filler and tie behavior; L2 normalization and
selection are therefore not fully compact. CPU/tiny-Qwen tests establish cache,
selection and attention parity,
but do not establish GPU latency, traffic or peak-memory improvements. Beam search,
chunked prefill, batches, gradients and CUDA Graphs are unsupported.

## Run the validated comparison

```sh
TLI_SCORER_BACKEND=difflib python -m benchmark.LongBench.prefill_frozen_skip \
  --attention tli --thresholds 0.05 --samples 0 \
  --model /path/to/Qwen3-8B --data /path/to/LongBench/data \
  --device cuda:0 --verify-native-full \
  --out agent_doc/results/frozen_region_skip_RUN_ID
```

This runs gates-off (`full`), Far-only and independent Near/Far on all rows of
Qasper, HotpotQA, Musique, GovReport and RepoBench-p. `full` means TLI gates-off,
not dense attention. `--samples 20` selects a smaller screen. For disjoint
workers use matching `--world-size N` and unique `--rank 0..N-1`; use a fresh
output directory when code, input or configuration changes. Context cap=31500.
The runner pins operator mode off, fine gather off, dense prefill and sparse
decode. Its generic `--attention sdpa` mode is a separate reference experiment.

After all workers complete:

```sh
python -m benchmark.LongBench.validate_frozen_skip agent_doc/results/frozen_region_skip_RUN_ID
python -m benchmark.LongBench.analyze_frozen_skip agent_doc/results/frozen_region_skip_RUN_ID
python -m benchmark.LongBench.export_frozen_skip agent_doc/results/frozen_region_skip_RUN_ID
```

## RULER B=1 eager production entry

Frozen regions are disabled by default. Opt in explicitly with the validated TLI
geometry (the ordinary RULER path and its treatment hash are unchanged when the
flag is absent):

```sh
python -u -m benchmark.RULER.pred_ruler \
  --model Qwen3-8B --model_path /path/to/Qwen3-8B \
  --task niah_single_1 --context_length 8192 --method tli --t RUN_ID \
  --data-root /path/to/KVCache-Factory/data/RULER \
  --output-dir exp/results_ruler/Qwen3-8B --max-num 1 \
  --tli_far_method minmax --tli_near_method avg \
  --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 \
  --tli_subspace tail --tia_level2_cmp_ratio 4 \
  --tli_sink_tokens 4 --tli_swa_size 128 \
  --tia_level1_topk 128 --tia_level2_topk 1024 \
  --tli_enable_layer_skip false \
  --tli_far_skip_backend legacy --tli_far_skip_cache retain \
  --tli_prefill_frozen_regions --tli_frozen_region_mode dual \
  --tli_frozen_region_epsilon 0.05 \
  --tli_frozen_region_statistic mean --tli_frozen_region_rows 256
```

The entry rejects non-Qwen3 models, unvalidated geometry, sparse prefill, dense
decode and active CUDA Graph configuration before indexer construction. It fixes
the eager physical execution environment, creates a fresh cache per request and
keeps RULER greedy decoding, first-token EOS handling, newline stopping and
MAX_GEN unchanged. Each JSONL row contains `_frozen_region` with profiles,
decisions, gate counts, prefill/frozen/final physical KV reports and real TLI
receipts. Receipt validation completes before that row is written; a failure does
not commit a new generation. Gate and resolved geometry join the canonical
treatment manifest only when frozen mode is enabled.

This entry is a correctness/production-quality route, not a CUDA Graph or TPOT
claim. Real Qwen3 BF16 GPU parity and paired memory/work/latency measurements are
still required before reporting performance benefits.

The paired GPU harness is `exp/bench/check_qwen_frozen_eager.py`; its one-click
wrapper checks the empty-GPU gate and records a 30-second sideband monitor:

```sh
bash exp/bench/run_frozen_eager_pair.sh \
  /path/to/Qwen3-8B /path/to/KVCache-Factory/data/RULER \
  exp/trace/results/frozen_eager_pair_RUN_ID 0 8192,32768,65536,131072 16 1
```

At 8K it teacher-forces all four frozen states and compares decode-layer outputs,
logits and generated IDs against the legacy logical-mask oracle. Longer lengths
use the independent dual gate. Performance uses reference/physical ABBA plus a
gates-off arm, reports prefill/frozen/final cache receipts, actual selected-KV and
L1/L2 counts, allocated/reserved peaks and decode TPOT. Profiler traces are separate
from timing runs. A report is evidence, not an automatic equivalence or speedup
verdict; inspect generation parity and the GPU hygiene log first.

直接调用 harness 时可加 `--trace-decode`，仅在 correctness 臂保存每一步、每一层、
每个 KV head 的最终逻辑 selected IDs；物理臂还保存实际 attention 消费的物理槽位及
其逻辑映射校验。配对比较会列出差异所在的 step/layer/head。该开关默认关闭，
performance、profiler 和普通 RULER 请求不采集这些大张量，也不能用开启开关后的耗时
替代既有 TPOT。关闭时报告结构保持原样；它补充支持集身份证据，不代表 dense 质量或
bit-exact 验收。

## Accepted evidence and limits

2026-10-31 真实 Qwen3 8K forced-near 逐步对拍（`355a0ff`）未通过 selected-ID
一致性：2016 个 step/layer/KV-head 组合中 1889 个集合不同，差异全部在 Far 区域；
生成词元仍一致，物理槽位映射全部有效。旧最终聚合差 1 已定位到 layer 5、decode
step 6、KV head 2（1022→1021）。详见
`exp/trace/results/frozen_eager_355a0ff_trace_8k_near_verdict.json`。该结果只否定此臂的
支持集身份 parity，不外推其他状态/长度，也不改变既有分口径 TPOT 数字。

Historical full evaluation: source `3e4a6885604ac3e719ee92a4906f87b532535314`,
1300 paired inputs / 3900 generations / 64 completed shards. Dual AVG5
48.239706 -> 48.154300 (-0.085406 percentage points); largest task decline
Qasper -0.683604 points. Musique had no per-request score decline. User accepted
up to 3 points of task-mean/AVG5 degradation after inspecting the results;
this is engineering acceptance, not a pre-registered equivalence test.
Near-only activated in 19 requests, both-skip in 2; coverage is limited.

Independent review checked stored predictions, scores, input identities and
cache/gate receipts, without independently rerunning GPU generation. Evidence
is retained under the project result directory
`agent_doc/results/frozen_near_far_full5_independent005_20261029/FINAL_REPORT_ZH.md`.
These historical quality figures belong to that frozen source, not a fresh
full evaluation of later main changes. Main integration checks cover the
four-state cache lifecycle, TLI selection parity, patched Qwen logits and
benchmark request guard/data coverage. No latency or whole-device memory claim.

## GLM-4.7-Flash

Use the same entry point with `--attention tli --model /path/to/GLM-4.7-Flash`.
The GLM adapter preserves native MLA projections and interleaved RoPE, then
routes the post-RoPE Q/K and full V to the region consumer before any generic
cache write. Prefill profiling uses the model attention scale, including any
configured scaling adjustment, rather than assuming `1/sqrt(head_dim)`.
The selector retains GLM's tail pairs `[208:224] + [240:256]`.

Only unpadded B1, one-shot prefill and single-token eager decode are supported.
GLM rejects active graph state, operator mode, outer batch contexts and external
masks before projection/cache mutation. Its ordinary dynamic far gate must be
disabled: the independent near/far gate owns the request decision. The generic
`--attention sdpa` segmented-cache reference remains Qwen-only. Transformers 5
integer query-length mask requests and the older tensor API are both accepted.

Qwen historical scores above do not establish GLM quality. Four-state routing,
cache lifetime and gates-off parity are integration checks; GLM task accuracy
requires a separately recorded evaluation.
