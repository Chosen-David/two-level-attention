# Optional exact selector compaction

Set `TLI_SPARSE_SORT_INT16=1` and `TLI_COMPACT_SCAN_TILED=1` before constructing
the runner to enable the measured combination. Both default to 0. Packed sort
requires a positive integer cache capacity below 65536; invalid configuration
is rejected before the Qwen decode path writes KV. Rebuild captured runners
when changing configuration.

Signed int16 worker keys reduce stable-sort traffic, then reconstruct the original
int64 keys before the existing normalization. Tiled compaction counts candidates
per tile and performs a stable partition with bounded tile scratch. It preserves
the valid prefix, inactive sentinel values and count. These changes retain the
existing duplicate, per-query-head softmax, protected-token denominator and GQA
rounding semantics. They do not remove normalization or approximate selection.

## Accepted measurement scope

Runtime `e5ce13e`, H20-3e, Qwen3-8B BF16, B=1, 32768 prompt tokens, 64 generated
tokens, whole-forward Graph, serial attention, no near/far skip. Two ABBA cycles
used eight processes, each with six warmup and three measured requests. The only
arm difference was these two flags. Reserved GPU1 passed idle admission and
exclusive-process telemetry. All 72 requests produced identical 64-token outputs.

Median of process-median decode TPOT fell from **17.934350 ms to 16.688392 ms**,
a **6.9473%** reduction. Four paired log ratios give a geometric-mean speedup of
**1.074304x**, Student-t 95% interval **[1.073437, 1.075171]**. All eight processes
passed the preregistered 2% drift and 3% measured-range gates for TPOT. Actual
submission counters verified all four optimized kernels in all 36 layers during
initial capture, with no later host submissions on replay.

Request wall time failed its separate stability gate (one process had 4.1669%
spread), so no request-time improvement is established. This is an own-baseline
ablation, not evidence of beating FlashAttention2, overlap benefit, batch/ring
speedup or general model quality. Fixed 64-token generation is not natural-EOS
completion latency. Both switches remain experimental and default off.

Final native regressions: 68 passed. Independent combined Graph reproduction and
strict memcheck passed with zero reported memory errors. Source, protocol, raw
requests, telemetry, independent correctness and performance reviews are retained
under `agent_doc/results/operator_selector_candidate_main_20261030/`, particularly
`paired/independent-review.md` and its JSON/hash bindings.

The benchmark also includes an explicit Graph FlashAttention2 adapter so future
comparisons can use the same Graph condition. Its presence is not a performance
claim. Reproduce this selector ablation with the existing reserved-GPU runner:

```bash
PYTHONPATH=. python exp/bench/operator_model_paired.py \
  --arms graph_sparse graph_sparse_index_optimized \
  --cycles 2 --warmup 6 --repeats 3 --max-gen 64 \
  --out-dir /absolute/path/to/a/new/result-directory
```
