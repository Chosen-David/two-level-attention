# Main audit and hybrid KV writes — 2026-10-27

Baseline: main `b0a378c304cebcba702a1454d0fe8bdf70033926` (previous audit
`55c38d8`). Existing PR #10 head before this update: `dd331cb4183fe8c2bfb6fae9607703254c234022`.
Main cherry-picked the hybrid builder/runner; #10 remains open, with no review
comments at inspection. `opt/ako-pool-score` remains at `a8eb0f0`; its unified
cache-class refactor is concurrent work and is not imported or overwritten.
Read root CLAUDE.md, TASK.md, current narrative, hybrid/parallel-route task and
validation records. No nested AGENTS.md was present in the main tree.

## Main analysis coverage and decisions

| Actual main path | Mechanism / evidence | Decision and remaining work |
|---|---|---|
| `qwen3_attn_forward` → `try_hybrid_batch_decode` | Separate request-local indexers and ring/linear storage. A Python loop still calls index and attention per row; projection/MLP are batched. | Confirmed on current main, not inferred from history. Full rectangular batch pipeline cannot simply accept distinct ring/linear pointers. Full heterogeneous batch attention is deferred. |
| `try_hybrid_batch_decode` KV write | For each `[1,Hkv,S,D]` row, each K and V executes `index_select` of the old token, `where(alive,new,old)`, `index_copy_`. Six Torch operations and four temporary token tensors per row, plus ring mapping. Work repeats in both staged and captured execution. | New candidate; old #10 still contains this sequence. Reuse its `write_pair` kernel, add device alive mask and linear addressing. One launch per row for K+V; active rows never read old KV, dead rows never read/write KV. Index/attention remain per row. |
| Linear `_prepare_index_graph` → `_iter3_maint` | Nonprojected tail-subspace CUDA path already retains compact `k_qat` and page statistics. BF16/FP16 compatible configurations update one page and one fine-key row in an existing fused kernel. | Reject duplicate incremental-stat kernel. Full-capacity initialization, full-L2 clone and projected-basis fallback remain separate lifecycle costs; no change without workload evidence. |
| Compact `FarSkipGraphWorkspace.select` → `_summaries` | `last=(stop-1)` page is gathered/reduced/written every decode execution, even when the newest complete eligible page has not changed. Warmup/replay, no-page intervals and cyclic slots matter. | Pending candidate: conditional on-device page-summary maintenance. Need exact treatment of initialize/capture/EOS and wraps before implementation. Do not reuse old gate quality as selector evidence. |
| Builder and E177 | Builder owns independent physical row allocations. E177 `measure_kv` only multiplies integers; it constructs neither the builder nor CUDA tensors and reads no allocator counters. | Preserve new upstream files and numbers, label E177 as capacity estimation. `70%` means 25/36 hypothetical layers; 4291 slots, not 4288. No measured 67.2% GPU savings or real gate-rate claim. |

## Implementation and merge

Extend the existing optional `TLI_FAR_SKIP_KERNELS=triton` write kernel instead
of adding a second kernel family. Mask loads/stores using the per-row device
alive flag. `TAIL=0` compiles linear addressing without modulo; ring uses its
existing logical-to-physical mapping. Actual K/V/input strides remain supported.
Torch fallback preserves the main sequence; no selector/numerical thresholds
or defaults change. Fixed pointers and no host scalar read make the new write
suitable for capture, pending actual CUDA verification.

Updating a shared kernel avoids introducing pointer tables and a separate
heterogeneous allocator ABI merely to batch writes. It still launches B times
per layer. Pointer-table batching or write+summary fusion may later reduce this,
but needs profiling and ownership/capture tests. No external implementation is
required for this bounded data-movement change; existing repository Triton and
Torch implementations are the competing baselines.

Merge new main ancestry into #10, retain its lazy-workspace preparation and
parallel-route fixes. Main's fixture additions duplicate #10's near geometry;
retain one copy and the current registration error contract. Keep main's new
speed table unchanged. No guide, paper, sglang or agent edits.

## Validation actually executed

Python 3.12 / Torch 2.8 CPU / Transformers 4.56.2 / Triton 3.4. No CUDA device.

- CPU write and associated batch/model/route regression: **68 passed, 29 skipped**.
- Adjacent static/padding/runtime/unified regression: **25 passed, 128 subtests passed**.
- NumPy 2.2.6 + Triton interpreter, new write and existing movement/selector tests:
  **30 passed, 55 skipped**. CPU/interpreter results are logic evidence only.
- Offline SM80/SM90 compilation: **30 specializations** (masked/unmasked,
  ring/linear write plus existing order/gather); not execution or speed evidence.
- E127 indexer and batch GPU gates attempted, both blocked by CPU-only Torch.
  The paired benchmark explicitly rejects a CPU-only host.
- Syntax checks passed. Exact source hashes, main write baseline and logs are
  retained here. Source-bound paired tests use the original main write sequence
  as the oracle; no relaxed tolerance: write buffers must match exactly.

Tests cover strided K/V/new token, prefix, multiple wraps, fp32/fp16 logic,
dead rows with NaN incoming tokens, fixed storage addresses and one fused call
per row. CUDA-only tests cover BF16, mixed B=1/2/8/32, changing alive flags,
ragged capacities and two independent capture lifecycles; **not run here**.

## Remote acceptance commands

Run from repository root with CUDA Torch/Triton and the project dependencies:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_hybrid_kv_write.py \
  tests/test_far_skip_gpu_kernels.py
TLI_FAR_SKIP_KERNELS=triton OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
python -m exp.bench.hybrid_kv_write_compare --batches 1 2 8 32 \
  --capacity 32768 --steps 100 --repeats 6 --out /tmp/hybrid-write-paired.json
```

The benchmark compares the same row write inputs/layout, alternates arm order,
reports event and wall times for eager/Graph separately, records source hashes,
and covers ring/linear/mixed plus live/half-dead/dead batches. It is a write-only
microbenchmark, **not full-model TPOT**. After correctness/sanitizer acceptance,
measure the actual hybrid RULER/runner path with identical prompts, per-request
frozen gates, geometry and outputs, changing only the backend flag. Profile
write launch reduction independently of other already-opted-in far kernels.
Check full-request setup/TPOT, allocated/reserved/peak separately. Selector
quality and end-to-end gain remain unverified; retain opt-in until accepted.

Next audit should revisit conditional compact summary maintenance and linear
selection-mask-to-gather reconstruction, rather than repeating this write finding.
