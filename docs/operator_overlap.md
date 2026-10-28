# Operator blueprint: staged implementation and acceptance

Base: semantic repair `0fd405150557e0688ab967cf569921b0cc96cdbc`.
Branch: `feat/operator-overlap-blueprint`. The feature is opt-in; default `off`
preserves the repaired main route. GPU acceptance is pending. This is a working
candidate, not a measured optimal kernel or a claim of full SM utilization.

## Implemented stages

1. `compute_selection_graph` reuses the exact existing L1/L2 scoring, global
   per-query normalization, dtype rounding, quotas and Top-K. Its terminal
   consumer returns bounded `TokenSelection(indices, counts)` rather than
   constructing a full-width attention mask and scanning it back into IDs.
   Selected IDs remain in ascending mask-scan order, deduplicated. Sink tokens
   may belong to a boundary page in the original selector; remove them only
   AFTER ranking/truncation, without refilling the quota. L2 protected scores
   are not removed from the original normalization domain.
2. `partial_attention.py` loads K/V directly from logical IDs or protected
   ranges. Both SIMT and optional TC kernels support split-KV and independent
   Dqk/Dv. Protected ranges are unioned, selected consumption excludes them,
   and FP32 `(m,l,acc)` states are merged stably, including empty partitions.
   The TC variant shares K/V across a GQA group and uses `tl.dot` for QK/PV;
   its probability cast and padded matrix rows make it a separately gated
   numerical/performance candidate. It is NOT automatically selected.
3. `OperatorPlan` forks protected attention after Q/KV readiness while the
   original stream maintains the index and selects IDs. Selected attention
   waits for the global K2 result; merge joins both streams. The enclosing
   runner captures the fork/join DAG. Warmup creates streams outside capture.
   No tensor result from a warmup geometry cache is written back from capture
   for subsequent eager/other-graph reuse.

## Reachable routes and limits

| Route | Status |
|---|---|
| Qwen3 / Llama shared fixed-cache graph path, B=1 | Integrated; original supported fixed-geometry selector guards apply |
| Qwen3 HybridBatchRunner, mixed far-on/far-off rows | Integrated; rows still dispatch serially in Python/capture |
| FarSkipGraphWorkspace | Integrated direct ring reads; no gathered full-KV buffer on Triton route |
| GQA grouped TC consumer | Implemented, opt-in, offline compiled; native numerical acceptance pending |
| Dqk=192 / Dv=128 consumer | Implemented/tested as a dimension contract; complete DeepSeek MLA model routing is NOT integrated |
| DynamicCache/non-graph generic patch, per-query-head selection, parallel batch indexer | Not new fast routes; use the explicit supported fixed-cache runner |

Far-on still uses existing linear KV storage. Far-off uses the existing
Sink+tail ring. Unified Sink/far-linear/near-ring allocation for far-on,
request-level scheduling across rows, L2 TC dispatch, a faster bounded
compactor and specialized exact Top-K are later stages. L1 geometry still
contains full-capacity masks: this change removes the terminal mask roundtrip,
not every O(sequence-length) metadata operation.

## Configuration

Set before warmup/capture; do not change any of these on a captured runner.
One owner/runner is used serially by the host, with one immutable request gate
layout. Rebuild caches and recapture for a new request or changed gate/layout.

| Variable | Values / default |
|---|---|
| `TLI_OPERATOR_MODE` | `off` (default), `serial`, `overlap` |
| `TLI_OPERATOR_BACKEND` | `simt` (default), `tc` (fp16/bf16), `torch` (reference) |
| `TLI_OPERATOR_SPLITS` | selected split count, 1–64, default 4 |
| `TLI_PROTECTED_SPLITS` | protected split count, 1–64, default 1 |

The Triton routes reject unsupported inputs or absent CUDA; they do not silently
fall back to Torch. The Torch route materializes gathered tensors and exists
for correctness testing only. Stream overlap requires CUDA.

```bash
# Existing project environment, no model downloads or installs:
bash scripts/test_tli_operator.sh --cpu
bash scripts/test_tli_operator.sh --compile    # offline SM80/SM90 compilation
bash scripts/test_tli_operator.sh --interpret  # CPU Triton math/address tests
bash scripts/test_tli_operator.sh --cuda       # refuses CPU-only acceptance

# Whole-model graph comparison; first runs the CUDA correctness gates:
MODEL_PATH=/absolute/path/to/Qwen3-8B bash scripts/bench_tli_operator.sh
# Repeat with TLI_OPERATOR_BACKEND=tc, and splits=2/4/8/16 as separate trials.
```

Use `PYTHON=/path/to/python` if the active interpreter is not the project env.
`CONTEXT_LENGTH`, `MAX_GEN`, `OUT_DIR` configure the paired benchmark. The
existing collector saves raw token sequences and per-step timing. Compare
off/serial/overlap using identical inputs, gates and budgets; repeat with
reversed trial order if differences are small. Real quality-suite acceptance
is separate from tiny random-model smoke tests.

## Numerical and scheduling contract

IDs and quotas must match the repaired selector exactly, including ties as
resolved by the existing Top-K. Attention changes reduction order: FP32 tests
use atol 2e-6–3e-6 / rtol 2e-5; native bf16 kernel smoke tests use atol/rtol
0.025 against a full logical-domain FP64 oracle with model-dtype Q scaling.
These tolerances are test gates, not a substitute for long-context quality.
TC probability rounding is an additional numerical change requiring that gate.

No partial selected attention starts before exact global K2 selection.
Independent protected work can end before Top-K; it cannot hide an arbitrarily
long selector. Streams give no SM reservation. Tune split count, tile geometry,
registers/shared memory and batch occupancy by measured critical-path latency.
Current TC `tl.dot`/compiler staging is not an explicit FA3-style asynchronous
warp-specialized pipeline. No claim of complete TC/CC overlap is made.

## Validation ledger

Local environment: Python 3.12.14, Torch 2.8.0+cpu, Triton 3.4.0; no CUDA device.
Final functional run: 77 semantic/integration tests passed, 10 CUDA skipped;
36 operator tests passed, 17 CUDA skipped (113 passed, 27 skipped total). CPU Triton interpreter: 32 combined
partial/merge cases passed against the independent full-domain oracle.
Offline compile: 32 partial variants + 2 merge variants for SM80/SM90 passed;
TC variants contain matrix-multiply instructions. These results do not validate
native low-precision TC behavior, actual concurrency, GPU latency or real-model
quality. Both the CPU-only environment and the CPU environment with Triton installed
were exercised; no implicit CUDA fallback is counted as native validation.

References informing the design (not copied kernels): CUDA Graphs stream
capture/fork-join documentation and FlashAttention-3 (arXiv:2407.08608).
