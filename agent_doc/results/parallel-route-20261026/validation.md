# Parallel route follow-up — 2026-10-26

Inspected main 55c38d8c7ce6afd4d726ce70cd1995408cc4728f and open PR #10
e825274eedb3f89a821193401ebac938be857993. No PR #10 comments/reviews were
present. No E176-named raw script/result was found in the recursive main tree.
Existing CPU/Torch environment and research-implement-optimize workflow reused.

## Findings and decision

- Main caches device lengths by BatchCtx identity, so [7,13] stays [7,13]
  after the host advances to [8,13]. This makes index maintenance/visibility
  stale. PR #10 already keys by step lengths and device; preserve that fix.
- Main returns from the parallel eager branch before checking graph state,
  bypassing live Graph buffers. PR #10 already dispatches Graph first; preserve.
- Both main and old PR #10 pass the attention module to the parallel route
  receipt, whereas QwenRequestCapture expects a layer number. Change both eager
  and Graph sites to record layer_idx (None only for indexer-only test callers).
- Main's per-row metrics loop is superseded by PR #10's existing batched,
  deferred accounting. Keep it rather than introducing per-row host reads.

The three batch modules in main have exactly the same blob SHAs as PR #10;
do not duplicate or rewrite their kernels. Merge the new main ancestry into
the existing PR, retaining the E176 report and explicitly qualifying attribution.
No new PR, defaults, guide or paper edits are necessary.

## Validation

`tests/test_parallel_route_lifecycle.py` extracts actual production functions
with AST for isolated host-route testing. Main source: **4 failed** (stale
length, Graph bypass, B=1/2 receipt identifiers). Candidate: **62 passed,
13 skipped** across this test plus test_tli_batch_parallel,
test_hybrid_batch_cache and test_far_skip_hybrid_cache. These include real Torch
selector/model tests; the four route-contract cases themselves use dispatch
spies, not GPU execution. Logs are retained beside this file.

```bash
TLI_ROUTE_SOURCE=/tmp/tli-main-55c38d8-qwen.py OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_parallel_route_lifecycle.py --tb=short
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_parallel_route_lifecycle.py tests/test_tli_batch_parallel.py \
  tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py --tb=short
```

Environment: Python 3.12, Torch 2.8.0+cpu, Transformers 4.56.2; no GPU.
The baseline source is the exact qwen3_attn_patch.py at main 55c38d8.
No speedup, GPU capture result, memory reduction, or model quality is claimed.

## E176 and next performance experiment

Preserve the remote reported timings; their causal explanation is unverified.
The changed `_tl_decode_batched` parallel branch returns before the legacy
request loop, so that loop cannot by itself explain a successful hit of this
branch. Other upstream/downstream loops may still matter. An ordinary
DynamicCache entry need not reach the fixed-capacity BatchCtx branch at all.
Also `TLI_BATCH_SCORE` defaults to torch; the indexer switch alone does not enable
Triton scoring or GPU pipeline fusion.

Before another timing comparison, record source SHA, all TLI_BATCH_* flags,
fixed-cache entry, actual dispatch counts and per-step lengths. Compare legacy,
parallel Torch, parallel Triton score, and full pipeline with identical geometry
and data, both eager and Graph. Run correctness/capture gates first, then profile
selector and full decode separately. Use the existing run_tli_gpu_pipeline and
docs/unified_far_skip.md commands; do not infer kernel speed from an unconfirmed
entry route. Hybrid per-row index/attention remains a separate candidate for
batching; no new implementation is justified by the unbound E176 numbers alone.
