# TL-PREFILL-LAYER-STAGING-20261104

Source: recurring user instruction to analyze current main and adopt bounded,
verified runtime improvements. Base: `40756b6c925ad0caa5741ccbc305299fb8189567`.
Status: host regression complete; native Torch/CUDA and performance pending.

## Mechanism and change

Qwen prefill calls `write_prefill` once per layer after RoPE and before profiling
and attention. Each call changes only `state.block_tables[layer_idx]`, but used
to rebuild every layer's CPU staging row. `add_request` initially clears all
rows, and each completed layer already has a valid row. Scope that refresh to
the current layer; preserve the no-layer-argument full refresh used by `freeze`
after request-wide reclamation. No new persistent state, allocation, GPU kernel,
synchronization, numerical operation, or KV-copy change.

For L layers and P pages per layer, initial prefill scalar page assignments fall
from P*L*(L+1)/2 to P*L; row fills fall from L*L to L. For L=36, P=2048
(128K tokens / 64), that is 1,363,968 -> 73,728 assignments and 1296 -> 36 fills.
These are source operation counts, NOT GPU launches, throughput or latency.
Freeze still performs its full L-row refresh. Decode and repeated ring wrap
are untouched; this does not remove the separate per-page K/V copy loop.

Existing per-row scalar assignment is the baseline. Vectorized tensor/scatter
reconstruction would add host temporaries and a broader change; a GPU writer
cannot remove this host metadata dependency on its own. Layer-local refresh
removes redundant work without a new library or an invalidation flag.

## Validation

`tests/test_prefill_layer_staging.py` executes actual source methods through a
NumPy tensor facade. Two host test methods pass in ordinary and `python -O`
execution: 18 geometry/layer-count configurations, each with two slot lifetimes,
two interleaved ragged requests, normal/reversed layer order, overwrites,
non-contiguous K/V, full refresh after holes, invalid logical-page filtering,
remove/re-add and independent whole-table oracle checks. CPU pool allocation
and freeing are modeled; this is not native allocator/async validation.

The 36-layer,17-page regression observes 612 page assignments after the change.
Against preserved main source it fails with 11,322 assignments. Syntax compilation
passes. Two native real-cache tests (CPU and CUDA) are skipped here because
Torch is unavailable. The native tests cover per-layer KV values, all four gate
combinations at freeze, ragged requests and reused slots without seeds.

An additional paired source-helper experiment (NumPy rows, L=1/36, P=512/2048)
compares the entire staging tensor after EVERY populated layer. All four pairs
agree. L=36/P=512 records 340,992 -> 18,432 assignments; L=36/P=2048 records
1,363,968 -> 73,728. L=1 is unchanged. This verifies work counts under the host
facade, not real Torch performance; real Torch paired commands follow.

Reproduce at repository root with NumPy and project dependencies:

```bash
git show 40756b6c925ad0caa5741ccbc305299fb8189567:sparse_attn/cache/paged_region_cache.py > /tmp/prefill-staging-baseline.py
PYTHONPATH=. python tests/test_prefill_layer_staging.py
PYTHONPATH=. python -O tests/test_prefill_layer_staging.py
# Expected baseline failure: 11322 != 612.
NFA_CACHE_SOURCE=/tmp/prefill-staging-baseline.py PYTHONPATH=. python tests/test_prefill_layer_staging.py PrefillLayerStaging.test_linear_prefill_page_assignments
# Paired REAL Torch CPU staging timings; excludes KV copies and model execution.
NFA_CACHE_SOURCE=/tmp/prefill-staging-baseline.py PYTHONPATH=. python tests/test_prefill_layer_staging.py --benchmark
PYTHONPATH=. python tests/test_prefill_layer_staging.py --benchmark
# Native CUDA writes / freeze / reuse. Do not count a skip as a pass.
PYTHONPATH=. compute-sanitizer --tool memcheck python tests/test_prefill_layer_staging.py PrefillLayerStaging.test_native_cuda_freeze_reuse
```

No real Torch timing, CUDA correctness, Graph capture/replay, model output,
gate/compact-selector quality or GPU memory measurement was performed here.
Before performance acceptance, pair the same model/prompt/gate inputs at
32K/64K/128K on base/candidate, measure wall-clock prefill including CPU staging
and capture separately, and validate selected IDs/counts/output parity plus
EOS/readmission/new capture. GPU-event-only measurements miss host staging.

## Overlap and integration

All 71 remote heads and 37 open PRs were checked before implementation. PR10
head `2449f56983d7e97c5fe1c4df300c2b50b0c16b81` has the older cache ABI. PR31-37
retain the same all-layer helper; PR26-30 use the earlier shared-table ABI.
The existing branch coverage registry supplies unchanged older diff/task facts.
No same-topic current-ABI fix was found. No branch was transplanted or overwritten.
This narrow change is independent of PR37's seed-capacity fix and PR36's
core-token slicing; it does not supersede those pending fixes. Preserve them
when integrating. Main remains unmodified; submit a draft PR for remote review.
