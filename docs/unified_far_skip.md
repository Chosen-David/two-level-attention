# Unified far-skip integration (2026-10-26)

This integration combines main `4846b04`, PR #7 `fc242e4`, and PR #9
`7e10baf`. Merge the integration PR once; #7 and #9 need not be merged
separately. This is a compatibility/integration change, not a GPU speedup claim.

## Supported routes

| Entry | Storage / gate | Eager and CUDA Graph implementation |
|---|---|---|
| Ordinary DynamicCache | B=1, frozen per-prompt gate | Compact eager; full history when gate is on; no ordinary DynamicCache capture |
| E152v4 / `fill_static_from_dynamic(..., model=model)` | B=1 ring/full layers | Staged eager and whole-model capture; compact workspace remains lazy for eager-only use |
| RULER compact batch / HybridBatchCacheBuilder | B=1/2/8/32 tested on CPU, unequal lengths, independent frozen gate per request/layer | Ring/linear rows, EOS writes frozen, fresh runner and capture per new batch; model projections/MLP batched, index/attention still a Python loop over rows |
| Rectangular fixed full KV + BatchCtx | Uniform selector configuration; B>1 per-row dynamic gates unsupported | Optional full-batch indexer and GPU pipeline for skip and non-skip, eager/Graph |

The hybrid path does **not** silently run the rectangular batch pipeline on ring
slots. `TLI_BATCH_*` flags control the rectangular path; they do not turn hybrid
row iteration into one batched GPU kernel. `TLI_FAR_SKIP_KERNELS=triton` enables
the compact selector's supported gathers/order kernels on CUDA; hybrid masked
row writes still use Torch. The B=1 static/dynamic route retains its fused K/V
write option. Further hybrid batching/write fusion is future work.

## Use

Physical eviction requires all of:

```
--tli_far_skip_backend compact --tli_far_skip_cache evict
--tli_near_len 4096 --tli_near_pages 32 --tli_near_tokens off
```

Also supply the intended K1/K2, sink/SWA, subspace, and per-prompt gate settings.
These example geometry flags do not select a gate or constitute a quality
recommendation. Preserve the experiment's frozen treatment snapshot.
RULER's fixed-cache compact entry now constructs the hybrid builder and runner;
`TLI_CUDA_GRAPH=0/1` selects staged/graph execution. Each unpadded prefill is
converted before the next request and its output released. Direct callers must
release their original prefill cache/output too. Never reuse an old graph after
refill, gate changes, or a different request batch.

The independent rectangular fusion route is explicitly enabled with:

```bash
export TLI_BATCH_INDEXER=parallel TLI_BATCH_SCORE=triton
export TLI_BATCH_FUSIONS=1 TLI_BATCH_GPU_PIPELINE=1
export TLI_FAR_SKIP_KERNELS=triton
```

Its supported full-KV cases fuse cache/index maintenance, Q projection, regional
L1 scoring, candidate construction, and final mask construction. Skip avoids far
scoring/top-k; non-skip uses the regional fusion in both pools. Torch top-k,
softmax and full-width masks still have costs. See `tli_gpu_pipeline.md` and
`tli_skip_noskip_fusion.md` for the original ablations and acceptance scripts.

## Integration fixes and evidence limits

- Seed each hybrid ring's graph workspace once from prefill. Share the setup
  helper with B=1 static decode while preserving its lazy eager-only allocation.
- Retain PR #9's cache refill preflight, linear-layer restoration, request-local
  snapshots, mask registration, explicit positions and EOS handling.
- Remove duplicate configuration/position arguments and superseded B=1-only
  RULER compact branches. Keep PR #7's batch pipeline and GPU kernels.
- Adapt cross-PR tests to explicit `near_tokens=off`, the model-dtype mask, and
  the new hybrid entry. CPU reference tests select real Torch implementations
  when Triton is installed; assertions/tolerances are not weakened.

For page64/near4096/sink4/SWA128 the strict ring needs **4291** slots, including
63 page-alignment guard slots. Main K/V storage is physically bounded; old full
storage must have no surviving references. This does not remove dense-prefill
peak memory, graph pools, selector scratch, or linear far-on rows. Main K/V
bytes, process `memory_allocated`, `memory_reserved`, and peak allocation must
be reported separately. Historical analytic 13% and historical E175 results
are not measurements of this integration.

Strict compact selection excludes legacy boundary-bypass far tokens; historical
gate quality results cannot establish its model quality. No target GPU is
available locally. CUDA capture/replay, sanitizer, end-to-end TPOT, throughput,
allocator memory and real-model RULER quality remain remote acceptance work.
Evidence and exact local commands are in
`agent_doc/results/unified-far-skip-20261026/validation.md`.

## Remote acceptance

Run the integration's CPU suites, then on the project's CUDA/Python 3.11 setup:

```bash
python -m pytest -q tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py \
  tests/test_unified_far_skip.py tests/test_far_skip_gpu_kernels.py \
  tests/test_tli_gpu_pipeline.py tests/test_tli_batch_fusion_kernels.py
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
python -m exp.bench.run_tli_gpu_pipeline --outdir /tmp/tli-unified-gpu \
  --batches 2 4 8 --lengths 32768 65536 131072 \
  --layouts head token --l2-subspaces same full --steps 256 --repeats 6 --sanitizer
```

Run compact CUDA tests with `TLI_FAR_SKIP_KERNELS=torch` and `triton`. Compare
RULER staged/Graph at B=1/2/8/32, mixed per-prompt gates, unequal lengths, EOS,
multiple wraps and a second fresh batch. Compare against the same geometry and
gate configuration; include calibration/fill/capture in total request latency
and report steady-state decode separately. Retain failures rather than counting
CUDA skips or offline compilation as GPU passes.
