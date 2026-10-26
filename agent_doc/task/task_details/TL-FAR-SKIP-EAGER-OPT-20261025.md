# TL-FAR-SKIP-EAGER-OPT-20261025

## Plan and contract

User requested implementation of all five eager waste reductions and continued
PR delivery. Parent is PR #7 v1, `3c95ea167d66b462e2c64840adcc7c532a5115bd`.
Preserve its strict far-off region, protected tokens, near page cap, K2 budget,
subspace/quantization choices, per-query-head normalization and deterministic
ties. This does not extend the implementation to CUDA Graph or batch decode.

Reused the research-implement-optimize skill and pinned Torch 2.8.0,
Transformers 4.56.2 tools. Host: CPU-only Torch build, Python 3.12.14;
`nvidia-smi` is absent. Target Python 3.11/CUDA remains unvalidated. No new
external kernel package, background process or permission was required.

## Implemented candidates

| Waste | Adopted change | Cost and contract |
|---|---|---|
| Copy all retained KV on every append | Fixed sink + cyclic suffix; two one-token writes | O(Hkv D) append instead of O(Hkv C D), C = n+w+b-1. Fixed reservation and one-time conversion cost remain. Logical positions stay absolute. |
| Rebuild all near page statistics | Bounded ring of immutable complete-page summaries | Cold O(Hkv n d); thereafter reduce each newly admitted complete page once. Query-dependent scoring still runs every step. |
| Gather full D before projecting to d | One advanced-index operation over token/head/dimension | Reads/materializes requested subspace only. Final attention still needs original-width selected KV. |
| Full score/candidate/output sorts | Exact integer score-position top-k; construct IDs in absolute order | No epsilon changes. Only selected K positions are sorted. FP64 uses stable-sort fallback. Integer key construction has a cost and needs GPU profiling. |
| Repeated index/scratch allocations | Request-local range and cat-output buffers; all-selected/zero-budget shortcuts | Bounded by geometry, not S. No claim of a fused L2 kernel or zero temporary allocations. |

This applies sliding-window/ring-buffer and immutable-page memoization patterns;
ordinary LRU or pointer-based structures are unnecessary because positions expire
monotonically. Candidates stay in array storage suitable for CPU/CUDA.

Rejected plain `torch.topk(scores)` because ties are not stable; rejected epsilon
perturbations because adjacent representable scores must preserve their order.
The IEEE FP32 integer order plus unique position suffix preserves exact ordering
for FP16/BF16/FP32, including signed zero, infinities and NaNs. FP64 keeps the
general reference path. Deferred handwritten fused GPU selection and graph
capture until target-device execution is possible; untested PTX is not a gain.

Primary API references checked on 2026-10-25:

- https://docs.pytorch.org/docs/2.8/generated/torch.topk.html — tie behavior and out buffers.
- https://docs.pytorch.org/docs/2.8/tensor_view.html — views versus advanced indexing copies.
- Installed HF 4.56.2 `DynamicLayer`/`DynamicCache` source — logical length and update dispatch.

## Verification and measurement

`agent_doc/results/far_skip_optimized_20261025/` contains raw output, source-bound
paired samples, an executable summary and validation receipt. The summary is
generated from raw data; CPU numbers are not GPU estimates.

- 72 tests plus 205 subtests passed; 2 real-CUDA tests skipped. Tests cover
  independent selection/attention oracles, page/ring rollover, cached-versus-
  fresh decisions under new queries, BF16 CPU, poisoned unused storage,
  deterministic top-k edge cases, request resets and real Qwen3 generation.
- CPU profiler verifies two single-token copy operations in steady cache update,
  with no old-KV cat/allocation, and one subspace gather without a full-D
  intermediate. Full-step operator/allocation counts are also preserved.
- Paired microbenchmarks alternate v1/new order for three repeats; all steps use
  identical fresh Q/K/V and outputs are compared after timing. Initial eviction
  remains in first-decode timing; steady-state samples exclude eight warmup
  steps. Comparator includes cache update, selection and SDPA, not model layers.
- Primary workload: S=8192/32768, n=4096, nb=32, K2=1024, Hq/Hkv=16/4, D=128,
  FP32 CPU, 64 measured steps. Holdout: S=64/8192, n=1024, nb=8, 32 steps.
- The `*_initial` and `*_pre_validation_guard` receipts preserve intermediate
  source versions. `paired_cpu_holdout_contended.json` overlapped a test process
  and is excluded from final comparisons; final timing runs are serial.

Important tradeoffs: first decode can regress; fixed ring reservation can exceed
the entire cache of a very short statically skipped request. Do not report only
the favorable long-context steady-state numbers. CUDA allocator reserved memory,
GPU latency/launches, pretrained quality and full-model throughput remain unknown.
Known unrelated baseline AST-harness failures remain documented in the v1 task;
the passing subset here does not claim the whole repository is green.

## Integration and lifecycle

Only far-off layers use this path. `retain` intentionally keeps ordinary full
DynamicCache copying as the storage ablation; `evict` uses the ring. Workspaces
are reset at prefill and `indexer.clear()`. They require append-only K, immutable
dimension indices and sequential single-stream use. Returned selection IDs borrow
scratch storage until the next call. Export/rewind/beam/offload guards remain.
New KV shape/dtype/device is validated before either ring tensor is modified.

Delivery is an additional commit on PR #7, with no main push or merge. Self-review
and local execution are complete; independent review and GPU validation are not.
