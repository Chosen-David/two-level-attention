# TL-FAR-SKIP-SUBSPACE-20261025

## Plan (before subspace edits)

User added two subspace analysis/optimization rounds to the existing three-round
PR task. Baseline checkpoint: f3b0347fd5b52fd6145a8db1dde407c9157a0b8b.
Preserve dimensions, their ordering, original quantization/rounding, page/token
decisions, strict far exclusion and original-width final attention. Do not
present d=32 versus d=128 as same-quality or equivalent algorithms.

S1 hypothesis: candidate K subspace extraction and token-local QDQ are repeated
across decode despite immutable historical K. Use a bounded head-major fine-key
ring, prepare live retained keys once and project/quantize only each new token.
This trades auxiliary d-wide K storage and setup work for smaller contiguous
gather and amortized QDQ. Both eager evict and fixed Graph paths; retain/full-D
fallback stays uncached. Measure setup and steady-state, actual auxiliary bytes,
same-width selected IDs and outputs, and cold/short-request regressions.

S2 hypothesis: coarse/fine often use exactly the same dimensions. Reuse scaled
query projection, and consider using the unquantized fine-key projection for
coarse page summaries when the semantics match. Quantized fine K must never be
used for L1 stats. Compare an isolated candidate with S1 and baseline; keep only
correct changes, label timing inside noise and report counterexamples.

Subspace removes only L1/L2 d-dependent work. Selection/top-k, cache management,
kernel/dispatch cost and final K2*D attention do not scale down with d. GPU
bandwidth/load coalescing and launch effects are hypotheses until GPU profiling.
CPU measurements never establish realized CUDA Graph/subspace GPU speedup.

Reused skill/environment and existing results. No new external kernel package.
Primary sources checked 2026-10-25: PyTorch 2.8 tensor_view and index_select
documentation. Basic slicing can be a view; gathers allocate/copy; a view alone
does not promise contiguous/coalesced accesses. Target strategy is data reuse
and layout, not a speculative device-specific instruction.

## Progress

- Baseline frozen after original three rounds. Scope remains PR delivery with
  no main update or merge. Actual CUDA remains unavailable.
- S1 commit b5d724fa540e835f47bea164fea58f8f1dac0140 caches projected fine K
  in original dtype. QDQ is applied once per immutable token, not once per query
  candidate appearance. CPU same-width results: d32,n4096,S8192 eager 4.153→3.079
  ms, fixed 5.745→4.799 ms; with QDQ 4.867→3.248 / 6.616→4.919 ms.
  Identical IDs/outputs. Extra FP32 side cache 2,260,480 bytes versus raw ring
  K/V 18,083,840 bytes. S1 unquantized eager first step 10.77→12.62 ms; fixed
  setup 8.60→10.51 ms. QDQ fixed setup 8.67→15.49 ms. Keep with explicit cost.
- S2 initial projection-sharing candidate: coarse/fine query projection reused;
  only unquantized fine K can feed coarse pages. Same-pointer/shape/stride/dtype/
  device checks are metadata only. Warm default timing was mostly neutral:
  S1→S2 eager 3.040→3.115 ms, fixed 4.939→4.939 ms; do not claim a speedup from
  this candidate alone. Raw s2_vs_s1* files preserve these results.
- S2 follow-up within the same iteration found a stronger unnecessary-work case:
  fixed L2 still ran when its complete candidate bound fit the remaining K2.
  Bypass that scoring/QDQ/softmax path and its fine-key cache; do not allocate
  an eager side cache solely for L1 when the bound already fits. Same-width
  holdout versus S1, S32768,n1024,d32: fixed 4.086→2.821 ms; eager 1.265→1.272
  ms (neutral/slightly worse). Setup+first fixed 14.75→9.85 ms. Keep this branch
  plus correct projection sharing; no universal warm-speed claim for sharing.
- Final remeasurement versus the three-round checkpoint, CPU one thread:
  d32,n4096,S8192 eager 3.840→2.815 ms (1.364x), fixed 5.538→4.559 ms (1.215x).
  QDQ eager 4.373→2.843, fixed 6.213→5.123 ms. BF16 CPU eager 3.859→2.876,
  fixed 5.707→4.599 ms, with 1,130,240-byte side cache. No GPU timing.
  Final QDQ setup+first step still regresses: eager 13.36→15.00 ms and fixed
  15.87→19.67 ms. This is retained as an explicit warm/cold tradeoff, not hidden
  by excluding initialization from all reported metrics.
- Independent-size holdout S32768,n1024: fixed d16 4.161→2.625, d64 3.852→2.205
  ms; both bypass unnecessary L2 and allocate zero fine-key bytes. Eager was
  already bypassing L2 and regressed about 2–3%. Short S64 eager 0.205→0.208 ms;
  fixed 4.783→3.847 ms, still much slower than dynamic eager due to padding and
  fixed-path overhead on CPU. Full-width d128 controls approximately unchanged.
  No performance claims based on profiler-instrumented wall times.
- Final 129 tests + 205 subtests pass; 5 real-CUDA tests skip. Final normal paired
  timing receipts assert 1,920 same-width ID/output matches, max abs error 0.
  Added tests independently compare projected storage to full-history selected
  dimensions, QDQ rounding, arbitrary dimension order, poison/unused slots,
  bounded allocations, gap rebuild/reset, mutation-free full/retain fallbacks,
  one-token QDQ reductions and all-fit elimination of softmax/quantization.
- `summary.json` is regenerated from raw samples and verifies final source hashes.
  `validation.json` binds source/test/receipt hashes. Final profiling allocation
  traffic is approximately unchanged in unquantized d32 (gather output still
  exists): eager 7,083,756→7,081,688 bytes; fixed 13,184,714→13,189,351 bytes.
  The gain is projection/QDQ reuse and gather layout, not elimination of all
  allocations. CPU op observations do not prove GPU memory coalescing or speed.
- Remaining costs: full-D final attention, gather output, top-k/int64 keys,
  masks/padding, query-dependent scores, launches and far-on layers. No change
  to model quality/subspace choice was authorized or made. CUDA graph capture,
  mixed-gate GPU model execution, GPU memory/latency, and pretrained quality are
  still unrun. Self-review plus executable tests only; no independent review.

## Complexity / reproduction / next gates

For C bounded retained slots, fine projection preparation is O(C*Hkv*d), with
O(C*Hkv*d) persistent auxiliary storage. Each append projects/QDQs O(Hkv*d);
each selected pool gather still moves O(Hkv*N*d), while query-dependent L2 is
O(Hq*N*d). Original final attention remains O(Hq*K2*D). This is memoization of
immutable inputs, not cached query answers. The warm improvement has a setup
amortization requirement; do not discard cold or short-request measurements.

Reproduce using exp/bench/far_skip_subspace_compare.py and each JSON config;
baseline roots are the commits above. CUDA `--capture` requires `--modes fixed`
and fails without real hardware. `summarize_far_skip_subspace` checks bindings,
medians, output counts and the completed test log. After future edits, use its
`--source-root` with the measured checkout. Dataset/model-quality evaluation is
not replaced by synthetic numerical correctness.

CODE-S001 (setup/S1): done, parent round3, projected-key ring plus same-width
oracle/measurements. CODE-S002 (S2): done, depends S001, query/K reuse and all-fit
branch with raw negative and positive outcomes. CODE-S003 (integration/delivery):
CPU verified, depends S001/S002, update existing PR; no merge. CODE-S004 (target
CUDA/quality): pending resource, depends S003, run actual capture/replay including
mixed gates and request recapture, profiler/latency/memory and model quality;
on failure keep opt-in and diagnose, do not claim GPU gain or silently fall back.
