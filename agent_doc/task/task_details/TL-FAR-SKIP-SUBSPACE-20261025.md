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
