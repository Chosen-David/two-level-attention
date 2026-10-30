# NFA paged cache reference

This opt-in eager B1 implementation keeps the existing frozen ring cache as its
correctness and performance reference. It does not change the default cache.
No performance or accuracy improvement is implied by the data structure.

`PagedRegionCache` uses request-local logical pages, each mapped to independently
allocated K/V tensors. K and V may have different channel widths. Aging changes
region boundaries; no old KV is copied from SWA to Near or Near to Far. The first
prefill copies the complete native KV into pages and returns the original KV to
full attention. That conversion overhead must be included in measurements.

| Frozen gate | Retention | Attention visibility |
|---|---|---|
| Keep | all pages | Sink, Far, Near, SWA |
| Near only | all pages | Sink, Far, SWA; aging restores Far visibility |
| Far only | Sink and Near/SWA pages | Sink, Near, SWA |
| Both | Sink and SWA pages | Sink, SWA |

Mixed pages are retained while any token needs storage. The inactive positions
inside them cannot participate in attention. A wholly unnecessary page releases
its tensor references to the Torch allocator and its integer ID to the free-ID
heap. This reduces active KV allocations; cached allocator reservations are not
claimed as released driver memory. Report inactive page slots separately.

The adapter reuses `TLIIndexer.compute_score/compute_mask` on low-dimensional
logical scratch to match the current tail-subspace selector. Final CUDA attention
reads selected full-dimensional K/V directly via page pointer tables. It does not
construct a contiguous full-width historical KV buffer. Scratch, table creation,
selection-index materialization and host synchronization remain real overheads.

The initial implementation is unpadded single-request/single-token eager decode,
with one-shot dense prefill and immutable gates. It has no Graph, batching, beam,
rewind, prefix sharing, quantized KV or offload support. GLM's expanded unequal
K/V widths are representable, but real GLM deployment requires a separate smoke;
this is not a latent-MLA compressed-cache implementation.

The isolated correctness and ABBA model scripts are
`exp/bench/check_paged_region_kernels.py` and `check_paged_region_model.py`.
The model script uses identical known decode input replay for each arm and never
labels it generated-answer throughput. Both arms warm up before measured ABBA
cycles. Compare every forced gate state, retain failures/slowdowns and do not
replace the ring default unless evidence supports that choice. Current measurements
remain pending independent acceptance in `agent_doc/results/nfa_paged_cache_20261030`.

CUDA pages are owned by the stream that first writes them. Updates, freezing and
pointer-table reads on another stream are rejected before state mutation; stream
handoff and CUDA Graph lifetime management are not implemented.

The TLI adapter matches the ring reference's activation-dtype Q scaling. For
fewer than 16,384 visible tokens, it additionally rounds normalized probabilities
to activation dtype, matching ring's dense arithmetic. This requires a second
page-read pass; its cost is included in timing. The standalone kernel's default
remains FP32 online softmax for comparison against SDPA. Neither setting promises
bitwise equivalence across reduction trees.
