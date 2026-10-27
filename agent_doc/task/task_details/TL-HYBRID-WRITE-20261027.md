# TL-HYBRID-WRITE-20261027

User requested substantive analysis of latest main every round, then PR updates
only for concrete improvements. Main baseline b0a378c; existing PR #10 dd331cb.

- Traced Qwen hybrid dispatch, row KV writes, linear incremental summaries,
  compact fine-key/summary maintenance and builder storage ownership.
- Identified six Torch operations plus ring mapping for each row's K/V update.
  Existing #10 kernel did not serve hybrid/EOS. Extended it with device alive
  masking and linear addressing, shared by staged eager and CUDA Graph.
- Preserved main's hybrid cherry-picks, test fixture intent and new evidence.
  Clarified E177's formula-only scope without changing its numeric estimates.
- CPU 68 passed/29 skipped; adjacent 25 passed/128 subtests; interpreter
  30 passed/55 skipped; 30 offline CUDA compilations. No GPU measurement.
- CUDA correctness, capture/replay, sanitizer, paired write latency and full
  request TPOT/quality/allocator acceptance remain open. No merge authorized.

[Coverage, evidence and remote commands](../../results/hybrid-write-20261027/validation.md).
Pending next candidates: conditionally refresh immutable completed-page summaries;
avoid full-width boolean selection reconstruction in hybrid linear attention.
