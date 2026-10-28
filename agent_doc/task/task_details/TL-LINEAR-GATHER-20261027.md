# TL-LINEAR-GATHER-20261027

Main17e56e8 audit: boolean mask reconstruction/topk remains full-width; defer
candidate-ID interface until ordering, quotas and duplicate semantics are pinned.
Reuse existing opt-in pair-gather kernel for linear rows, preserving topk order
and SDPA. Eager/Graph share per-row bounded scratch. No new kernel or default.
CPU 28 passed/7 skipped; interpreter6 passed/13 skipped. CUDA performance,
BF16/capture, memory and end-to-end gates pending; not merged.
[Evidence and commands](../../results/linear-gather-20261027/validation.md).
