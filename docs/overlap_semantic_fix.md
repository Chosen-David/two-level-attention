# Semantic baseline before operator overlap

Input: main `1c942ad2ce8d5134ee005673baf126819db8f2f2`.

This repair disables `_iter3_fused_softmax`: its independent 1024-slot
normalizers change scores and Top-K for multi-tile candidate domains. Both
callers retain the existing Torch per-query softmax/nan_to_num/dtype-rounded
GQA mean chain. It can be slower than the invalid fusion; no speedup is claimed.
Do not re-enable the dormant kernel even for short rows without checking its
cross-head shift and dtype rounding against that reference.

The compact kernel now sanitizes inactive IDs before reading MIDM, and the
experimental L2 bypass sanitizes them before scatter. The bypass still changes
the selected-token budget by design and is not enabled by this fix.

The repair follows the independently audited mechanisms in PR #10; it does
not merge that PR's other changes. Operator redesign is developed separately.

Run using the existing project environment (no automatic installation):

```bash
bash scripts/test_tli_semantics.sh --cpu
bash scripts/test_tli_semantics.sh --cuda
# Optional actual CUDA memory-access validation:
compute-sanitizer --tool memcheck --error-exitcode 1 bash scripts/test_tli_semantics.sh --cuda
```

CUDA mode refuses to pass on a CPU-only machine. CUDA tests exercise compact
selection in eager and captured execution. CPU tests establish dispatch and
selection behavior, not CUDA kernel correctness or end-to-end model quality.

Local validation (2026-10-28): Python 3.12.14 / Torch 2.8.0+cpu;
73 passed, 10 CUDA skipped using the script. An additional 48 compact-kernel
CPU Triton 3.4 interpreter cases matched stable argsort exactly (empty/mixed/
full/tail activity, R=17/2049/4224, H=1/8, capacity=65/129). The interpreter
used a test-only NumPy scalar-index compatibility shim. No GPU performance,
native CUDA memory-safety or model-quality acceptance is claimed.
