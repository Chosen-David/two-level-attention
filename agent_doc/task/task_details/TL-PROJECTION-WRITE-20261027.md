# TL-PROJECTION-WRITE-20261027

Plan: inspect latest main's compact subspace maintenance and reuse kernels where
contracts match; implement only exact direct writes, preserve quantized behavior.

Progress: main17e56e8 and PR10 01d5cd8 checked; projection update not yet fused.
Added optional direct ring-slot projection for both host-position eager append
and tensor-position Graph/hybrid execution. Existing projection memory reused;
linear iter3 maintenance, quantization, selectors and batch routing unchanged.
44 CPU regressions and 12 interpreter-run cases pass; 48 offline CUDA variants
compile. GPU correctness/performance/quality, E127 and batch acceptance pending.
No merge/main push. Update existing PR10 rather than opening a duplicate.

Evidence and reproducible remote paired commands:
[validation](../../results/projection-write-20261027/validation.md).
Next candidates: safe conditional summary refresh; linear mask-to-ID contract.
