# Fixed workspace capacity — 2026-10-27

Main17e56e8, PR10 start0450426; reviews/comments and other branch heads unchanged.
Read root CLAUDE83dbec28, previous task/evidence and linear-ID negative probe.
New coverage: main _prepare_index_graph warmup/stale buffers, clear(), hybrid
snapshot ownership, ring construction, FarSkipWorkspace.cat allocation lifetime.
_ginc stale references preserve already captured addresses intentionally; do not
blindly clear them. Hybrid snapshots explicitly drop _ginc and ring construction
copies into independent bounded buffers. No new full-prefill-reference leak proven.

Main fixed-shape Graph workspace concatenations use geometric capacity rounding
intended for growing eager arrays. PR10 additionally uses the same rounding for
persistent linear K/V gather scratch. H8/N1156/D128/bf16 has4734976 payload bytes,
but its two buffers actually reserve8388608 bytes of Torch backing storage.
Earlier linear-gather record called payload a bound on scratch; that was wrong.

Change: opt-in fixed_shapes allocation mode uses requested sizes for cat/scratch,
never shrinks, and reuses storage when sufficient. Enabled for FarSkipGraphWorkspace
and hybrid linear row workspace only; variable eager geometric growth remains.
Range caching unchanged. Fixed geometry is initialized/warmed before capture.
No CUDA kernel arithmetic, selector semantics, quantization or gate change. No
new external kernel: existing Torch allocation and current movement kernels suffice.

Validation:64 CPU passed,9 CUDA skipped, including exact paired selector outputs
through85 steps/multiple wraps, pointer stability, mixed ring/linear/native dense
and repeated-request regressions. CPU pair allocation:8388608 ->4734976 bytes,
3653632 bytes less backing storage per linear row (3.484375MiB).100 repeated
scratch requests preserve pointers. paired.json records source hash/CPU-only
scope. This is NOT GPU allocated/reserved/peak or KV eviction savings, nor latency.
Existing linear CUDA/interpreter tests and benchmark now request the fixed mode;
not run here. No kernel changed, no new offline compilation claim. Syntax passed.

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_fixed_workspace_capacity.py tests/test_far_skip_graph.py tests/test_hybrid_batch_cache.py
TLI_FAR_SKIP_KERNELS=triton python -m pytest -q tests/test_hybrid_linear_gather.py
python -m exp.bench.workspace_capacity_compare
python -m exp.bench.hybrid_linear_gather_compare --lengths 32768 131072 --steps 100 --repeats 6 --out /tmp/linear-paired.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

The allocation comparison reports CUDA allocator deltas only if CUDA is available;
CPU always reports logical backing storage. Remote full model allocator accounting,
Graph capture/recapture and paired latency remain acceptance gates. Keep prior PR
opt-in GPU status. Linear ID dedup/zero-fill and conditional page summaries remain
open; do not relax bitwise tests based on previous small CPU attention discrepancy.
