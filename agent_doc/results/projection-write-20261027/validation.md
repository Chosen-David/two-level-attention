# Projection maintenance — 2026-10-27 13:40 CST

## Main coverage and candidate

Analyzed latest main `17e56e8fe30cf8037caf3ce7b38040e754f0521e` with root
CLAUDE.md unchanged (83dbec28); current recursive tree has no AGENTS.md.
Read prior hybrid-write, summary-audit and linear-gather task/evidence records.
PR10 remains open at `01d5cd8563b1277258713558785dae2a7e4f7486` at start,
no reviews/comments. Other PR heads and opt/ako-pool-score a8eb0f0 unchanged.
This isolated candidate extends PR10, preserving all main/PR changes. No sglang,
agent, guide or paper edits. No actual concurrent worker is visible locally;
remote branches are only concurrency context, not proof of idle remote workers.

Actual main paths inspected:
- HybridBatchCacheBuilder -> per-request frozen ring/linear rows ->
  try_hybrid_batch_decode still loops rows for index and attention.
- ring -> FarSkipGraphWorkspace.select -> FarSkipFineCache.update_tensor:
  device ring mapping, advanced gather [H,1,d], index_copy to [H,capacity,d].
- staged eager -> fine_cache_for -> append: index_select [H,d] then copy to slot.
- TLIIndexer l2_quant defaults false; far_skip_runtime passes it into cfg.
  Tail32 and needs_fine allocate the bounded projection. Full-width/no-fine
  paths do not allocate it and are unaffected.
- linear non-skip _prepare_index_graph already has iter3 maintenance; do not
  duplicate that kernel or claim this change accelerates non-skip.

PR10 already fuses reading projection pools, but neither of the two projection
maintenance writes above. H=8,d=32,bf16 moves 512 bytes source and 512 bytes
output; the removed projected intermediate is another 512-byte write/read per
row. These are tensor payload calculations, not measured DRAM transactions,
GPU latency, allocator savings, or end-to-end benefits. Mapping/launch overhead
may matter more than payload. Existing persistent projection capacity is unchanged.

## Implementation and alternatives

Opt-in TLI_FAR_SKIP_KERNELS=triton adds project_token, a pure selected-dimension
load/store with ring mapping, writing the existing projection storage directly.
One launch per row, no scratch allocation or host device-value read. Integer
positions use do_not_specialize (no JIT specialization per token); device int64
positions allow warmup/repeated position/replay. Eager append's same-length no-op
and gap rebuild remain unchanged. EOS rows still refresh their unchanged projection;
this patch does not eliminate that work or fuse all batch rows.

Compared existing gather_rows (writes a contiguous result, cannot target a dynamic
ring slot), write_pair (no dimension projection), iter3 maintenance (linear layout,
block stats), and _qat_quant_kernel (linear multi-token QDQ and its own rounding
contract). Reusing them directly requires changing unrelated output/semantic
contracts. A small no-arithmetic direct copy kernel is bounded and independent.
Torch index_select/copy remains the default baseline. Quantized projection still
uses original QDQ; no rounding approximation, relaxed threshold or new external
kernel dependency. No claim that torch.compile or a vendor scatter was benchmarked.

## Executed checks (CPU-only Torch 2.8 / Triton 3.4)

- CPU projection/subspace/hybrid regressions: **44 passed, 23 skipped** (cpu.txt).
- TRITON_INTERPRET=1, NumPy2.2.6: **12 passed, 11 skipped** (interpreter.txt).
  Includes 7 Torch/fallback cases and 5 interpreter cases: four exact copy
  scenarios plus a 44-step selector integration comparing selected IDs/validity
  and the full projected cache to the Torch path. Not GPU execution.
- **48** SM80/SM90 specializations compile: 18 new projection combinations
  fp32/fp16/bf16 x i32/i64/device-int64 x two architectures, plus 30 existing
  movement kernels. compile.json binds source SHA256 and emitted PTX hashes.
- Exact rtol=0/atol=0; arbitrary ordered/strided dims, strided K/output, multiple
  wraps, same-position rewrite, position 2**33, independent new-request storage.
- Syntax checks passed. CUDA paired benchmark refuses CPU timing substitution.
- E127 indexer and batch gates attempted, blocked by CPU-only Torch/no GPU;
  raw logs retained. CUDA BF16, eight ragged rows/EOS-stable keys and second
  capture tests are present but **not run**. No independent reviewer or GPU
  performance/quality/memory verdict claimed.

Commands executed from repository root:
```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_far_skip_projection_write.py tests/test_far_skip_subspace.py tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py
TRITON_INTERPRET=1 PYTHONPATH=/path/to/numpy-2.2.6:. OMP_NUM_THREADS=1 python -m pytest -q tests/test_far_skip_projection_write.py
python -m exp.bench.compile_far_skip_kernels --out /tmp/projection-compile.json
```

## Remote acceptance

```bash
TLI_FAR_SKIP_KERNELS=triton OMP_NUM_THREADS=1 python -m pytest -q tests/test_far_skip_projection_write.py tests/test_far_skip_subspace.py tests/test_far_skip_graph.py tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
python -m exp.bench.far_skip_projection_compare --batches 1 8 32 --steps 100 --repeats 6 --out /tmp/projection-paired.json
```

Benchmark freezes the pre-change main expression, compares same K/dims/positions,
checks exact full cache equality, alternates arm order and reports event/wall
samples for staged eager, tensor eager and Graph. Source hashes are recorded.
It repeats a current-token write to isolate maintenance; it is not model TPOT or
a ragged-serving throughput benchmark. Run sanitizer and full-model same-prompt,
same per-request gate/geometry comparisons; include setup and allocated/reserved/
peak memory, source identity and route receipts before promotion. Keep opt-in
until target-GPU correctness and paired latency are accepted.

## Remaining coverage

Prior summary-audit SWA0/rewrite counterexample still blocks page-ID-only reuse.
Conditional summary refresh and linear mask->topk reconstruction remain candidates,
not completed optimizations. Existing linear gather scratch lifetime tradeoff and
strict compact selector quality requirement remain. No duplicate analysis-only
commit; this record accompanies the concrete projection implementation.
