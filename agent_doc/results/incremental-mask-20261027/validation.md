# Hybrid input mask maintenance — 2026-10-27

Main analyzed: 17e56e8fe30cf8037caf3ce7b38040e754f0521e. PR10 start:
ddf0c4d95b049f75daccf01dfac55eb764f63a8c, open, no reviews/comments.
Main/other PR/branch heads unchanged; CLAUDE83dbec28 unchanged, no AGENTS.
Read projection-write and earlier coverage records; no duplicate kernel work.

New coverage: RULER _decode_hybrid_batch_chunk -> runner.inject -> run/capture
-> hybrid rows and native dense layers. RULER stops when all rows finish; before
then EOS rows retain compute, not just storage. Native dense layers consume the
full causal mask, so deleting the mask or shrinking it to an identity marker is
not generally safe. The runner's pre-injection validation enforces +1 length for
live rows, unchanged lengths for dead rows and no resurrection. Every runner is
new per request batch. Current main and PR10 nevertheless rebuild [B,1,1,S_cap]
mask by zero_, capacity-wide comparison and masked_fill_ on EVERY inject, outside
Graph replay. This is a previously uncovered O(B*S_cap) input-maintenance path.

Change: first inject retains full initialization; later injections scatter zero
into [row,length-1]. Dead rows rewrite an already-valid slot. Reuse PyTorch's
existing GPU scatter implementation; a new Triton kernel would add maintenance
without removing another operation here. Existing far_skip movement kernels have
KV/dimension contracts, not this mask write. No new allocation or dtype/numerical
change. Same tensor pointer and full shape preserve dense consumption/capture.

At B32/S131072/bf16, mask payload remains 8MiB; the old comparison materializes
4MiB of bools and writes across the full mask. New logical mask writes are 64
bytes/step (32 elements), but actual GPU memory transactions, allocator effects,
launch costs and latency are NOT measured. Mask storage is NOT eliminated.
Other inject H2D inputs, per-row index/attention, EOS compute and model operations
remain. Both eager and Graph callers execute the same inject method.

Verification (CPU Torch2.8, Transformers4.56.2): 19 passed,12 skipped (cpu.txt).
Exact full-rebuild oracle for B1/8/32, fp32/bf16, 128 sequential steps, two fresh
runners, EOS, capacity boundary, invalid rewind and stable pointers. CPU profiler
confirms steady inject has scatter_ and no ge/masked_fill_; not a GPU launch census.
Existing real tiny-Qwen3 regressions cover mixed ring/linear/native dense, EOS,
ragged batches, repeated requests and production entry. Six new CUDA cases and
six existing CUDA model/graph cases skipped. Syntax checks passed. Paired CUDA
benchmark refuses CPU substitution. No GPU speed/memory/quality result; no new
GPU kernel and no new offline-compilation claim. E127/batch GPU gates remain
pending for overall PR acceptance. Temporary test environment was recreated;
initial missing dependencies resolved before the successful run.

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_hybrid_incremental_mask.py tests/test_hybrid_batch_cache.py
python -m exp.bench.hybrid_mask_compare --batches 1 8 32 --capacities 32768 131072 --steps 100 --repeats 6 --out /tmp/mask-paired.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

Benchmark freezes old full reconstruction versus incremental scatter, exact
outputs, paired inputs, alternating arm order, source hashes, event/wall samples.
It measures fixed-position steady mask update only, not full inject/TPOT. Remote
acceptance needs full runner inject+model wall comparison in eager and Graph,
including first-step cost, EOS/recapture tests and memory profiler. Mask is owned
by the runner; arbitrary caller mutation between steps is unsupported.

Next: conditional summary refresh with rewrite counterexample; direct linear
selection-ID contract; EOS compute requires device-aware kernels or rebatching,
not a host bool(alive) inside capture. No guide/paper/sglang/agent changes.
