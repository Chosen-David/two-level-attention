> Correction 2026-10-27: the 4,734,976-byte figure below is payload, not the old backing-storage bound. Geometric rounding allocated 8,388,608 bytes. Fixed-shape allocation now removes that slack; see ../workspace-capacity-20261027/validation.md. No GPU allocator measurement is implied.

# Linear gather audit — 2026-10-27

Main 17e56e8fe30cf8037caf3ce7b38040e754f0521e; prior PR10 b2463d4.
Root CLAUDE and prior task/summary audit reviewed; main runtime unchanged from
b0a378c, new E178 files retained. No PR reviews/comments or parallel branch moves.

Traced hybrid decode -> compute_mask_graph -> _attend_linear. The selector emits
[1,1,Hkv,S_cap] boolean mask; consumer casts it to int32 and runs full-width topk
before gathering original-width K/V. At S_cap=131072,Hkv=8 this is 1MiB mask plus
4MiB int32 temporary per row, excluding topk scratch. These are shape-derived
bytes, not measured allocator peaks. Direct candidate-ID return could remove
this scan, but must preserve duplicate removal, sink/SWA, short valid lengths,
oversized L2 zero-score filling and SDPA order. Defer that interface change.

Adopt a smaller exact movement optimization: use existing PR10 gather_rows
kernel through far_skip_accel.gather(layout=None, other=v, valid=valid). It loads
K/V in one launch, suppresses invalid loads, writes zero padding directly and
keeps the same positions/topk/SDPA operation. Replaces two advanced gathers and
two masked_fill operations. No new Triton code, thresholds or default switches.
Each linear request row owns a separate workspace initialized before warmup.
TLI_FAR_SKIP_KERNELS=triton is opt-in; absent workspace preserves old API behavior.

Persistent K/V output scratch is bounded by 2*Hkv*min(S_cap,budget)*D*element_size
(4,734,976 bytes for Hkv8,D128,budget1156,bf16); it lives until the batch cache is
released, unlike eager ephemeral gather tensors. This is NOT a memory-saving
claim. Workspaces are per row and must not be shared across in-flight requests.
Existing ring consumers already use this kernel; old PR10 linear consumer did
not. Whole hybrid index/attention still loops rows.

Validation: CPU integration 28 passed/7 CUDA skipped; interpreter exact-output
and scratch-address tests 6 passed/13 skipped. Covers fp32/fp16, short/partial
lengths, budget above length, all-empty selection, NaN invalid slots, differing
K/V strides and fixed pointers. BF16 and CUDA capture with changed masks and
new workspace lengths remain device-only tests, not executed here. Syntax and
whitespace checks pass. No GPU available; benchmark explicitly rejects CPU.
Existing E127/batch GPU gates remain pending (prior attempts CPU-only blocked).

Remote commands:
```
TLI_FAR_SKIP_KERNELS=triton python -m pytest -q tests/test_hybrid_linear_gather.py tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
python -m exp.bench.hybrid_linear_gather_compare --lengths 32768 131072 --steps 100 --repeats 6 --out /tmp/linear-paired.json
```
Benchmark compares identical mask/topk/gather/SDPA inputs, alternates arm order,
records source hashes, reports eager/Graph event/wall times separately. This is
one-row microperformance, not full-model TPOT. Full batch GPU/EOS/recapture,
sanitizer, allocated/reserved/peak and actual model paired TPOT remain gates.
No new quality/performance claim. Latest main evidence is preserved, including
E178 data; prior E177/E178 scope warnings are not overwritten.

Next: direct sparse-selection contract with explicit ordering/duplicate rules;
conditional compact summaries with SWA0 repeated-position guard from prior audit.
