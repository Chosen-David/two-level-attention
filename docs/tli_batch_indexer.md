# Full-batch TLI indexer candidate (2026-10-26)

PR #7 parent: `9dbc6e9c8dada0abedee99a96ce79967dec0dd9f`.
This revision addresses **indexer serialization**, separately from the B=1
compact far-eviction path. Default behavior remains `TLI_BATCH_INDEXER=legacy`.
It is not a claim that the previous GPU eager regression is resolved.

Further iterations: [five-round evidence and fusion switch](tli_batch_five_rounds.md).

## What was serialized

`qwen3_attn_patch._tl_decode_batched` cleared the indexer and called
`prepare_mask` once per request. Only the attention consumer was batched.
`_tl_decode_batched_graph` likewise looped through single-row index preparation
and mask computation; capture removes host dispatch during replay but retains
that per-row device work. TIA/TWI assert B=1; Quest's scoring squeezes batch and
reads `q[0]`/the first sequence interval. None is a drop-in complete-batch TLI
selector. Existing batch attention kernels are the reusable consumer.

The new public API is:

```python
mask, block_size = indexer.prepare_mask_batched(
    q, k, valid_lengths, softmax_scale=128**-0.5)
# q [B,1,Hq,128], k [B,S_capacity,Hkv,128], valid_lengths int64 [B]
# mask [B,1,Hkv,S_capacity], block_size=1
```

There is **no Python request loop** in this path's index maintenance, scoring,
L1 selection, candidate assembly, L2 selection, mask assembly or metrics
reduction. Operations retain batch as a tensor dimension. In Qwen integration:

```bash
export TLI_BATCH_INDEXER=parallel
export TLI_BATCH_SCORE=triton  # optional GPU scorer; torch is the default oracle
# Optional new data-movement fusion bundle; default off until GPU validation:
export TLI_BATCH_FUSIONS=1
```

The same API is wired to staged eager and Graph for B>1. B=1 retains the old
route. Unsupported configurations raise rather than silently run the old
selector. Backend settings must be chosen before request initialization.

## GPU work and memory

- Initial preparation creates persistent `[B,pages,Hkv,d]` block statistics and
  `[B,S,Hkv,d_fine]` fine features. Steady decode reads only one ending page and
  one new token per row for maintenance. It does not clear/rebuild per row.
- Eager uploads lengths once per step/device using the request context and
  shares that tensor across layers. Graph reads the caller's live device buffer.
- Torch scoring uses batched contractions with GQA query groups, without
  materializing a full-width `[B,Hq,S,D]` repeated K tensor.
- Optional Triton scoring schedules `(token tile, B*Hkv)` programs in one launch
  per score operation. One K tile is reused across GQA query heads. Candidate
  scoring fuses indexed K loads with the dot product and removes the
  `[B,Hkv,R,d]` intermediate K gather. It uses the repository's FP32 reduction /
  output-dtype rounding pattern; correctness on actual CUDA remains pending.
- `_score.N` is runtime, not specialized by value. Grid uses fixed capacity and
  fixed candidate slots. No per-step length is a constexpr. Strides/dimensions
  are fixed for a request and may specialize at request initialization.
- Full boolean masks and top-k/softmax/sort temporaries still exist. Batch
  increases parallel work and live memory; this is not a paged allocator or a
  fused sparse-attention kernel. K/V cache capacity is unchanged by this path.
  The full-capacity fine side cache costs `B*S*Hkv*d_fine*element_size` bytes;
  coarse summaries and temporary scores add to that. Measure allocated,
  reserved and peak memory on the target GPU before enabling large batches.

## Contract and numerical limits

Fixed near pages/length, near_tokens=off, boundary bypass, static subspace,
non-projected GQA D=128, single query, valid-prefix fixed-capacity K/V,
FP32/FP16/BF16, inference only. Static skip-far is supported with zero far L1
quota and **legacy boundary-bypass semantics**. It is not strict compact far
exclusion: far summary storage/scoring still exists. Dynamic gates, per-query
heads, clusters, MoBA, async and compact eviction are rejected.

The previous compact far-eviction implementation remains B=1. This revision
does **not** extend its physical KV eviction to a ragged batch. That needs
per-request ring/page ownership and a batch-aware attention gather contract;
packing requests into a single ring would mix token positions and be incorrect.

One state owns one layer of one append-only batch. Lengths may advance by one
or stay unchanged (finished rows / capture warmup). Rewind, jumps and changed
KV storage are rejected. Row reorder, same-pointer refill, changing batch or
configuration require `indexer.clear()` and graph **recapture**. Warm up before
capture; no cold state construction is allowed while capturing. Prefill clears
state via the existing request-boundary hook.

This implementation follows the **fixed-slot Graph candidate formulation**,
not a bitwise promise for every legacy eager shape. Different softmax domains
and `topk` tie handling can change selected IDs. It additionally prevents reads
from invalid candidate slots, suppresses duplicate sink/SWA bypass pages for
short sequences, and excludes padding from the final mask. Original Graph's
behavior on those degenerate cases is not claimed as the desired oracle.
Selected IDs, attention output and final generated tokens all require gates.

## Reproduce and accept

```bash
python -m pytest -q tests/test_tli_batch_parallel.py tests/test_tli_batch_kernels.py

# Actual old eager loop vs new complete-batch path (indexer-only timing).
python -m exp.bench.tli_batch_indexer_compare --device cuda --dtype bfloat16 \
  --batch 4 --length 32768 --steps 128 --repeats 6 --score triton \
  --out /tmp/batch-old-eager.json

# Isolate batching: identical new algorithm/cache per row vs full batch.
python -m exp.bench.tli_batch_indexer_compare --device cuda --dtype bfloat16 \
  --batch 4 --length 32768 --steps 128 --repeats 6 --score triton \
  --reference serial-parallel --out /tmp/batch-isolation.json

# Real capture/replay of both isolation arms. Not the old optimized Graph arm.
python -m exp.bench.tli_batch_indexer_compare --device cuda --dtype bfloat16 \
  --batch 4 --length 32768 --steps 128 --repeats 6 --score triton \
  --reference serial-parallel --capture --out /tmp/batch-graph-isolation.json
```

The benchmark alternates same-input arms, records every sample, source hashes,
mask differences and attention error. Cold initialization is separate; profiler
runs are separate from timing samples. `--profile` records CPU/CUDA event
aggregates. CUDA events and host wall time are both reported on GPU. KV append,
length uploads, Q generation and attention are excluded from indexer timing;
**these are not model TPOT results**. B=2/4/8 and 32K/64K/128K should be measured
where memory permits, with frozen flags, masks, dtype, GPU and source versions.

Run repository E127 and TLI batch gates on CUDA, then compare **old Graph vs
new Graph** and old eager vs new eager through the full production harness.
Full-model RULER / generated-token parity, profiler launch counts, bandwidth,
GPU latency/throughput and memory remain mandatory before default enablement.
The existing gates exercise the legacy default too; explicitly set the new
route and assert the `tli_batch_indexer_parallel_{eager,graph}` receipt route
for new-arm full-model runs. No independent GPU reviewer has accepted it here.

## Evidence and source comparisons

Local receipts: `agent_doc/results/tli_batch_parallel_20261026/`.
The environment is Torch 2.8.0+cpu, no CUDA device. CPU interpreter checks and
SM80/SM90 cubin compilation do not prove CUDA execution or performance.
CPU results separate cache-maintenance savings from batching; do not attribute
all of the old-loop improvement to batch parallelism.

Final local CPU experiment: B=4, S≈4096, Hq/Hkv=32/8, D=128, tail32,
K1/K2=64/1024, 8 measured steps × 3 alternating repeats, one CPU thread.

| Comparison (indexer only) | Serial median | Batch median | Mask/output differences |
|---|---:|---:|---|
| Old eager clear + rebuild vs candidate | 89.590 ms | 22.994 ms | 0 / max abs 0 |
| Same new incremental algorithm, serial vs batch | 25.411 ms | 23.732 ms | 0 / max abs 0 |

Each comparison checked 96 row outputs. CPU profiler diagnostic aten call
counts were 2778→713 and 2861→713 respectively; these are **not GPU launch
counts**. Most old-eager CPU improvement comes from eliminating repeated
index construction. CPU measurements do not establish H20 batching speedup.
Final regression: 158 CPU tests + 205 subtests passed, 45 skipped; Triton
interpreter 24 passed / 12 BF16 skipped; 24 offline CUDA specializations built.

External design references inspected on 2026-10-26:

- [FlashInfer KV layout](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/tutorials/kv_layout.rst):
  per-request page metadata and batch-oriented execution, adapted as a design
  principle. Its batch attention wrapper does not implement TLI's two-stage
  scoring or remove this repository's selector loop by itself.
- [vLLM sparse indexer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/sparse_attn_indexer.py):
  dedicated batched indexer and backend dispatch, useful architecture reference;
  not a compatible replacement for TLI's minmax/AVG + GQA + boundary-bypass
  semantics. No external source code was vendored in this change.
