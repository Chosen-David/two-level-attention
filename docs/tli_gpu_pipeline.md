# GPU producer/consumer fusion candidate (2026-10-26)

Parent: PR #7 `2e21174b6bd41c6712f92c3ff0b5f41661bf7fdf`.
The user requested actual GPU fusion for remote validation, beyond Torch-side
rearrangement. This revision adds four Triton kernels in
`sparse_attn/indexer/tli_batch_pipeline.py`, reached from the full-batch indexer
used by both Qwen staged eager and CUDA Graph. No GPU was available locally.
Status: **implemented/compiled; GPU correctness, speed and model quality pending**.

## What is fused

| Kernel | Removed intermediates / separate work | Scheduling and remaining cost |
|---|---|---|
| `_update_cache` | Projected page and fine staging buffers; separate amin/amax/mean launches, advanced-index cache stores, previous-length copy | Grid `(feature tile, B*Hkv)`, 32 feature columns per tile, FP32 page reduction; directly writes cached summaries, fine token and lengths. Reads only selected feature columns. Still scans the ending page every step. |
| `_queries` | Separate index_select, scale and FP32 cast; two launches/projections for differing coarse/fine spaces | One program per query head, preserves input-dtype rounding before FP32. Shared spaces share the output tensor. |
| `_candidates` | Page gather/expand, validity expansion, bypass assembly, concatenation, full unpartitioned IDs/flags, subsequent compact pass | One program per batch/KV head; count live tokens at page granularity, then scan/scatter in chunks of 256. Stable order of both valid and invalid slots is preserved. |
| `_mask_top` | Selected-ID gather, selected-validity gather, rank and budget temporaries | Directly consumes top-k slots and writes final mask. A separate `_mask_base` launch initializes forced tokens first; this ordering avoids a cross-block initialization/scatter race. |

For B=4, Hkv=8, page size=64, d=32, BF16, the old projected page
alone is 131,072 bytes. Direct cache update eliminates its materialization and
subsequent reads; the fine staging buffer is another 2,048 bytes. These are
logical tensor byte counts, **not measured HBM traffic or latency**. Coalescing,
cache lines, reuse and layout affect actual transactions. Full KV/fine-cache
capacity, the final dense boolean mask and the attention consumer are unchanged.

The initial full-row candidate scan compiled to **255 registers/thread** at
R=4224. It was replaced by page counts plus a bounded token scan: the same
compiled matrix now uses at most **60 registers/thread** for `_candidates`.
The original source/compiler receipt is preserved in `full_scan_trial/`.
This reduces static register pressure; it does not establish faster execution.
The bounded scan adds a serial chunk loop and must be timed on the target GPU.

Final offline resource ranges (SM80/SM90, Triton 3.4.0, 4 warps):

| Kernel | Registers/thread across compiled cases | Spill cases |
|---|---:|---:|
| `_update_cache` | 32–96 | 0 |
| `_queries` | 10–18 | 0 |
| `_candidates` | 18–60 | 0 |
| `_mask_top` | 24 | 0 |

These are ptxas compilation reports, not runtime occupancy or Nsight counters.
The report retains every signature, constant, PTX hash and compiler resource log.

## Route and numerical contract

```bash
export TLI_BATCH_INDEXER=parallel
export TLI_BATCH_SCORE=triton
export TLI_BATCH_FUSIONS=1
export TLI_BATCH_GPU_PIPELINE=1
```

All switches are opt-in. The new flag requires the existing fusion bundle and
`l2_quant=False`; unsupported combinations raise rather than silently use an old
path. QDQ, dynamic projections, B>1 physical KV eviction and other exclusions of
the original batch API remain outside this candidate. Core support is decode
Q[B,1,Hq,128], K[B,S,Hkv,128], static dimensions, device int64 lengths, finite
valid KV, poisoned padding allowed, append by zero/one, fixed storage identity.
Warm cold state/JIT outside capture; clear and recapture on refill/reorder.
No per-request Python loop is added. No host read of live lengths is added.

Candidate ordering, bypass handling, quotas and mask domains are unchanged.
Projection/scaling keeps the previous dtype-rounding boundary. Min/max and fine
copy gates are exact; AVG uses a different FP32 reduction tree and can differ by
roundoff. It is **not a blanket bitwise-compatibility claim**: the remote harness
still requires exact output masks and exact same-device attention outputs. A
top-k change from rounding fails that gate; do not relax it to claim a speedup.
Existing batch-versus-legacy numerical/tie limitations still apply.

## Remote commands

From the repository root, on a GPU host with the repository dependencies and
compatible Triton installed, start with:

```bash
python -m exp.bench.run_tli_gpu_pipeline \
  --outdir /tmp/tli-gpu-pipeline-smoke \
  --batches 4 --lengths 32768 --steps 128 --repeats 6 --profile
```

The runner rejects CPU and `TRITON_INTERPRET=1`, checks E127/batch repository
gates and the new real-CUDA unit/capture gates first, then runs eager and Graph
in fresh processes. It compares **score-only → new pipeline** and **previous
movement bundle → new pipeline**, with identical math/config/data/score backend.
It verifies requested dispatch flags are active. These are ablations within the
parallel candidate, not an old full-model Graph baseline. A failure is retained
and returns nonzero; a fresh output directory is required on rerun.

Expand the pressure matrix after the smoke gate:

```bash
python -m exp.bench.run_tli_gpu_pipeline \
  --outdir /tmp/tli-gpu-pipeline-stress \
  --batches 2 4 8 --lengths 32768 65536 131072 \
  --layouts head token --l2-subspaces same full \
  --steps 256 --repeats 6 --sanitizer
```

`--sanitizer` additionally requires NVIDIA compute-sanitizer and runs memcheck
on the new kernel gates. OOM/compile/launch/correctness failures are failures,
not skipped successful results. This matrix can be substantial; use available
GPU capacity and retain failed-case receipts. `--methods avg minmax mixed`
and `--dtype float32|float16|bfloat16` provide further explicit coverage.

For one isolated old-movement/new-pipeline pair:

```bash
python -m exp.bench.tli_batch_round_compare \
  --baseline-root sparse_attn/indexer --device cuda --dtype bfloat16 \
  --score triton --baseline-fusions --fusions --gpu-pipeline \
  --method mixed --batch 4 --length 32768 --steps 128 --repeats 6 \
  --profile --out /tmp/tli-pipeline-pair.json
# Add --capture to measure actual Graph replay.
```

Expected receipts: `fusion_active=[true,true]`,
`pipeline_active=[false,true]`; profiler candidate trace should contain
`_update_cache`, `_queries`, `_candidates`, `_mask_top`, and should no longer
contain `_gather_update`/`_compact`/`_mask_selected`. Trace files are exported
beside the receipt. The remaining `_score`, topk/softmax and base-mask kernels
are expected. Use Nsight Systems for gaps/launches and Nsight Compute on these
exact kernels for DRAM bytes, occupancy, stalls and spills when available.

Receipts include all wall/CUDA-event samples, cold time, source hashes, route
flags, exact mask/attention checks, unique persistent cache bytes and extra
allocated-byte peaks. Event intervals include enqueue gaps; inspect traces to
separate them from kernel execution. Allocation peaks are PyTorch allocator
observations, not HBM traffic or total model memory. Graph private-pool allocation
mostly happens at capture, outside steady-state peaks. Timings exclude Q/KV
updates, length upload, attention and correctness checks. Run the unchanged
legacy full-model eager/Graph and candidate full-model eager/Graph on identical
checkpoints/requests before accepting TPOT/throughput or quality improvements.

## Evidence and remaining work

`agent_doc/results/tli_gpu_pipeline_20261026/` contains source-bound receipts:

- 205 CPU tests + 205 subtests passed, 138 skipped; default paths regressions.
- 95 interpreter checks passed, 33 BF16 cases skipped. This includes 40 new
  pipeline checks and 55 existing fusion/score checks; no CUDA execution.
- 188 new SM80/SM90 specializations compiled, including FP16/BF16/FP32,
  head/token layouts, coarse/fine width combinations and unused-stat variants.
- CPU harness smoke validates packaging/default-off behavior only. No CPU
  timing is used to claim any GPU gain. The GPU runner fails closed locally.
- Initial compilation issues (unsupported Python bool in Triton, ptxas target
  suffix, loop-carried vector shape) were fixed; attempt logs are diagnostic
  only. The final compiler and interpreter receipts bind the delivered source.

Still substantial: L1 GQA mean, topk, fine softmax/mean, geometric mask work,
full-width final mask, duplicated full-width fine cache in the full-L2 mode,
and selected-KV/attention traffic. Fusing fine softmax/GQA needs a separate
rounding/tie gate; fusing selection directly into an attention consumer is a
larger interface change. Neither is claimed implemented here. An incremental
sum/min/max update could avoid rescanning the page but changes accumulation
order further. Default enablement and any “fastest” claim wait for target-GPU
measurements, full-model regression/quality, and independent acceptance.

Design references checked 2026-10-26 (mechanism references, not measured baselines):

- [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html): keep intermediates on chip, inspect registers/occupancy, account for approximate exponentials before changing selection.
- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalesced access and register/occupancy tradeoffs must be evaluated with measurements. No external kernel source was copied.

## Subsequent skip/non-skip and residency revision

See [`tli_skip_noskip_fusion.md`](tli_skip_noskip_fusion.md). It adds region-aware
L1/GQA/mask fusion, B=1 fixed-capacity dispatch and frozen-gate admission, tests
both skip states, and audits physical cache release separately. Earlier counts
and limitations above describe commit 80e151e; current evidence is in the new
revision result directory.
