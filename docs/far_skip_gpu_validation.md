# GPU revision after the eager regression report (2026-10-26)

PR #7 is an experimental compact far-off implementation, not a demonstrated
GPU speedup. CPU timings from its earlier rounds did not establish that the
extra ring indexing, dispatch and scratch traffic are worthwhile on H20.

The review on `perf/gpt-kv-eviction` at `ee44158` contains an older H20 result:
evict ~1.11 ms vs retain ~0.97–1.02 ms for a synthetic layer. Its graph rejection
description predates PR head `b2c79ba`. Its full-model arms used different gate
settings, so those numbers cannot establish a matched end-to-end improvement
or identify the latest PR's exact regression. They are retained as negative
motivation, not reused as validation of this revision.

## Changes and expected GPU effects

| Change | Removed work | Preserved semantics |
|---|---|---|
| StaticCache staged eager uses host-length selector | Per-layer position H2D, tensor boundary arithmetic, padded candidates, eager execution of graph maintenance | Same logical near/protected region and original-width attention |
| Triton stable-order encoding | Separate normalize/cast/bitwise/NaN/tie/padding kernels and their intermediate buffers | Original fp32 total order, ±0 ties, NaNs first, valid lanes before padding; Torch top-k and final index sort remain |
| Triton fused gather | Materialized physical indices, separate K/V mapping, index/broadcast intermediates, padding-zero passes | Actual K/V strides, original dtype, exact copies; invalid lanes never read KV |
| Triton fused append | Separate K/V writes and tensor ring mapping | One logical token written to one physical slot, no old-KV copy |

No dot-product reduction, softmax, QDQ formula, token budget or gate decision
was replaced. Kernel launches are structurally fused; no GPU latency or launch
count has been measured in the development environment. Selected K/V are still
materialized before SDPA. Query/projection/QDQ and top-k remain optimization
candidates. This is not the production iter3 selector's semantic equivalent.

Default kernels remain `torch`. Enable the experimental GPU kernels explicitly:

```bash
export TLI_FAR_SKIP_KERNELS=triton
```

This applies to compact DynamicCache eager, staged StaticCache eager and the
fixed-position graph route. Opt-in compilation/execution failures propagate;
there is no catch-and-silently-benchmark-Torch fallback. Torch is the explicit
rollback setting. The setting must remain fixed for a request/captured graph.
Kernel scratch is request-local and reusable, but consumes additional bounded
memory. Measure `workspace_storage_bytes`; the earlier 12.5% fine-key-cache
figure does not include these new full-width K/V output buffers.
Candidate counts, changing strides and host token positions are non-specialized
runtime scalars. They must not cause a new Triton specialization per decode
position. Dtype, feature width, validity/projection mode and ring geometry still
specialize legitimately; warm up every route used in the measured workload.

## Reproduce on an otherwise idle target GPU

Keep the old PR checkout at `b2c79badf0a1b000ea130c1b46f596bfaa189eb3` as
`/tmp/far-parent`. Use the candidate checkout as the working directory.
First run correctness, including actual graph replay:

```bash
TLI_FAR_SKIP_KERNELS=triton python -m pytest -q \
  tests/test_far_skip_gpu_kernels.py tests/test_far_skip_compact.py \
  tests/test_far_skip_graph.py tests/test_far_skip_subspace.py \
  tests/test_far_skip_ruler_entry.py
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
```

CPU interpreter cases in the first command skip intentionally. They must not
substitute for CUDA tests. The CUDA tests include BF16 selector decisions,
AVG/minmax, QDQ on/off, changing inputs, ring wrap and real capture/replay.

Compare the old PR to this candidate with the *same* selected-token semantics:

```bash
python -m exp.bench.far_skip_subspace_compare \
  --baseline-root /tmp/far-parent --device cuda --dtype bfloat16 \
  --kernels triton --baseline-kernels torch --modes eager staged \
  --heads 32 --kv-heads 8 --sink 4 --window 128 --widths 32 \
  --length 32768 --near-len 4096 --near-pages 32 --budget 1024 \
  --steps 128 --warmup 16 --repeats 6 > /tmp/paired-eager.json
```

Also compare new Torch against new Triton with `--baseline-root .` to isolate
kernel effects. Repeat at 8192/65536/131072, short prompts, and with `--quantize`.
For actual CUDA Graph timing use `--modes fixed --capture`. Capture warmup/JIT
are outside steady timing and setup/first/capture costs are reported separately.
`staged` invokes the real StaticCache adapter and compares outputs; `eager` and
`fixed` additionally compare selected IDs on every step. This synthetic harness
does not measure model throughput or downstream quality.

Run `--profile` separately. The receipt includes CUDA device events and CPU ops;
the instrumented step is excluded from latency samples. Compare normal timing
runs first, then inspect actual `order_keys`, `gather_rows`, `write_pair`, SDPA
dispatch, host gaps and remaining top-k/softmax. Reject a fused configuration
that fails correctness or loses on the target workload; fewer launches alone
are not an acceptance criterion.

Finally run the Qwen/RULER entry with identical explicit far-off layer sets,
near/K2/subspace settings, inputs and stop policy for both arms. Record source
paths/hashes (including shadow packages), actual gates/layer types, complete
request time, TPOT, post-prefill/post-first-decode allocated and reserved memory,
and peak memory. Compare main separately as an algorithm/quality baseline;
strict far exclusion changes boundary behavior, so matching its outputs is not
assumed. RULER quality and full-model GPU speed are release gates, not completed
results of this PR revision.

## Development evidence boundary

`agent_doc/results/far_skip_gpu_revision_20261026/` records CPU regressions,
Triton interpreter checks and offline SM80/SM90 compilation with source hashes.
Interpreter execution uses NumPy/CPU, not a GPU simulator or a GPU timing tool.
The environment has CPU-only Torch; CUDA correctness, E127/batch GPU gates,
kernel latency, SDPA backend choice and whole-model performance remain pending.
