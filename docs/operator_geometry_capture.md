# Capture-scoped selector geometry reuse

`capture_graph()` gives each actual CUDA Graph capture a fresh geometry-cache
namespace. Within that capture, selector layers with the same live length buffer,
mutation version, producing stream and geometry reuse the first layer's device
computation. Different captures, raw Torch captures, different streams and
version-less inference tensors retain conservative recomputation. Existing graph
pool ordering and fixed-buffer lifecycle requirements still apply.

The four repository capture entrypoints use this wrapper. Set
`TLI_GRAPH_GEOMETRY_SHARED=0` to disable sharing for an ablation; the default is 1.
This does not change selection scores, softmax denominators, rounding or ranking.

## Verified scope (2026-10-30)

Measured runtime source: `ef40686`, based on main `309519a`. Qwen3-8B BF16, H20-3e,
B=1, 32768 prompt tokens, 64 generated tokens, cache capacity 32840; no far/near skip,
serial SIMT attention, exact sparse selector and whole-forward CUDA Graph in both
arms. Only the geometry-sharing switch differs. The reserved GPU passed idle
admission before each process, with exclusive-process telemetry throughout.

Two ABBA cycles used eight processes, each with six warmup and three measured
requests. No samples were removed. All 72 requests generated identical 64-token
outputs. All 36 layers used the sparse selector; first-capture geometry builds
were 36 with sharing off and 1 with sharing on. Later requests reused the graph.

| Decode metric | Sharing off | Sharing on |
|---|---:|---:|
| Median of process-median TPOT | 20.177443 ms | 17.806836 ms |

Latency decreased 11.75%. The preregistered four-pair geometric-mean speedup was
1.133366x, with a Student-t 95% interval of [1.131948, 1.134785]. Every process passed
the 2% warmup/measurement drift and 3% measured-range gates. This interval has
only four process pairs and describes this workload and machine.

Request wall-time variability failed its separate gate, so no request-time gain
is established. Measured requests reused the graph (`capture_ms=None`), so no
capture-time gain is claimed. This is not a FlashAttention2, ring, overlap, batch
throughput or model-quality comparison. Native final-source Graph regressions:
30 passed; CPU suite: 43 passed, 14 CUDA skips, 15 subtests passed.

Independent evidence is retained under the project directory
`agent_doc/results/operator_geometry_main_20261030/`: the frozen protocol,
`native/`, and `paired/independent-review.md`, JSON review, raw samples, source
snapshot, commands and telemetry. The review binds source hashes and exit status;
a successful process alone is not acceptance.

Reproduce the paired workload on the reserved GPU1 environment with:

```bash
PYTHONPATH=. python exp/bench/operator_model_paired.py \
  --arms graph_sparse_geometry_off graph_sparse \
  --cycles 2 --warmup 6 --repeats 3 --max-gen 64 \
  --out-dir /absolute/path/to/a/new/result-directory
```

The runner requires the existing GPU1 lease. Results remain pending independent
validation; do not infer acceptance from this command's exit code.
