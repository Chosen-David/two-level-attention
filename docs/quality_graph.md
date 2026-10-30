# Quality generation with CUDA Graphs

`TLI_CUDA_GRAPH=auto` is the quality-entry default. `0`/`off` uses
the eager reference; `1`/`on` requires capture and raises if unsupported or
capture fails. A selected environment variable is not evidence of execution.
Inspect `model._quality_graph_execution`: `actual`, `captures`, `replay_steps`,
`fallback_reason`, and per-layer `capture_layers` describe the path actually
captured. Deferred attention errors are checked before predictions are committed.

Each request owns its fixed-address KV, Graph and frozen gate state. Selection
uses the complete deduplicated candidate domain, including all live sink/SWA
tokens in the normalization denominator in both eager and Graph paths. It
preserves model-dtype probability rounding, per-request live normalization,
GQA reduction, native TopK ties and zero-probability boundaries. Unsupported
selector geometry has explicit fallback receipts; it does not silently relabel
the full-width reference as candidate-only selection.

The exact fused Qwen batch route is explicit. See
[exact_linear_batch.md](exact_linear_batch.md) for its configuration, supported
geometry and ordered attention reduction. Dynamic GLM gating at B>1 is not
supported. Ring serial/overlap consumers have separate eligibility and receipts;
overlap is not assumed faster than serial. See
[operator_overlap.md](operator_overlap.md).

## Decode input updates

`TLI_DEVICE_DECODE_INPUTS=1` enables a row kernel that updates tokens, positions,
lengths, live flags and the decode mask on device after successful capture, for
capacity at most 65536. `0` retains host preparation. The receipt includes
`input_update_requested`, `input_update_route` and `input_update_fallback`.
Requests without a captured Graph or above the capacity limit retain the host
path. Sampling remains greedy argmax; token readback, host stopping decisions,
EOS/min-generation semantics and capacity checks remain in the generation loop.

`TLI_REUSE_WARMUP_STREAM=1` reuses one warmup stream per calling thread/device;
`0` creates a stream per request. Stream reuse retains neither request KV nor
Graph ownership. `warmup_stream` and `warmup_stream_reused` report actual use.
This bounds the stream registry per thread/device; it is not a measured bound
on all CUDA allocator or cuBLAS memory.

Both switches default to `1`; explicit `0` retains the control path. Input
fusion still requires actual capture and supported capacity. These defaults
do not imply a speedup for every model or request, and do not change Graph policy.

## Measurements and scope

`TLI_QUALITY_TIMING=1` records replay CUDA events and step wall time including
input updates, sampling, readback and EOS processing. Instrumentation has
overhead and is off by default. Full request time must also include prefill,
KV conversion and capture; `setup_wall_ms + decode_wall_ms` does not include
every request cost. Batch step TPOT is distinct from output-token throughput.

The real-model harnesses in `exp/bench` retain every paired request and actual
route. `check_quality_batch_qwen.py --device-input-reference` and
`check_quality_graph_glm.py --device-input-reference` compare input/stream modes
under the same successfully captured Graph (GLM requires strict capture;
Qwen uses auto policy and checks actual capture). The GLM harness otherwise compares eager/Graph.
Repeated pairs alternate AB/BA; the first process cycle includes startup effects
and is reported separately. These diagnostics do not establish whole-benchmark
quality, significance, continuous-serving throughput or memory boundedness.

Canonical raw data and independent scope-limited reviews are stored under the
main project's `agent_doc/results/`. G4 Qwen runs 1070/1080/1090 show small TPOT
reductions, with mixed or slightly worse total-request times. Do not describe
them as full-request speedups, combine incremental speedups, or reuse their
numbers as measurements of later source revisions.
