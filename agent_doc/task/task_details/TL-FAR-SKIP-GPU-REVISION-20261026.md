# Far-skip GPU revision after eager regression

Parent: PR #7 `b2c79badf0a1b000ea130c1b46f596bfaa189eb3`.
User requested a GPU-oriented upgrade after remote eager slowdown.

The earlier rounds optimized CPU behavior without establishing GPU deployment
benefit. GPU launches, ring-index intermediates, padding and layout costs were
not measured. The alternate `perf/gpt-kv-eviction` branch's E175 review provides
older H20 evict > retain latency evidence, but no current-head matched profile.
Its full-model gate settings differ, and its capacity arithmetic is internally
inconsistent (sink4/near4096/window128/block64 gives 4291, not 4551 or 4415).
Adopt the regression signal; reject transferring its speed/footprint claims to
this revision or assuming it tested the current static/graph implementation.

Implemented:

- StaticCache eager consumes the host length and dynamic eager selector; no
  graph-padding work, per-layer device position construction or graph side-cache
  allocation. Replay remains tensor-position-driven. Staged position reuse,
  rewind and gaps reject before KV mutation.
- Optional Triton kernels fuse total-order keys, ring mapping + projected/pair
  gather, and single-token K/V writes. They preserve stable tie rules, validity,
  original dtype and arbitrary supported strides. No attention/QDQ arithmetic
  rewrite. Default Torch backend remains available.
- Changing lengths/strides/positions are not JIT-specialized per token. GPU
  scratch is bounded and reused; extra output buffers are explicitly accounted.
- Paired harness includes Dynamic eager, actual staged adapter and captured
  fixed path; explicit GPU backend/config/source hashes and CUDA profiler
  events. Profiled steps no longer contribute to timing samples.

Validation in `agent_doc/results/far_skip_gpu_revision_20261026/`:

- CPU regression: 134 passed, 205 subtests; five CUDA cases not run.
- Triton interpreter: 20 passed, 43 skipped (CUDA and unsupported BF16 CPU
  interpreter cases). Independent stable-order/copy checks plus selector ID and
  output comparisons across wrap, AVG/minmax and QDQ on/off. This is CPU logic
  evidence, not CUDA correctness or performance.
- Triton 3.4.0 offline compilation: 18 SM80/SM90 specializations produced cubins.
- Same-input CPU comparison: 320 outputs, max absolute error 0. Static eager
  median 10.0178→5.9883 ms. Dynamic eager 8.4553→7.9928 ms
  (-5.47%; no GPU claim). CPU values do not predict H20 gains.
  Normal timing and profiling are separate runs; CPU operation counts are not
  GPU launch counts. Earlier exploratory Dynamic eager was slower; the final
  modest reduction is not stable evidence of a Dynamic eager speedup. That
  earlier source/harness snapshot is retained separately, not pooled.
- No GPU/SSH execution capability in this environment. E127/batch CUDA gates,
  actual fused CUDA correctness/capture, matched GPU latency/memory, actual SDPA
  dispatch and full-model RULER quality remain pending. No independent review
  is claimed. Do not promote the Triton path based on fewer source operations.

The GPU acceptance commands, matching rules, rollback and remaining costs are
in `docs/far_skip_gpu_validation.md`. Keep this task open until deployment-GPU
results establish whether the reported eager regression is resolved.
