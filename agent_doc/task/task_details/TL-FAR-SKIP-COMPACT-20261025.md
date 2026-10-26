# TL-FAR-SKIP-COMPACT-20261025

## Plan

User scope: implement actual far-region compute skipping, include physical KV
storage savings, test correctness, submit a PR. Do not require GPU experiments
before PR delivery. Base: `2c726b54ad084ddb8f4a061954c6352afffb2da5`.

Adopt bounded near-only compute and request-local immutable eviction. Adapt to
the existing Qwen3 fixed-geometry API. Defer fused kernels, CUDA Graph, batching,
pretrained quality and hardware speed claims until target-GPU validation.
Do not change the human guide or manuscript or push to main.

## Progress

- Added opt-in compact near-page/L2/attention implementation and DynamicCache
  layer adapter. Sink/near/window K/V are independent allocations; logical
  positions continue growing. Only far-off layers evict.
- Integrated before ordinary cache update/index preparation in Qwen3. Added
  configuration guards and output treatment identity. Legacy defaults retain
  byte-identical manifests for default and fixed geometry configurations.
- Added real Torch/HF tests and a single-layer CPU/CUDA benchmark; usage and
  semantic differences are in `docs/far_skip.md`.
- Validation environment: Python 3.12.14, Torch 2.8.0+cpu, Transformers 4.56.2;
  target project Python is 3.11. CPU results do not establish GPU correctness,
  speed, pretrained accuracy, or full-model memory savings.
- New suite: 40 passed, one CUDA BF16 case skipped. Adjacent fixed-geometry
  suites: 10 passed, 205 subtests passed. Combined log is under
  `agent_doc/results/far_skip_20261025/`.
- Three older harness suites (`test_tli_runtime_shape_contract`,
  `test_qwen_static_mask`, `test_tli_async_candidate_gate`) report the same
  failures on this change and on untouched base sources: missing harness
  `_graph_state_fn`/`batch_decode_ctx`, and stale three-value expectation for a
  four-value return. Base-only run: 15 passed, 15 failures counting subtests.
  These unrelated tests were not edited. This is not a claim that all repo
  tests pass.
- Self-review and executable tests completed; no independent reviewer or GPU
  validation was available in this session. No paper-performance conclusion
  is promoted from this result. The PR is for code review and GPU follow-up.

## Remaining validation

Run the CUDA BF16 integration test and benchmark on the deployment GPU. Measure
whole-model latency, active/peak GPU memory and gate coverage; run paired
quality evaluation because strict far boundary/tie semantics differ from the
historical gate. Keep defaults on legacy until that evidence is available.
