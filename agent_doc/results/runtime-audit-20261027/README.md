# Five-round implementation audit — 2026-10-27

Base: `Chosen-David/two-level-attention@8f213e48926431ea2d56cf3e7bc1a667d1105c74`.
Scope: TLI semantics, Qwen3 batch/cache integration, LongBench and RULER
generation/publication. This is a code review plus CPU regression record, **not
a new quality/performance verdict or independent scientific acceptance**.

## Findings and fixes

| Round | Finding and trigger | Fix / regression evidence |
|---|---|---|
| 1 | **P1:** parallel batch selection cached the first Python lengths indefinitely. In graph mode its early dispatch bypassed the runner's live device lengths, baking warmup lengths into the selection inputs. | Refresh the eager length tensor in place; route graph mode to `valid_len_buf`. Real torch batch selector regression advances unequal rows and checks new-token inclusion/pad exclusion. CPU graph-state simulation is not CUDA replay validation. Also pass `layer_idx`, not the attention object, to receipt routing. |
| 2 | **P1:** legacy staged batch calls `indexer.clear()` per row, destroying prefill-calibrated dynamic gate/static-pair state; graph paths cannot reconstruct each request's prefill state. `compute_mask_graph` also silently accepted disabled boundary bypass although its implementation assumes bypass. | Reject legacy batch requests requiring local calibration before state reset; direct these callers to compact/evict `HybridBatchRunner` or single-request non-batch decode. Reject graph geometry with boundary bypass disabled so the existing caller can fall back. Hybrid request-local state is covered separately. |
| 3 | **P1:** RULER batch generation omitted `eager_decoding_request_guard`, permitting deferred consumer errors to escape publication checks. Its three prefill paths ignored the already-computed `keep_logits` flag; fixed-cache initialization also allocated an avoidable full batch-by-prompt temporary. | Guard generation before writing rows/receipts; propagate `logits_to_keep=1` through compact, fixed, and dynamic prefill; initialize fixed storage from a shape-only view. Injected guard errors produce no completed receipt; real tiny HF Qwen3 verifies one-position logits on all three routes. |
| 4 | **Protocol mismatch:** LongBench unconditionally split the question suffix and sent it tokenwise through sparse decode, inconsistent with a complete-prompt dense-prefill interpretation. Separate tokenization can change BPE boundaries and prefill gate calibration. Llama's hand-built chat template already contains BOS but tokenization added another BOS. | Default to explicit `--prompt-mode full`, keep historical `split` opt-in, isolate output directories, bind mode into run-cap identity, reject mixed protocol scoring, and disable extra BOS for the Llama template. Test actual producer call widths, BOS handling, identity binding/tampering, and aggregation rejection. Existing tables are not revalidated by this migration. |
| 5 | **P1 memory risk:** LongBench's manual loop materialized every prompt position's vocabulary logits although only the last is consumed. | Inspect the explicit model forward signature and request one logit position via `logits_to_keep` or `num_logits_to_keep` where supported. Real tiny Qwen3 and Llama tests reproduce full-width logits on the base and preserve three greedy token IDs after the fix. Models without an explicit supported parameter retain their original call contract. |

At 128K tokens and a 151,936-entry vocabulary, one full fp32 logits tensor
alone is about 74 GiB (bf16 about 37 GiB); this is a size calculation, not a
measured peak-memory result. All five rounds include review of the changed
paths and regression checks. No new obvious bug was found in the final targeted
CPU scope. This does not certify all optional modes, CUDA kernels or pretrained
task quality.

## Paper/code correspondence and supported scope

- The main GQA path uses paired low-frequency RoPE coordinates, not the final
  32 contiguous coordinates: for D=128 and ratio=4, the selection is
  `[48..63, 112..127]`. Both index stages use this scoring view; final attention
  consumes original-dimensional K/V. Far minmax, near avg, absolute near page
  allocation, free token competition and sink/window protection are present.
- Champion settings are K1=64, K2=1024, nb32, nt-off, nl4096, tail32, sink4,
  swa128 and explicit `--tli_enable_layer_skip false`. README previously used
  K1=128 and omitted the layer-skip override; the example now matches the
  current champion. Existing `4bit` branch names do **not** mean default INT4
  storage (`tli_l2_quant=False`; optional simulated quantization is separate).
- Qwen3, Llama and DeepSeek-V2 adapters exist with no-match rejection. Compact
  eviction/hybrid batching is Qwen3-specific and must not be advertised as
  available to all three families. Registration still rejects dynamic gate on
  Llama/MLA; this audit does not remove those support boundaries. MLA uses its
  own RoPE/nope layout; the GQA tail32 claim cannot be copied onto it.
- `compact/evict` physically bounds skipped-layer storage, but its strict
  far-off support excludes unprotected sink-boundary far tokens. It is a
  distinct selector from legacy masking, with separate tie behavior. The
  existing compact quality gate and large-model performance work cannot be
  replaced by tiny-model equivalence tests.
- `agent_doc/advice/dac_manuscript/source/paper_en.tex` is a manuscript advice
  artifact, not implementation authority. **Adapt:** compare its full-prompt
  prefill/selection semantics to the producer. **Reject as a current-code
  description:** line 164's claim that skipped-layer KV only remains resident
  and masked; compact storage is now implemented. **Defer:** revising paper
  conclusions/tables until matching quality and performance evidence exists.
  The manuscript's ordinary full-width L2 scan description also needs to be
  scoped against the candidate-compressed production path. No manuscript or
  guide assets are edited in this PR.

## Validation

The unmodified console logs named below are preserved in `logs.zip`;
`comparison.json` records their uncompressed SHA-256 hashes.

Environment: Python 3.12, torch 2.8.0+cpu, transformers 4.56.2, pytest 9.1.1,
OMP/MKL threads=1. Dependencies were installed outside the repository. Tiny HF
models use random weights, without model downloads or actual dataset runs.

- `targeted-final.log`: **73 passed, 7 skipped, 7 subtests passed** across batch
  live inputs, RULER producer/publication, LongBench generation/identity, and
  hybrid cache suites. The seven skipped cases require CUDA.
- `round5-before.log`: both new real-model prefill-logit tests fail on the
  unmodified base (23 positions instead of one); greedy IDs are unchanged.
- `baseline-regression.log`: unmodified base broad suite has **51 failed,
  470 passed, 10 skipped, 1628 subtests passed**. This run predates copying the
  new round-5 regression into the detached baseline checkout.
- `cpu-regression.log`: **49 failed, 488 passed, 10 skipped, 1628 subtests
  passed**. All 27 remaining failing test nodes (49 failures including subtests)
  also fail on the unmodified base; there are **no new failing nodes**. See
  `comparison.json` for the exact sets. The broad suite remains non-green.
- `e127-gates.log` and `batch-gates.log`: both required GPU gates were invoked,
  but stop because CUDA is unavailable. They are **blocked, not passed**.

Broad failures reproduced on the base include: seven E140 figure provenance
checks (current assets/manifest drift and a missing author-local renderer
dependency), E141 verifier tests requiring absent `/tmp/tla-hisa-official`
source, an observation renderer dependency, old async-selector tuple arity
assertions, an outdated batch error-message assertion, and SparQ lifecycle
fixtures missing the adapter's now-required explicit position. These are
pre-existing validation debt outside this TLI runtime patch; the broad suite
is not green. Two pre-existing LongBench capture fixture failures are fixed
by loading the actual helper functions and supporting the current input object.

Reproduce the targeted suite:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_batch_live_inputs.py tests/test_ruler_batch_prefill.py \
  tests/test_ruler_batch_treatment_snapshot.py \
  tests/test_longbench_prompt_protocol.py tests/test_longbench_prefill_logits.py \
  tests/test_longbench_request_capture.py tests/test_longbench_empty_handoff.py \
  tests/test_longbench_prompt_split.py tests/test_lbv1_input_uniformity.py \
  tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py
```

Before merge, run the required gates on the supported CUDA stack and the
existing whole-model replay tests (ensure the CUDA tests execute, not skip):

```bash
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
python -m pytest -q tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py
python -m pytest -q tests/test_batch_live_inputs.py
```

The parallel graph-length fix additionally needs real CUDA capture/replay
coverage with growing, ragged lengths; the CPU state test cannot substitute.
Rerun real-model RULER/LongBench with identical flags, input identities and
prompt modes before updating quality claims. Keep the PR draft until these
CUDA requirements are resolved; no merge or deployment is performed here.
