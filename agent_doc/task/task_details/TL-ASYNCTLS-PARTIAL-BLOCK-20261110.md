# TL-ASYNCTLS-PARTIAL-BLOCK-20261110

Status: CPU-verified draft fix; GPU, corpus quality and performance pending.
Source: authorized recurring two-repository correctness audit. Tested baseline:
`0edefe244f8aa64ae8244de9eff8a803e2013a8b` (main), re-fetched before publication.

## Confirmed defect and narrow repair

Both `exp/vendor_baselines/asynctls_vendor/{qwen3,llama3}_asynctls_patch.py`
append zero keys to the final incomplete block before min/max scoring. Zero is
not neutral for extrema. For query +1, block size 8, eight real keys -1 followed
by a final real key -2, the genuine block scores are [-1,-2]. Zero padding changes
them to [-1,0], so top_kb=1 chooses the worse partial block. Negating Q and K gives
the corresponding negative-query failure. The error is upstream of candidate
clipping: excluding padded token IDs later cannot restore the displaced block.

Use an expanded view of the last real key for score-only padding. Repeating a
member does not change that block's extrema. Preserve real-length candidate
clipping, complete-block behavior, INT4 arithmetic, budgets, aggregation,
full-dimensional attention and persistent cache contents. The decode path has
one new token, so a real tail key exists. No new GPU kernel, cache policy or
claim of performance improvement.

## Method/source boundary

Read AsyncTLS arXiv **2604.07815v1**, 9 April 2026, section 4.1:
https://arxiv.org/html/2604.07815v1#S4.SS1
Its block representation is max/min over actual members of a block. The paper
writes floor(n/B), while these adapters intentionally retain a partial final
block. This patch fixes that existing partial-block adaptation; it does not
claim an official implementation of the complete method. The fetched HTML has
SHA256 `e012adccf334855e214ed106d90056dec38b6abd1715caf5c8723c42062ba6c6`.

The repository explicitly labels these adapters a semantic reproduction, not
vendored official source. The paper's group-specific block/token selection and
these adapters' extra aggregation across KV heads are distinct contracts. This
patch deliberately preserves the latter and must not be presented as repairing
all fidelity differences. The paper's own GQA group-mean equation is explicit;
the inaccurate adapter comment claiming GQA is absent from the paper is corrected.
A current GitHub repository-name search returned unrelated TLS networking
projects; the pinned paper HTML contains no author code link. No official
implementation version was established, so no official-equivalence claim is made.
INT4 unpacked storage, calibration data and absent asynchronous offloading also
remain reproduction boundaries, not solved by this patch.

## CPU reproduction and acceptance

Temporary test environment: Python 3.12, Torch **2.8.0+cpu**, Transformers
**4.56.2** (not the repository's Python 3.11 GPU environment). No model downloads,
CUDA execution or GPU reservation. Tests use real Torch tensors.

- `test_asynctls_partial_blocks.py`: three unittest methods / 128 subcases,
  source-extracted actual Stage-1 statements versus an independently constructed
  unpadded-block oracle. Both adapters; FP32/FP16/BF16; B=8/64; lengths 1, B-1,
  B, B+1, 2B-1; MHA/GQA; signed query counterexamples and valid-zero/valid-tail
  controls. Baseline has **55 failing subcases**, candidate passes all 128 in
  ordinary and `-O` runs.
- `test_asynctls_partial_block_forward.py`: eight full native attention-forward
  cases with deterministic projections and an append-only tensor cache. Baseline
  has **4 failures / 4 passes**, candidate passes all eight; cache remains length
  9, never 16. Two additional randomly initialized tiny Qwen3/Llama model tests
  compare full-budget logits with dense through three decode steps crossing a
  block boundary. These are correctness tests, not trained-model quality claims.
- New tests plus `test_llama_vendor_adapters.py` and
  `test_baseline_integration.py`: **33 passed + 128 subtests passed** in ordinary
  Python and `-O`. Pytest warns about optimized-out assertions outside test
  modules under `-O`; Torch/unittest comparisons execute normally.
- `py_compile` and `git diff --check` pass. Independent read-only review of the
  patch confirmed the extrema and candidate/cache invariants.
- Required E127 gate was attempted and stops because Torch is CPU-only.
  README names a nonexistent `test_nfa_batch_gates.py`; the existing
  `test_tli_batch_gates.py` was attempted and stops importing deleted
  `sparse_attn.indexer.tli_indexer`. Neither is reported as passing. No unrelated
  gate-script repair is bundled here.

Reproduce candidate:
```bash
python tests/test_asynctls_partial_blocks.py
python -O tests/test_asynctls_partial_blocks.py
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q \
  tests/test_asynctls_partial_blocks.py tests/test_asynctls_partial_block_forward.py \
  tests/test_llama_vendor_adapters.py tests/test_baseline_integration.py
# Repeat the command above with python -O.
```
For negative controls, export the two baseline adapters into a temporary source
root preserving `exp/vendor_baselines/asynctls_vendor/`, then set
`ASYNCTLS_SOURCE_ROOT=/absolute/baseline-root` for the new test files. Never
replace a working checkout's source to run the negative controls.

## Historical evidence and pending acceptance

Only AsyncTLS adapters executing an incomplete final score block are in scope.
The recent Llama vendor archive cites these adapters; this is a reason to recheck
that AsyncTLS arm, not evidence of how often selection changed or the direction
or magnitude of its reported quality. Do not invalidate or modify FullKV, NFA,
Quest, JSSA or historical files. No predictions were deleted or regenerated.

Before accepting corpus numbers for the repaired adapter, compare selections
and predictions on identical inputs, pin both adapter SHA256s, calibration file,
model, budget and harness source, and independently re-score. Actual CUDA dtype
parity and timing remain pending. No GPU matrix is authorized by this audit.

Open PRs were enumerated for overlap; none addressed this AsyncTLS pad rule.
Work used an isolated sparse checkout, with no other writer and no main edits.
The task index remains the existing `agent_doc/task/TASK.md`; no guide changes.
