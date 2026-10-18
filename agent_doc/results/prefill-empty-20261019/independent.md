# Independent empty-row fallback review

Date: 2026-10-18. Baseline HEAD: `7eb42416288f4f2643442d8044b773e31e97e835`.

## Verdict

Accept the scoped reference-prefill empty-row fix. No blocking correctness issue found in the production diff or focused regression test. This is CPU/reference support-semantics acceptance, not GPU performance, production deployment, or full-model quality acceptance.

## Reasoning

The previous condition checked whether any row/head in an execution tile was empty, then added the diagonal to every row/head in that tile. Consequently, an unrelated empty row could enlarge an already nonempty support set. The new expression computes emptiness independently along the key dimension and gates the diagonal by that result. Nonempty masks remain unchanged; empty masks receive exactly their absolute-position key.

Broadcasting is appropriate for all three mask forms: dense `[R,1,1,T]`, KV-shared `[R,Hkv,1,T]`, and per-query `[R,Hkv,G,T]`. The last singleton dimension retained by `any(..., keepdim=True)` is required and present. For KV-shared selection, every query head in a KV group has the same support, so sharing its emptiness result is correct. Absolute positions include the execution-subtile offset `r0`. Protection bands are applied before checking emptiness, as required. The public equal-length, causal prefill contract ensures the diagonal key exists for every executed query row.

The removed `bool(...)` eliminates this fallback's tensor-to-Python scalar decision. No GPU timing or synchronization profiling was performed; do not turn that source-level observation into a measured speedup claim.

## Independently executed validation

Environment: PyTorch `2.8.0+cpu`, CUDA unavailable. Command:

```text
python .audit-empty-rows/independent_check.py
PASS: 1324 independent CPU cases; torch=2.8.0+cpu; CUDA=False
```

The independent oracle enumerates each row/head's visible key indices in Python and computes attention only over those indices; it does not reproduce the production mask-repair expression.

- 1,296 random-input cases: fp32/fp16/bf16; `(H,Hkv)` of `(1,1)`, `(4,1)`, `(4,2)`; KV-shared and per-query selections; all-empty, all-selected, mixed, and future-only selections; execution row tile sizes 1, 2, and effectively unsplit; no protection, sink-only, and window-only protection; contiguous and strided feature layouts.
- 24 public-wrapper cases: sequence lengths 0, 1, 2, 7; entirely sparse empty-selection and entirely dense modes; three dtypes. Includes short tail chunks.
- 4 exact fp32 support-mean cases: mixed row/per-query support under tile sizes 1, 2, 3, and effectively unsplit. Exact zero-tolerance comparison verifies execution-tiling invariance.

Random low-precision comparisons allow reduced-precision arithmetic tolerances because the independent multiply/sum oracle and einsum have different intermediate accumulation behavior. The dedicated zero-score support tests are exact fp32 checks, so support pollution cannot be hidden by those tolerances.

Also inspected the supplied pre-fix log: reference failed all three mask-head subcases (1, 2, 4); candidate controls did not fail. The new regression constructs a genuinely empty causal intersection followed by a nonempty row whose diagonal is absent from the selected set, and tests head-local effects as well as row-local effects.

The new focused regression was independently rerun with `python -m unittest discover -s sparse_attn/ops/tests -p test_prefill_empty_rows.py -v`: one test passed (all six runtime/mask-head subcases). `git diff --check` passed. No tracked production or test file was modified by this reviewer.

## Limitations and nonblocking follow-up

- No CUDA kernels, GPU memory/performance, real indexer integration, or model-quality runs were performed. The unconditional diagonal construction trades some tensor work for removing the prior host decision; net performance is unmeasured.
- The committed focused regression uses fp32 and one execution tile. Independent audit coverage above handles subtiles, all-empty/full selections, low precision, short inputs, and strided inputs, but adding a small permanent tile-invariance regression would protect that property in routine CI.
- This review does not extend the private helper's contract to negative query positions or missing diagonal keys; neither arises in the validated public equal-length prefill path.
