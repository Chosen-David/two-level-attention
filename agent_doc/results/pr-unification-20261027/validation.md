# Unified review — 2026-10-27

Main8f213e4, PR10 baseline06e59d4. Isolated local snapshot/git baseline:
`unified-review-20261027`; only selected changed paths are published on PR10.
CLAUDE, orchestrator, GUIDE (read only), TASK and CURRENT_NARRATIVE checked.
All12 open PR heads/reviews/inline discussions/issue comments were read; no
reviews/comments were present. PR7/10 file lists were paginated to213/277 paths.
Source heads and per-PR dispositions are in sources.json. No PR is closed or
merged and main is untouched. Other opt branches are not open PRs and are not
silently transplanted. PR11/12 are concurrent independent work, not approvals.

## Source mapping

| PR | Disposition |
|---|---|
| #1 | Production signature guard, complete-page reuse and new_empty growth already in main. Keep tests; do not replay old indexer. |
| #2 | Integrate FASA bool-mask exclusion and flat [B] valid-length extraction, plus tests. |
| #3 | Adapt missing complete-model/derived-r installation preflight, batch-context guard and per-head identity. Preserve main V-mean lifecycle and allowed cache=None rebuild. |
| #4 | Keep dated design advice as historical proposal; do not overwrite newer TG completion record or claim implementation. |
| #5 | Keep dated AsyncTLS comparison plan/advice; experimental protocol remains a proposal, not GPU evidence. |
| #6 | Integrate early host valid_length validation and compatible regression fixtures. |
| #7 | Already subsumed by #10 and its later fixes; no old code replay. |
| #8 | Defer manuscript/figures. Its quality/timing identity caveats and older full-history/far residency descriptions must be reconciled with current hybrid code and evidence, not blindly replace latest manuscript. No publication assets changed. |
| #9 | Hybrid implementation already cherry-picked into main; retain #10's subsequent fusion and memory fixes. |
| #10 | Target: preserve all work through06e59d4, including request geometry and subspace-before-pad. |
| #11 | Integrate producer guards, prefill logits, fixed-init view, explicit full/split LongBench protocols, BOS and output identity. Adapt overlapping batch fixes to existing #10 live-Graph routing and parallel B1 gate support. |
| #12 | Prefill-logits objective covered by #11. Defer remaining coupled runtime changes; see exact conflicts below. |

This is a single review entry point, NOT a claim that every source patch is
safe or included. Deferred #8/#12 remain open with explicit dependencies.

## Conflicts resolved

- #11 reused-length allocation differs from #10's allocate-on-change policy.
  Adopt in-place refresh, preserve live graph buffers and layer-index receipts.
  Update old allocation-identity test to the new in-place contract. Actual
  ragged selection tests verify latest lengths, not just object identity.
- Legacy staged row clear loses prompt gate/pair calibration. Reject that route;
  preserve #10 parallel B1 frozen-gate path and hybrid request-local state.
- #3 old tests rejected cache=None, but current main explicitly rebuilds safely
  without a cache. Preserve main semantics; new exact fresh-adapter comparison
  verifies no previous-cache mean reuse or old cache mutation. No numeric
  tolerance is relaxed. Missing preflight checks now run before module mutation.
- #6 fixture assumptions updated to current selector keyword/four-output API.
- Narrative additions are appended; historical task tables are not overwritten.

## Why #12 is not blindly merged

Its replacement of hybrid attention removes `_attend_linear` and routes through
per-row gather/masked_fill/pad/stack + one SDPA, bypassing #10 paired gather.
Its older KV-write context conflicts with `_write_kv_pair`. TokenSelection
canonicalizes indices into ascending order, whereas #10 consumes bool-mask
int32 topk(sorted=False): equal sets do not prove identical floating-point
reduction or quality. Device generate also changes ownership of lengths/alive,
mask updates and capture buffers. Joint integration needs explicit old/new
selection/logit/quality gates plus actual GPU replay/allocator/latency pairs.
The geometry scope module could be adapted independently, but must update #10's
cache-policy benchmark: scope priority bypasses the frozen old-policy arm.
Do not inherit its existing comparison as combined evidence. Retain this
concrete next step instead of overwriting working fusion or claiming closure.

## Tests and evidence limits

- integrated.txt:322 passed,84 CUDA skipped,157 subtests passed. Includes
  producer/LongBench protocol, guards, FASA, incremental metadata, hybrid model,
  ragged/EOS/ring wrap and prior write/gather/projection/mask/output regressions.
- sparq.txt:37 passed,22 subtests passed. A separate independent review ran
 25 SparQ-selected tests,22 subtests,12 deselected and examined conflicts.
- geometry.txt: final request-cache regression result is recorded verbatim;
  includes old/new policy model pairs, reset and owner-lifetime checks.
- Initial/pre-review logs retain discovered failures. Missing local snapshot
  dependencies were restored from exact main. Old-contract tests were adapted
  as described, not counted as unexplained green. Full repository suite was
  not run and is not claimed green.
- gpu-gates.txt: required E127 and batch gates attempted but unavailable on
  torch2.8.0+cpu. Existing upstream main H20 acceptance does NOT cover this head.
  Prior PR10 interpreter/offline logs remain historical scoped evidence; no
  GPU kernel body changed in this integration and no new GPU measurement exists.

Independent review covered overlapping guards, source mapping and four SparQ
files; not full-repository or independent scientific acceptance. An additional
SparQ zero-query/cache=None nonfinite output was reproduced on frozen baseline
by the reviewer; upstream edge case remains unresolved, not a new regression.

## Remote acceptance

LongBench now defaults to full-prompt prefill; split is explicit, with distinct
output directories and manifest/scorer identity. Old quality tables do not
transfer to the new protocol. Strict compact selector still needs its own
quality gate; historical dynamic-gate quality is insufficient.

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_batch_live_inputs.py tests/test_ruler_batch_prefill.py tests/test_ruler_batch_treatment_snapshot.py tests/test_longbench_prompt_protocol.py tests/test_longbench_prefill_logits.py tests/test_longbench_request_capture.py tests/test_lbv1_input_uniformity.py tests/test_indexer_valid_length.py tests/test_fasa_boolean_mask.py tests/test_tli_incremental_metadata.py tests/test_baseline_integration.py tests/test_sparq_install_preflight.py sparse_attn/ops/tests/test_sparq_request_lifecycle.py tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py tests/test_graph_geometry_request_cache.py tests/test_subspace_before_pad.py tests/test_tli_batch_parallel.py
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
python -m exp.bench.geometry_cache_compare --mode both --batches 2 8 10 32 --steps 40 --repeats 5 --out /tmp/geometry-paired.json
python -m exp.bench.subspace_pad_compare --dtype bfloat16 --lengths 32769 131073 --repeats 20 --out /tmp/subspace-paired.json
```

Run actual CUDA cases rather than accepting skips. Pair production eager/Graph
B1/2/8/32, ragged32K/64K/128K, fp16/bf16, per-request gate, mixed EOS, repeated
wrap and new-request recapture. Report setup/prefill/capture, replay TPOT, full
wall time, allocated/reserved peaks and matching quality separately. No speed,
GPU memory or quality improvement is asserted by this integration.
