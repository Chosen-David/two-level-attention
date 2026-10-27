# Five decode-runtime optimization rounds

Base: `8f213e48926431ea2d56cf3e7bc1a667d1105c74` (main, checked again before submission).
Date: 2026-10-27, Asia/Shanghai. These are implementation and local correctness results,
**not GPU speedups or permission to reuse main's earlier GPU acceptance**.

| Round | Commit | Change | Validation at that round |
|---|---|---|---|
| 1 | `5c2713a` | Share geometry within a forward/capture scope; hybrid prefill computes only last-token logits | 32 passed, 8 skipped; real HF last logits and KV comparison |
| 2 | `97d7b9e` | Pass a sorted, deduplicated `TokenSelection` directly to attention; preserve the mask API for diagnostics/other callers | 40 passed, 9 skipped; independent boolean union and dense attention oracles |
| 3 | `9800324` | Score only near AVG pages while retaining the original full-width top-k domain | 84 passed, 8 skipped; 48 boundary combinations and bounded-work check |
| 4 | `6954007` | Persistent per-row indexes; one batched hybrid SDPA; opt-in ragged pointer-table attention | 49 passed, 8 skipped, 128 subtests; additional lifetime suite 23 passed, 6 skipped, 128 subtests |
| 5 | `ec89126` | Device-side greedy tokens/EOS/next inputs; incremental one-column causal mask update | 10 passed, 2 skipped; real two-layer Qwen3 vs former host loop, including all three KV layouts |

The table contains overlapping suites: do not sum its counts. Subsequent review pins
the parent KV/context identities until request reset, validates backend names before
KV mutation, removes the obsolete mask-to-topk helper, and adds multi-tile/empty-head
kernel tests. `source-sha256.json` identifies the final tested code independently of
the documentation commit.

## Final local evidence

- `cpu-regression.txt`: **229 passed, 28 skipped, 803 subtests passed** across
  selected existing contracts plus new tests; see exact file
  list in `cpu-tests.txt`. Skipped CUDA tests are not passes.
- `final-review-regression.txt`: 23 passed, 8 skipped after lifecycle hardening.
- `final-selection-regression.txt`: 18 passed, 5 skipped after removing a redundant
  rank-mask allocation from the direct-selection branch.
- `triton-interpreter.txt`: 14 passed, 1 skipped, 9 deselected. Executes the new
  ragged kernel and existing direct attention kernel in the CPU interpreter, with
  empty heads, multiple tiles, mixed ring/linear storage, and dead rows.
- `offline-compile.json`: 24 specializations compile for SM80/SM90, fp32/fp16/bf16,
  D=16/128, for both attention kernels. **Zero GPU executions.**
- `baseline-negative-control.txt`: the new shipped-prefill regression fails on
  frozen main because `logits_to_keep=1` was absent, as expected.
- `e127-gate.txt` / `batch-gate.txt`: both mandatory GPU commands were attempted
  and blocked by CPU-only torch / no GPU. They remain required before merge.

Environment: Python 3.12, torch 2.8.0+cpu, transformers 4.56.2, pytest 9.1.1;
Triton 3.4.0 installed separately for interpreter/offline compilation. The initial
existing-kernel interpreter attempt failed in Triton's scalar `__index__` conversion
with NumPy 2.4 (retained in `triton-interpreter-numpy24.txt`). Running in an isolated
NumPy 2.2.6 environment passes; no production kernel or oracle was weakened to mask it.

## Scope and remaining costs

- Default hybrid attention is batched SDPA over selected KV only. It allocates
  O(B * selected_tokens * Hkv * D) gather/stack buffers, never padded full-history KV.
  `TLI_HYBRID_ATTN=triton` opts into direct row storage reads without those KV gathers.
  This new backend is experimental until real-GPU correctness and performance pass.
- Compact selections remove the **final mask -> scan -> index** round trip in
  the ordinary graph route and hybrid linear rows. Existing intermediate geometry
  masks and oversized-quota full-domain fallback still exist. This is not a claim
  that the entire indexer is O(K) or that all full-width operations are gone.
- At 128K, block size 64 and near length 4096, the tested AVG dot-product input
  shrinks from 2048 pages to at most 66. Top-k still has its original page domain.
  This work-count reduction is not a measured latency speedup.
- Hybrid selection remains per row. Attention is batched; the heterogeneous
  ring/linear selectors have not been combined into one batch kernel.
- Cross-request CUDA graph reuse remains unsupported. Each new request batch can
  change gate decisions, pair dimensions, workspace shapes and KV addresses.
  Current graphs and ragged pointer tables are request-owned. A future reuse path
  needs fixed-address refill plus complete compatibility guards and GPU replay tests.
- Generation polls the whole-batch EOS flag once per 16 steps by default, then
  copies outputs at completion. It can perform up to 15 extra dead-row iterations;
  tests verify frozen KV and identical returned sequences. `poll_interval` is
  configurable on the runner. First-token EOS behavior preserves the old driver.
- No RULER quality score, production TPOT, HBM peak, or capture-time improvement
  is claimed. The two unresolved structural costs above are not labeled fixed.

## Reproduce and close the GPU gates

From the repository root in an environment with the stated dependencies:

```bash
OMP_NUM_THREADS=1 xargs -a agent_doc/results/runtime-five-rounds-20261027/cpu-tests.txt python -m pytest -q
PYTHONPATH=. TRITON_INTERPRET=1 python -m pytest -q tests/test_hybrid_attention_kernel.py tests/test_token_selection.py -k kernel
PYTHONPATH=. python exp/bench/check_runtime_kernels_offline.py
```

Interpreter reproduction uses NumPy 2.2.6. On an exclusively reserved target GPU,
run the following at the **PR head**, not just the base commit:

```bash
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
python -m pytest -q tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py tests/test_hybrid_generation.py tests/test_decode_geometry_scope.py tests/test_token_selection.py tests/test_regional_score.py tests/test_hybrid_attention_kernel.py
```

Then compare frozen base and PR head with identical model weights, inputs, tokens,
frozen gates/pairs, batch sizes 1/2/8/32, ragged lengths 32K/64K/128K, and fp16/bf16.
Use the canonical flags from `agent_doc/CURRENT_NARRATIVE.md`, explicitly including
K1=64, K2=1024, `--tli_subspace tail --tia_level2_cmp_ratio 4`, near4096/nb32,
sink4/swa128, and the same L2 quantization setting on both sides.
Report prefill, capture, steady replay CUDA-event latency, whole-request wall time,
and peak allocated/reserved memory separately. Benchmark SDPA and opt-in Triton
as separate treatments. Include short/long generations and mixed EOS, warm-up,
multiple paired runs, selected-set/logit checks, and RULER quality acceptance;
report regressions as well as wins. Do not substitute the E177 KV storage formula
for measured peak memory. Keep the new backend opt-in pending these results.

This branch is based on main and does not merge or overwrite open PR #10; overlapping
runtime changes need reconciliation if #10 lands first. The active advice note
`2026-10-21_t4_t6_status_sync_by_claude.md` was reviewed: defer historical T4/T6
status edits because this runtime PR neither reopens nor schedules those experiments.
