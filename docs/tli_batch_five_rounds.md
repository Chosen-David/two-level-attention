# Five further batch-indexer iterations (2026-10-26)

Baseline: PR #7 `d31795618455b50889531092cd33c4f34c3a1171`.
Scope: the opt-in full-batch indexer, shared by staged eager and Graph. The
original B=1 compact eviction code is unchanged. Default route remains legacy.
No GPU is available locally; these are five implementation/local-validation
iterations, **not five verified GPU speedups**.

| Round | Waste found | Retained change | Evidence / limits |
|---|---|---|---|
| 1 | AVG-only or static far-skip still maintained/scored unused min/max | Allocate, update and score only needed summary types | Original Graph mask oracle and storage checks; AVG-only CPU cache saves 540,672 bytes at the measured shape. Mixed minmax/AVG needs both and does not receive this saving. |
| 2 | Identical coarse/fine Q projected twice, with full-width scale/cast first | One shared narrow Q projection; preserve input-dtype scale then FP32 cast | Exact FP32/FP16/BF16 elementwise checks, independent old Graph mask oracle |
| 3 | Generic stable sort used to partition valid/invalid candidate slots | Linear prefix-count partition; optional one-kernel Triton scan/copy | Both valid and invalid slots retain their exact order. Extra rank cache costs 33,792 bytes at the measured shape. Torch aten count increased slightly; GPU timing pending. |
| 4 | Final full-width mask repeatedly initialized/ORed/ANDed | Build forced base once, then selected-token scatter; optional two Triton kernels | Short rows, sink=0, oversized sink, invalid sentinel and poisoned-slot checks; no attention algorithm change |
| 5 | Page update reads full D before subspace selection; unnecessary cold fine-cache copy | Optional fused direct subspace page/new-token reads; reuse projected last-token features in Torch; clone only if fine cache aliases source K | Source-K ownership and incremental-state checks. Rejected a default Torch multi-axis-index trial: no measurable latency gain and 27 extra aten calls. Trial source and raw results preserved. |

All arithmetic reductions, budgets, candidate order and selection domains are
preserved relative to the previous batch candidate. Its pre-existing numerical
limits versus legacy eager still apply. Static far-skip here uses legacy
boundary bypass and full-capacity KV; strict compact eviction remains B=1.

## Switches and experiments

```bash
export TLI_BATCH_INDEXER=parallel
export TLI_BATCH_SCORE=triton
export TLI_BATCH_FUSIONS=1  # new, default 0; GPU acceptance pending
```

The new fusion flag controls stable candidate partition, final-mask assembly,
and update-feature gather together. It requires the explicit Triton backend;
invalid combinations fail. All first-time initialization/JIT must be warmed up
outside capture. State remains request-local; refill/reorder needs clear and
Graph recapture. Valid candidate IDs are unique within each KV head, so mask
scatter programs have no cross-program duplicate destinations.

Use the same final source on both arms to isolate the new fusion bundle:

```bash
python -m exp.bench.tli_batch_round_compare \
  --baseline-root sparse_attn/indexer --device cuda --dtype bfloat16 \
  --score triton --fusions --method mixed --batch 4 --length 32768 \
  --steps 128 --repeats 6 --out /tmp/batch-fusions.json
# Add --capture for actual Graph replay; omit --fusions for Torch/score-only.
```

The baseline defaults to fusions off; `--baseline-fusions` enables them in that
arm. Receipts record `fusion_active` for both states: check it is `[false,true]`
for this ablation. To compare the previous PR with the final candidate, use
`--baseline-root agent_doc/results/tli_batch_five_rounds_20261026/source_snapshots/round0`.

The harness records source/harness hashes, all samples, side-cache storage,
mask differences, attention output error, CPU wall time and (on CUDA) device
events. Cold preparation and profiler steps are excluded from steady timing.
KV append, metadata upload, Q generation and attention are outside timing;
these are indexer measurements, not model TPOT. Run B=2/4/8 and 32K/64K/128K
where memory permits, plus actual full-model eager/Graph and quality gates.

## Local evidence and unresolved work

Final CPU parent-to-candidate pair: B=4, L≈8192, Hq/Hkv=32/8, tail32,
mixed minmax/AVG, 16 steps × 4 alternating repeats, one CPU thread:
**34.526 → 33.525 ms** median (about 2.9% lower). One of four repeat medians
regressed, so this is not a robust/general speedup claim. All 256 row masks
matched and attention maximum absolute error was 0. Per-round smaller effects
and all individual samples remain in the receipts.

Final checks: **204 CPU tests + 205 subtests passed, 84 skipped**; **55 Triton
interpreter checks passed, 18 BF16 cases skipped**; **64 SM80/SM90 specializations
compiled**. These are distinct evidence levels, not CUDA runtime acceptance.


Raw evidence and every round's core source snapshot:
`agent_doc/results/tli_batch_five_rounds_20261026/`.
Rounds 1–4 used the preserved `round1_to_4_harness.py`; the rejected round-5
trial used `round5_trial_harness.py`. Later runs use the current benchmark.
`validation.json` binds each receipt to its actual source/harness hashes.

The CPU runs show small changes, not evidence of a substantial latency win.
Do not add their percentage changes: they used different source pairs, and
round 1 isolates AVG-only while rounds 2–5 use mixed minmax/AVG. Report the
final parent-to-candidate pair separately. CPU aten counts are not GPU launch
counts. Interpreter NumPy deprecation warnings are retained in the raw log;
they are not CUDA runtime results. Offline cubins likewise establish compilation
only. There has been no independent GPU review.

Remaining costs to profile on GPU:

- `topk`, fine softmax and GQA reductions, plus fixed candidate-slot work.
- Batch geometry is still recomputed per layer; cross-layer reuse must handle
  capture lifetime and changing lengths without stale device buffers.
- Full boolean masks and the attention consumer's selected-KV traffic remain.
- Full-capacity fine side cache remains; static far-skip does not remove all
  far summary scoring in this batch path. Batch physical KV eviction is pending.
- The simple Torch maintenance path still reads full-width page features;
  only the explicit fused gather narrows those device reads. Reading d32 rather
  than D128 reduces logical feature elements, not a measured 4× HBM speedup.
- End-to-end GPU speed, memory peak, generated-token parity, RULER, repository
  E127 and batch CUDA gates remain unverified in this CPU-only environment.

Further GPU producer/consumer fusion and the remote stress entry are documented in
[`tli_gpu_pipeline.md`](tli_gpu_pipeline.md); GPU validation remains pending.
