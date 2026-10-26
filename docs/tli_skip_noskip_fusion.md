# skip / non-skip fusion and physical residency audit

2026-10-26; parent PR #7 `80e151e`. Kimi's advice was supplied directly by
the user in the conversation, not a remote advice file. This note records its
disposition and the implementation/validation added in response.

## Fusion coverage

The optional GPU pipeline now uses `_pool_score`: region validity, projected
L1 dot products, per-query-head dtype rounding, GQA mean and the score mask are
one kernel. Far and near each get their method specialization. Whole inactive
tiles branch around Q/K loads and dot products; partial tiles mask K loads.
Full-P outputs preserve the previous top-k domain. Empty quotas allocate no
score tensor, launch no scoring kernel and call no top-k.

Thus **skip does not run far scoring/top-k**, and near scoring no longer reads
or computes all far-page features first. **Non-skip gets the same regional
fusion for both pools**, as well as the previously added cache maintenance,
Q projection, candidate construction and final-mask kernels. This still writes
full-P masked scores and uses Torch top-k and fine softmax: it is not a claim
that every O(L) operation or allocation has disappeared.

| Request/storage | skip | Execution route with explicit switches |
|---|---|---|
| Fixed full KV, B=1 or B>1 | off | New pipeline, far+near regional scoring, eager/Graph |
| Fixed full KV, B=1 or B>1 | on | New pipeline, near-only regional scoring, eager/Graph; legacy boundary bypass retained |
| Compact ring, B=1, per-prompt frozen gate | on | Existing compact eager/Graph selector plus `TLI_FAR_SKIP_KERNELS=triton` writes/gathers; does not treat ring slots as chronological full KV |
| Compact mode configured, B=1, frozen gate chooses full KV | off | Now admitted to the new full-KV pipeline; permits different layers in the same prompt to use ring/full storage |

`TLI_BATCH_INDEXER=parallel` now also admits B=1 **in the fixed-capacity BatchCtx
entry**. B=1 dynamic gates are accepted only after freezing; the Qwen staged
entry freezes before first use, and capture refuses an unfrozen gate. Changing
the skip decision after state initialization requires clear/recapture. B>1
per-row dynamic gate dispatch and B>1 physical eviction remain unsupported.
Ordinary DynamicCache decode without the fixed-capacity BatchCtx does not
automatically acquire this full-KV pipeline. The new pipeline remains opt-in; it does not change the default legacy selector.

For a mixed compact/full-KV B=1 model run use both explicit backend families:

```bash
export TLI_BATCH_INDEXER=parallel TLI_BATCH_SCORE=triton
export TLI_BATCH_FUSIONS=1 TLI_BATCH_GPU_PIPELINE=1
export TLI_FAR_SKIP_KERNELS=triton
# Model flags: --tli_far_skip_backend compact --tli_far_skip_cache evict
# Enable the existing per-prompt dynamic gate, dense prefill, then use
# fill_static_from_dynamic(..., model=model) outside capture and release dyn/out.
```

This is opt-in and GPU acceptance is pending. L1 GQA accumulation now takes a
different FP32 reduction order; output masks/attention still face the exact
remote gate. A difference is a failed gate, not an automatically accepted
approximation. Initial wide tiles caused up to 255 registers and two spilling
specializations; narrower full-width tiles and a non-unrolled head loop were
retained for GPU validation. Both rejected resource trials remain archived.

## Kimi advice: adopt, adapt, reject

| Statement | Disposition | Verified boundary |
|---|---|---|
| Masking far is insufficient for physical memory savings | **adopt** | Default main path and the full-KV batch path retain full storage; analytic 13% is not an allocator measurement. |
| Freeze each prompt's gate, compact once before decode, then append in place | **adopt** | Already implemented by PR `prepare_far_skip_static` / `FarSkipLayer.from_layer`; tests now explicitly exercise changing pending decisions across prompts and old-storage release. |
| Current GPT ring copies O(L) historical KV every decode step | **reject for PR 80e151e** | `FarSkipLayer.update` and `FarSkipStaticLayer.update` write one K/V token into a cyclic slot. The new dispatch test checks constant destination elements over two complete wraps at two prompt lengths. Selection/gather costs are separate and remain. |
| Exact required capacity is 4228 | **adapt** | Current page-aligned strict selector needs at most `sink + max(near_len,bs) + swa + bs - 1`: 4+4096+128+63 = **4291**. Removing alignment padding can overwrite still-readable tokens. |
| Frozen skip means legacy selector never reads any far token | **reject as a general claim** | Legacy sink-boundary bypass can keep non-sink far tokens. PR strict compact deliberately removes them, so its quality cannot inherit historical gate results. A fully legacy-compatible ring would need to preserve additional boundary tokens and its exact scoring/ordering contract. |
| One-time setup is O(L), tens of ms and invisible after amortization | **adapt** | Skipped-layer conversion copies only retained sink/tail in at most two tail runs, O(C), not all L tokens. Full-KV layer transfer can be O(L). Calibration and allocator work also exist. Setup must be measured and included in complete-request cost; no timing is asserted here. |
| Ring update has zero hot-path cost | **adapt** | It removes historical-KV relocation. New-token stores, slot addressing, launches and selector gathers still cost time. Compare against an equally preallocated append baseline, not an unrelated concatenating cache. |
| 5/36 implies 13% KV saved and quality is closed | **adapt/reject transfer** | For equal-sized layers and L=131072, 5/36 × (1−4291/L) is about 13.43% of KV bytes only. Total GPU allocation also includes weights, workspaces and graph pools. Gate counts are prompt-dependent; strict-eviction quality and the actual allocator delta are pending. |

No historical GPU timing or quality figure is promoted by this audit.

Related remote evidence read:

- [Main E170 report](https://github.com/Chosen-David/two-level-attention/blob/9058345e6a32aa4dac9cd07a465ae6aad5375777/agent_doc/advice/e170_subspace_farskip_verification/2026-10-25_e170_innovation_verification_by_claude.md): §3 explicitly uses an analytic byte ledger without allocation. Its “memory realized” wording cannot establish physical savings.
- [Older compatibility review](https://github.com/Chosen-David/two-level-attention/blob/ee44158abcf10a2f6e154b29456664e5ba96912d/agent_doc/advice/e175_gpt_far_skip_eviction/2026-10-25_compatibility_review.md): confirms one-time conversion and the strict/legacy semantic difference. Its eager-only/Graph rejection describes an older implementation, superseded by the current PR's static adapter.
- [Older eviction review](https://github.com/Chosen-David/two-level-attention/blob/ee44158abcf10a2f6e154b29456664e5ba96912d/agent_doc/advice/e175_gpt_far_skip_eviction/2026-10-25_far_skip_eviction_review.md): retains a negative single-layer timing signal and acknowledges incomparable gate configurations and missing quality gates. Do not inherit its speed conclusion.
- [Older footprint receipt](https://github.com/Chosen-David/two-level-attention/blob/ee44158abcf10a2f6e154b29456664e5ba96912d/exp/trace/results/e175_gpt_far_skip_eviction/e175_m1_footprint_verdict.json): the `savings` field mixes within-arm prefill→decode deltas with between-arm deltas. At 128K, the recorded between-arm allocation difference is 72.850−65.681=7.169 GiB; 4.703 GiB is the evict arm's own prefill→decode drop. Neither establishes current matched quality/performance. Its stated capacity arithmetic also does not match 4/4096/128/64.

## Remote verification

The existing runner now defaults to both `--skip-modes off on` in eager/Graph.
It retains gate state in every paired receipt and tests mixed ring/full model
routing, with the CUDA test skipped explicitly on CPU-only hosts.

```bash
python -m exp.bench.run_tli_gpu_pipeline \
  --outdir /tmp/tli-skip-both --batches 1 4 --lengths 32768 \
  --skip-modes off on --steps 128 --repeats 6 --profile
```

Physical residency has a separate probe:

```bash
TLI_FAR_SKIP_KERNELS=triton python -m exp.bench.far_skip_residency \
  --device cuda --length 131072 --layers 36 --skip-layers 7 8 10 11 35 \
  --steps 256 --out /tmp/tli-residency.json
```

The five layer IDs above are **synthetic supplied gate outcomes**, not a fixed
production policy and not fresh calibration. The probe calls the actual freeze
and static conversion code, verifies old tensor references are gone and that
ring pointers stay fixed, and records actual storage bytes plus CUDA allocated,
reserved and peak memory before/after releasing source tensors. It does not
call `empty_cache`. Reserved allocator memory may remain high after live
allocation drops. The append timer covers skipped cache writes only, not model
TPOT. Real per-prompt gate calibration and matched full-model quality remain
mandatory for any paper claim.

Local CPU storage smoke (two layers, L=8192, one skipped): actual tensor storage
134,217,728→102,301,696 bytes; full layer capacity 8197, ring capacity 4291; all
old prefill tensor weak references released. This is **CPU storage evidence**,
not measured GPU memory or 13% model savings. Final test/compile counts and
hash bindings are in `agent_doc/results/tli_skip_noskip_fusion_20261026/validation.json`.

## Integration with main 262517b

This revision merges current main, retaining its manuscript/data updates and
CLI defaults. The clean text merge alone was insufficient: the upstream
constructor cleanup removed `_inc`, `_triton_ok` and far-cluster initialization;
these are restored. The new full-stat call also passed original subspace indices
to an already-gathered tensor. It now passes local coordinates (`None`) and has
a regression with known channel values. This removes the double-index / potential
out-of-bounds access without claiming a measured speed gain for that kernel.
Three old AST-only test harnesses were updated for the current graph-context,
metadata-signature and four-value candidate interfaces; production assertions
were retained. The local checkout contains a selected test subset, not every
repository test. After integration: 254 tests and 217 subtests passed, 182 skipped;
the merged-source Triton interpreter subset passed 99 with 38 skipped. The
unchanged pipeline still has 284 source-bound offline SM80/SM90 builds, including
96 regional-score builds at 31–64 registers and zero spill cases. Neither this
nor the earlier 123-pass interpreter run is a CUDA execution result.
