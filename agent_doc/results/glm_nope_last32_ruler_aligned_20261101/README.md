# GLM nope_last32 full-task run, aligned to latest origin/main

- Run id: `glm_nope_last32_ruler_aligned_20261101`
- Host: `33.253.189.125`, GPUs 0–5 and 7; GPU6 reserved by owner lease.
- Source commit: `42821f81a4739d05d8bc81e1d4cc58307e34bb7e`, clean.
- Selector: `nope_last32` = post-RoPE K dimensions `range(160,192)`.
- Arms: `dense` vs `selected`.
- Tasks: RULER `niah_multikey_2` 100 + LongBench AVG5 1300.
- Protocol schema: `glm-nope-last32-full-aligned-v1`.
- Remote raw output: `/tmp/tli-glm-nope-last32-aligned-full-run`.
- Scripts: `scripts/glm_nope_last32_full_aligned.py`, `scripts/run_glm_nope_last32_aligned_125.sh`, `scripts/analyze_glm_nope_last32_full_aligned.py`, `scripts/verify_glm_nope_last32_full_aligned.py`.
- Scope: raw paired full-task diagnostic; not independent benchmark/paper acceptance.

## Why this run exists

The old full-run RULER producer only used `generation_config.eos_token_id` and omitted the official RULER newline EOS, so dense/selected generations could run to 64 tokens and repeat. Latest `origin/main` already contains that newline EOS and moved GLM TLI to `PagedRegionCache`, which must go through `prefill_frozen_request`. This run uses a clean `origin/main` checkout plus the aligned runner to rule out the EOS/cache-alignment bug.

## Result

All 1400 requests complete, 2800 arms complete, zero failures. Independent verification: `raw/validation/independent_review.json`. Local recomputation: `summary_recompute.json`.

| Task | Dense | nope_last32 | Delta |
|---|---:|---:|---:|
| RULER MultiKey2 64K | 90.0 | 28.0 | -62.0 |
| Qasper | 38.01 | 37.01 | -1.00 |
| HotpotQA | 62.50 | 61.53 | -0.97 |
| Musique | 35.36 | 34.45 | -0.91 |
| GovReport | 33.42 | 32.68 | -0.74 |
| RepoBench-P | 63.44 | 60.65 | -2.78 |
| LongBench AVG5 | 46.55 | 45.27 | -1.28 |

Paired bootstrap 95% CI for AVG5 delta: [-1.96, -0.64]. RULER MultiKey2 delta CI: [-72.0, -52.0]. The EOS/cache-alignment fix restored dense output cleanliness, but `nope_last32` still drops raw RULER accuracy substantially.
