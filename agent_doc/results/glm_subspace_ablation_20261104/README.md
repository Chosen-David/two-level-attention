# GLM-4.7-Flash 32-dim subspace ablation (2026-11-04)

Status: partial evidence; new-candidate screen runs pending (125 unreachable as of 13:07 UTC).
Finalist full run remains `glm_nope_last32` from prior run.

## Scope
- Tasks: LongBench v1 `hotpotqa`, `gov_report`; RULER 64K `niah_multikey_2`
- Selected arm: NFA gate `full`, selector subspace only
- Dense arm: native HF SDPA/DynamicCache
- Source worktree: `~/tla-glm-subspace-main-20261104` on `33.253.189.125`, `GIT_HEAD=e28f5af`

## Already validated full result
`glm_nope_last32` full run (`agent_doc/results/glm_nope_last32_full_20261101/`):

| task | dense | nope_last32 | delta |
|---|---:|---:|---:|
| hotpotqa | 62.50 | 60.84 | -1.66 |
| gov_report | 33.42 | 32.65 | -0.77 |
| niah_multikey_2 64K | 90.0 | 28.0 | -62.0 |

Conclusion: LongBench is close to dense, but RULER 64K MultiKey2 is far outside noise.

## Existing screen evidence (first 20 rows/task, `glm_subspace_scan_20261101`)
Screen selected/dense deltas for the three tasks:

| selector | hotpotqa delta | gov_report delta | multikey2 delta |
|---|---:|---:|---:|
| nope_first32 | +1.18 | -0.01 | -75.00 |
| nope_last32 | +0.18 | -0.53 | -50.00 |
| nope_first64 | +1.00 | -0.17 | -62.50 |
| nope_last64 | +0.13 | -0.56 | -41.67 |
| rope_low32 | +0.73 | -1.30 | -91.67 |
| rope_high32 | -12.65 | -10.01 | -91.67 |
| mixed32-16n16r | +1.18 | -1.21 | -83.33 |
| full256 (full) | -0.06 | -0.14 | -56.00 |

Screen ranking favors `nope_last32` among 32-dim NoPE-only candidates on RULER; LongBench differences are small.

## New candidate screen runs
Launched on 125 GPUs 0/1/2/3/6 with `--samples 20` for:
- `glm_nope_uniform32`
- `glm_nope_random32` seed 0 and seed 1
- `glm_rope_uniform32`
- `glm_rope_random32` seed 0

Remote output dirs: `/tmp/tla_glm_*_screen`.
Pending network recovery and verification; do not treat as final results yet.

## 2026-11-05 update: resource check + RULER test correction

- Current hosts: 125/113 still `No route to host`; all reachable H20 docker hosts have
  existing jobs and no full free card. A GLM LongBench paired screen needs ~68-70 GiB
  and OOMs when sharing a card with an existing ~70 GiB process.
- Auto scheduler started locally: `/tmp/glm_subspace_scheduler.py`, log
  `/tmp/glm_subspace_scheduler.log`. It waits for a GPU with enough free memory,
  then launches pending LongBench screens and RULER diagnostics. It never kills
  existing processes.
- Important correction: previous RULER `nope_last32` runs used `pred_ruler.py`
  without explicitly setting `--nfa_frozen_region_mode full`; for GLM the default is
  `dual`, so near/far skip was active and the result is **not** a subspace-only
  comparison. The LongBench path already used gate mode `full`.

## How RULER is tested (subspace-only)

Use `--nfa_frozen_region_mode full` on every NFA RULER arm. This keeps all
near/far physical KV and isolates only the selector subspace.

1. Native fullKV reference: `--method none` (native HF SDPA/DynamicCache).
2. NFA full-index sanity: `--method nfa --nfa-subspace full --nfa_frozen_region_mode full`.
   If this differs from native fullKV, the issue is NFA integration/selector, not
   the 32-dim subspace.
3. Candidate subspaces: `--method nfa --nfa-subspace glm_* --nfa-subspace-seed N
   --nfa_frozen_region_mode full`.

Task: `niah_multikey_2`, `--context_length 65536`, `--max-num 100` for full.
