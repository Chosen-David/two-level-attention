# TL-GLM-SUBSPACE-32D-20261105

## Goal
- Find the best 32-dim GLM-4.7-Flash indexing subspace on the ablation set:
  LongBench v1 `hotpotqa,gov_report` and RULER 64K `niah_multikey_2`.
- Isolate selector-subspace quality: RULER NFA runs must use
  `--nfa_frozen_region_mode full` (no near/far skip).
- Diagnostics first: native fullKV `--method none` vs NFA full-index
  `--method nfa --nfa-subspace full --nfa_frozen_region_mode full`.

## Current status 2026-11-05
- 125/113 unreachable. H20 docker hosts partially occupied; no full free GPU.
- LongBench paired screen (`--samples 20`) on 8100 GPU0/GPU1 OOM because each
  GLM LongBench run needs ~68-70 GiB and the cards already held other jobs.
- Auto scheduler running: `/tmp/glm_subspace_scheduler.py`, log
  `/tmp/glm_subspace_scheduler.log`. It only launches into a GPU with enough free
  memory; it never kills existing processes.
- Pending queue: `ruler_diag_native_fullkv`, `ruler_diag_nfa_full256`, then
  LongBench screens for uniform/random/rope candidates.

## Important correction
Previous `glm_nope_last32` RULER result likely used the default
`--nfa_frozen_region_mode dual`; therefore the RULER -62 delta is confounded by
near/far skip and is not a clean 32-dim subspace measurement. LongBench runs
already used gate mode `full`.
