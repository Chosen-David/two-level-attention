# GLM-4.7-Flash 32-dim subspace ablation 2026-11-04

- scope: LongBench v1 hotpotqa + gov_report, RULER 64K niah_multikey_2
- source: remote 125 worktree ~/tla-glm-subspace-main-20261104, GIT_HEAD=e28f5af
- command candidates: glm_nope_uniform32, glm_nope_random32(seed0), glm_rope_uniform32, glm_rope_random32(seed0), glm_nope_first32
- existing full baseline: glm_nope_last32 full run (agent_doc/results/glm_nope_last32_full_20261101)

## 2026-11-05 status
- 125/113 unreachable; H20 hosts partially occupied. No new candidate completed yet.
- Auto scheduler: /tmp/glm_subspace_scheduler.py (PID see /tmp/glm_subspace_scheduler.out).
- RULER all future subspace runs must pass --nfa_frozen_region_mode full.
- Pending first: ruler_diag_native_fullkv, ruler_diag_nfa_full256, then 5 LongBench screens.
