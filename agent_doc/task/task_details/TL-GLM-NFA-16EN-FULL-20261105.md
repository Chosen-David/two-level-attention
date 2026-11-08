# TL-GLM-NFA-16EN-FULL-20261105

## Trigger
User directive 2026-11-05: after the best 32-dim GLM-4.7-Flash indexing
subspace is chosen, run the full LongBench v1 English 16-task set under the
GLM NFA path, then push to remote.

## Dependency
Depends on TL-GLM-SUBSPACE-32D-20261105:
- Select one final 32-dim subspace from the ablation set
  (`hotpotqa,gov_report`, RULER 64K `niah_multikey_2`).
- RULER runs must use `--nfa_frozen_region_mode full` for a clean
  subspace-only comparison.

## Planned full-16 run
- Model: GLM-4.7-Flash (`glm4_moe_lite`).
- Entry: `benchmark/LongBench/prefill_frozen_skip.py`.
- Datasets (all LongBench v1 English tasks):
  `narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p`
- Mode: `--attention nfa`, `--samples 0`, `--thresholds ""`,
  `--native-full-arm`, `--context-cap 31500`, `--seed 42`.
- Subspace: final selected `--nfa-subspace ...` (placeholder until selected).
- Sharding across free GPUs with `--rank/--world-size` after resource
  admission; do not use one card unless full run fits.

## Push
After the full-16 run and independent verification pass, commit the result
summary/scripts into the current work branch and push to the configured remote
(verify target branch with current project guidance before force/ordinary push).
