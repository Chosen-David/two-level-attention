# GLM subspace scan 2026-11-01

Scope: find GLM-4.7-Flash K-dim subspace (256/192/128/64/32, NoPE/RoPE/mixed) that keeps RULER `niah_multikey_2` and LongBench AVG5 close to dense, then full-run finalists. Also records per-layer near/far skip decisions for cross-task consistency.

Source: clean detached worktree `/home/wangyuanshuo02/tli-glm-origin-main`, commit `42821f81a4739d05d8bc81e1d4cc58307e34bb7e`.

Reference already available:
- `agent_doc/results/glm_nope_last32_ruler_aligned_20261101/` full nope_last32 baseline.
- `agent_doc/results/glm_full256_ruler_aligned_20261101/` full256 full run (in progress at launch time).

Current screen protocol: `RULER_LIMIT=24`, `PER_TASK=20`, first-N rows, dense+selected paired, gate mode `full` for subspace isolation. Finalists will be full-run.

Configs and launch state: `configs.json`, `screen_launch_state.json`.
Scripts: `scripts/glm_subspace_runner.py`, `scripts/orchestrate_screen.py`, `scripts/analyze_subspace_scan.py`.

Status: running.
