# Prefill-frozen near/far permanent KV eviction exploration

## Plan

User-authorized branch: `explore/prefill-frozen-near-far-skip`, isolated worktree
`/home/wangyuanshuo02/two-level-attention-near-far-skip`, base c4dfb08.

Implement sink -> far -> near -> SWA ring, independent node tensors. Freeze
request-local near/far profiling after dense prefill; physically delete selected
nodes, preserve logical absolute RoPE positions. User explicitly confirms
permanent eviction. Near-skip implies surviving prefill far only: deleted near
and future SWA overflow never reappear after aging. No shadow full cache.

Screen the existing five tasks only: qasper/hotpotqa/musique/gov_report/repobench-p.
First exploration: seeded random20 per task, 100 paired requests, full/far0.10/
dual0.10/far0.15/dual0.15, 500 generation results, context cap31500, repository
prompts/scorers/max generation lengths, Qwen3 non-thinking template except code.
This is a subset exploration, not a completed full200/500-row screening.

Gate: last256 causal queries with query-relative page geometry, head-mean mass,
mean over query rows; independent region signals with joint dropped-mass budget.
Both nodes may be removed if combined calibrated mass fits the same epsilon.
Report activation and per-task/macro score, paired stratified bootstrap5000,
physical node bytes and output identity. CI containing zero is not equivalence.
No TLI L1/L2 or performance claims in this initial method-isolation run.

Hourly local supervisor: exact worker PID/heartbeat/GPU/progress audit, resume
same source/input identity, at most3 retries, terminate only own process group
after2h stalled. No automatic parameter changes. Aggregate on completion.

## Progress

- 2026-10-29: branch/worktree created; main worktree and other experiments untouched.
- Physical four-state cache, chronological rings, migration/deletion lifecycle,
  query-relative profiling and paired runner implemented.
- 12 CPU tests passed, including tiny native Qwen3 logits against explicit
  full-cache gather for all4 states through ring wraps/page migrations, physical
  deletion ownership, and known-answer score/bootstrap aggregation.
- Real Qwen3-8B smoke: qasper/gov_report one request each, context8192, all5 arms.
  Full node-cache generation matched native DynamicCache on qasper. Real near,
  far and both-node deletion activated. Smoke is not an accuracy verdict.
- Formal exploratory run scheduled under `agent_doc/results/frozen_near_far_20261029/`;
  source/data/config identities and atomic per-request-arm checkpoints enforced.

Results remain preliminary/pending independent verification. No external
independent agent review has been represented as completed.
