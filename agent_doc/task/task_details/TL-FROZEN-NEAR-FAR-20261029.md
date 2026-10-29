# Prefill-frozen near/far permanent KV eviction exploration

## Plan

User-authorized branch: `explore/prefill-frozen-near-far-skip`, isolated worktree
`/home/wangyuanshuo02/two-level-attention-near-far-skip`, base c4dfb08.

Implement sink -> far -> near -> SWA ring, independent node tensors. Freeze
request-local near/far profiling after dense prefill. Revised user semantics:
Near-only keeps an attention-invisible staging FIFO, including SWA overflow,
and migrates its KV into visible Far after aging. It saves attention work, not
KV memory. Far-only physically evicts Far; both-skip physically evicts both and
retains sink/SWA. Absolute RoPE positions continue increasing. No hidden full
backup exists; explicit staging is included in resident-byte accounting.

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
- Formal exploratory run scheduled under `agent_doc/results/frozen_near_far_staging_20261029/`;
  source/data/config identities and atomic per-request-arm checkpoints enforced.

Results remain preliminary/pending independent verification. No external
independent agent review has been represented as completed.

- Policy revision: user clarified Near-only is compute-only staging, not permanent
  deletion. Stopped old supervisor4045407/workers4045669,4045670, preserving63
  completed old-policy requests and STOPPED.json. New schema/output separates
  staging results; no old-policy reuse.
- Added explicit aging regression: a token invisible in Near staging becomes
  visible after migrating into Far, with unchanged resident bytes. Native tiny
  Qwen reference now uses complete dynamic Far; total correctness tests13.

- Staging-policy real Qwen3 smoke passed: all5 arms on qasper/gov_report;
  native full-cache output parity retained. In GovReport near-only layers
  L23/25/28/30, Far live tokens grew3132->3644 via Near staging, while each
  layer resident KV ratio stayed exactly1.0. Both-skip and far-only also activated.
