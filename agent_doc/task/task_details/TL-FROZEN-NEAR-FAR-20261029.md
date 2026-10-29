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

## Completion — 2026-10-29

- Current staging-policy run completed100/100 paired requests and500/500
  generated results, exactly20 per screening dataset. Workers completed at
  12:07:44/12:11:10 Asia/Shanghai; hourly supervisor's log records automatic
  completion at12:41. Supervisor is now closed.
- Generation source commit62f6fb4; identity
  54fb76f1ea02b2b007e1a2e82b7c05d467be0af78cf8e03eda5e7c9095212b48.
- Full AVG5 53.8789; far0.10 54.0283; dual0.10 54.0548; far0.15 54.3552;
  dual0.15 53.3765. Dual0.10−full +.1759 pp, 95% paired bootstrap
  [−.2380,+.6534]; dual0.15−full −.5024 pp, CI [−2.4623,+1.4752].
- Dual0.10 has near activated23/100 requests,92 near-only layer cases and4
  both-skip cases. Dual0.15 has near activated93/100,378 near-only and133 both.
  Near-only resident-node KV ratio is exactly1.0; no near KV saving claimed.
- All500 scores re-evaluated with repository difflib scorer; all18000
  request-layer cache records checked for geometry, token conservation, linked
  visibility, full Far migration and source/input identities.470 near-only
  layer cases,90 with observed Far growth; two native FullKV generation
  checks; all arm prefill profiles identical.14 regression tests pass.
- Multi-hop scope: Musique and HotpotQA are included. At0.10, all20 per-task
  scores match FullKV (near active9/20 and4/20 respectively); this is not full
  200-row coverage. At0.15, Musique +2.5 pp comes from one improvement;
  HotpotQA has one degradation (−1.0 pp task average). Qasper and GovReport
  also decline under dual0.15. AVG5 must not mask those regressions.
- Final report, TSVs, compact predictions, raw trace archive, validation and
  provenance live in agent_doc/results/frozen_near_far_staging_20261029/.
  Local executable verification is complete; no independent third-party review
  has been claimed. No production defaults or TLI L1/L2 integration changed.

## TLI integration — in progress, 2026-10-29

User approved continuing on this branch with operator mode off. The earlier
completed scores are native SDPA region ablations and do not answer the TLI
quality question. Implemented real register_patch + frozen treatment snapshot,
pre-update segmented-cache dispatch, compact logical tail32 scratch, existing
TLI compute_score/compute_mask reuse, visible-position mapping and decode
receipts. No core TLI scorer or selector changes. Fixed K1=128/K2=1024,
nb32/ntoff/nl4096, sink4/SWA128, minmax/avg, tail32; default no L2 quantization.
TLI_FINE_GATHER=0 is explicitly recorded: correctness reference, no latency or
peak-memory claim. Near staging is not read until it migrates into Far.

Validation so far: 25 CPU regressions, including patched Qwen logits in all four
states against a logical full-cache reference. GPU BF16 32Q/8KV masks match
original TLI at 4231/8193/24001/31520 tokens with zero differences. HotpotQA
8192 smoke completed5 arms, native TLI output parity passed, Far-only and both
activated. GovReport 8192/512-token smoke pending. LongBench100-request paired
TLI screen and hourly supervision will follow smoke acceptance. Full score,
multi-hop regression and any no-loss conclusion remain pending.

TLI screen launched 2026-10-29 14:13 Asia/Shanghai, generation commit27d4464;
identity b9597b507f0baa9a9d526d4ba9086dcf17997c0b820db89a87a983968e22e5c0.
Output agent_doc/results/frozen_near_far_tli_20261029/. Supervisor PID84807,
workers85029/85030 on GPU0/1, hourly checks, next15:13. Source hashes bind72
files; no generation-source changes permitted during the run. Both workers
passed native TLI greedy-output parity. First5 complete requests/25 arms/900
layer states mechanically validated, including6 Near-only states. At14:15,
6/100 complete; all still Qasper, so no AVG5 or multi-hop conclusion yet.
Automatic completion performs validation and compact export. This launch is
not completion of the TLI quality task.


## User correction — independent gates, 2026-10-29

Near and Far each compare only their own mass; delete the joint sum budget and
lower-mass winner rule. Both passing means both skip. Mode/presence/strict
threshold comparison remain unchanged. The old joint-budget TLI run is stopped
and marked superseded, with raw checkpoints preserved. No new threshold or GPU
experiment is selected in this code-only correction. New protocol schema:
frozen-near-far-screen-v3-independent-gates. Validator retains explicit v2
historical rule handling; v3 validates independent decisions. Reports identify
the policy by schema, so v2 output cannot be described as independent gates.

Validation:31 local regressions passed, including both individually qualifying
with combined mass above epsilon, all four states, exact threshold boundaries,
mode/presence guards, independence from the other region, and existing TLI/cache
logit/migration checks. No GPU experiment used for this code correction.
