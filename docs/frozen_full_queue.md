# Autonomous full LongBench queue

The three authorized remote hosts each keep a local daemon plus shell watchdog.
They share a durable, locked queue. An idle GPU claims one pending shard; no
local Codex session or SSH connection is needed after startup. The hosts, shared
filesystem, and model/data mounts must remain available.

Scope: Qwen3-8B, five full datasets (Qasper/HotpotQA/Musique/GovReport200 each,
RepoBench-p500),1300 paired requests,3900 generations. Arms are TLI gates-off,
Far-only0.05, and independent Near/Far0.05. Dense prefill profiles the final256
queries, gates freeze for decode. Operator mode remains off. Full means all
samples; context cap31500 and official generation caps remain part of protocol.

Runtime source is a fixed Git snapshot. `--samples 0` selects all rows.64 logical
shards partition the same seeded row sequence with modulo assignment, independent
of which host/GPU claims the job. A worker resumes atomic request-arm checkpoints.
Each host uses only the GPUs explicitly listed in the plan. The preceding
screen's corresponding rank must finish, then memory≤64MiB and utilization≤5%
are required before taking that card. Polling is every300seconds; audit records
are hourly and on launch. The predecessor and any other GPU workload are not
terminated to make room.

An atomic symlink to a fully written owner record protects shared queue claims (cross-host `flock` is unreliable on this mount); a host lock prevents duplicate daemons. Dead-owner recovery is restricted to the same host and serialized locally. A
published lease survives daemon restart, and an orphan launched before PID
publication is recovered by exact run/rank command matching. Up to4 attempts
per shard; a failed shard does not block remaining work. Only an exactly matched
owned process group with a stale heartbeat for2hours may be terminated. SSH
loss or assistant inactivity does not stop the host watchdog. Host failure is
visible in stale daemon status; the implementation does not steal potentially
live leases from an unreachable host.

The plan's `return_path` is the fixed entry under
`/home/wangyuanshuo02/two-level-attention/agent_doc/results/`. It points to shared
storage, so remote predictions/logs are immediately readable here; no final
copy depends on a local agent. All successful shards trigger one locked
finalizer: recompute official scores, verify input/source/gate/cache invariants,
then write REPORT.md, summary.json, validation.json, scores.tsv,
per_request_scores.tsv, predictions.jsonl and raw_index.json. FINISHED.json is
`production_complete_pending_independent_review`; automatic mechanical checks
are not an independent scientific acceptance or a no-loss conclusion.

Operational files: `queue.json`, `daemon_<host>.json`, per-attempt worker logs,
per-request atomic JSON, `worker_done_rank<N>.json`, and `NEEDS_ATTENTION.json`
if retries are exhausted. Creating `PAUSE` in the result directory prevents
new claims while running workers finish; removing it resumes dispatch.

Launch on each host using the plan's Python and source paths:

```sh
nohup bash SOURCE/scripts/run_remote_frozen_queue.sh PYTHON SOURCE PLAN HOST_KEY \
  > RESULT/daemon_HOST_KEY_console.log 2>&1 < /dev/null &
```

Re-running this command is safe: the existing host daemon holds its lock. No
wildcard process killing or deletion of partial results is needed for recovery.


## Verified full-run outcome (2026-10-30)

Task `TL-FROZEN-FULL5-QUEUE-20261029` completed all 64 shards, 1300 paired
requests and 3900 generations. The fixed result entry is
`/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_full5_independent005_20261029`.
`FINAL_REPORT_ZH.md` is the final report;
`ai_supervision/round_20261029T161704/VERIFICATION_INDEX.md` links the two actual
independent CPU review sessions, their reports and hashed evidence.

Relative to TLI gates-off (not dense attention), Far-only AVG5 changed by
-0.082408 pp and independent Near+Far by -0.085406 pp. Both reduced Qasper by
0.683604 pp, with paired bootstrap intervals below zero. Musique had no scored
regressions in its 200 samples; its sole tiny F1 increase still gave a wrong
answer. No predeclared equivalence margin exists, so this is not a no-loss claim.
Near-only occurred in 33/46800 layer-requests (19/1300 requests), Both in 2/46800
(2/1300). The threshold and generation protocol were not changed.

The independent reviews bind the same 1300 request files, digest
`fd595e95527362034fe2bf859061d93060c378cf3b179ce38ed21b98de8fa431`.
The expanded 1533-file binding digest is
`901e031bbf6110667ebcdabacad7c03447e3750cea84910deddb84206b5471f5`.
Acceptance covers recorded inputs, predictions, scoring, bootstrap, exports and
logged gate/cache invariants. It does not certify independent GPU regeneration,
unrecorded per-step GPU tensors, historical full-weight hashes, speedups or total
GPU-memory savings. The frozen production HEAD remains
`3e4a6885604ac3e719ee92a4906f87b532535314`; this entry changes documentation only.
