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

Shared `queue.lock` protects claims; host lock prevents duplicate daemons. A
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
