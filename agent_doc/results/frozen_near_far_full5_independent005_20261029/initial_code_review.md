# Independent code review: autonomous full-dataset queue

Reviewer: /root/verify_pool_initial. Code-only review; no GPU jobs launched,
no processes stopped, no source edited. Reviewed current working-tree files;
SHA256 bindings below identify this uncommitted implementation.

## Verdict

The identified worker-publication recovery blocker is fixed. No additional
blocking defect was found in the reviewed fixed-plan path. This is not a
future experiment/data acceptance, GPU runtime proof, or absolute bug-free claim.

## Finding and verified correction

Initially `tick` released the durable lease on any exception after Popen,
including PID-publication failure. Independent fake-Popen fault injection
reproduced a live worker's job becoming `pending` with no PID. That could
cause redundant relaunch attempts and exhaust retries despite the original
worker still running.

The corrected code initializes `proc=None`, releases the lease only when
Popen has not succeeded, and preserves `claiming` after post-Popen failure.
`find_worker` can then recover the original process by exact run/rank.
The added regression checks the publication fault, prevents another host
claiming the live lease, and verifies recovery to the same PID/running state.
I independently reran the updated queue tests: **6 passed**.

## Checked mechanisms

- Shared `flock` transaction protects claim/state write; UUID token prevents
  stale owners finalizing a different lease. One active lease per host/GPU.
- Rank checkpoint flock in the benchmark remains an additional double-write
  guard. Scheduler retries use the same fixed64-rank partition and run identity.
- Daemon per-host flock avoids duplicate host scheduling; orphan scan covers
  daemon death between Popen and PID write. Wrapper resumes daemon after
  nonzero exit without depending on the parent SSH session.
- Worker failures count attempts, return to pending below the cap and become
  failed at the cap. Completed/failure states are durable. Final attention
  marker is written if all remaining work is terminal and some jobs failed.
- Stall termination checks this run/rank command and requires PID==PGID before
  signaling its group; no broad process-name kill. GPU idle threshold and
  predecessor worker_done gate are checked before claims. Actual protection
  of local GPU1 depends on the deployed host/GPU plan; code does not enumerate
  unauthorized hosts beyond that plan and verifies physical hostname.
- `--samples 0` selects all rows. Independent 64-way check covered exactly
  1300 unique items, 20 or21 per shard, with no missing or duplicated row.
  Positive sample count retains legacy selection/order and manifest fields.
- Validator/analysis derive expected coverage from selected_rows, so the64
  logical manifests can aggregate all1300 records. Finalizer lock serializes
  validate/analyze/export. Export uses actual task counts and no longer labels
  this as a20-row screen. FINISHED remains pending independent review.

## Checks run

- tests/test_remote_frozen_queue.py:6 passed after the correction.
- tests/test_frozen_skip_full_dataset.py plus tests/test_frozen_skip_analysis.py:
  6 passed.
- Independent 64-shard partition check:1300 distinct rows, sizes20/21.
- No actual worker process was created for the fault-injection check.

## Deployment limitations to record

- Shared-filesystem cross-host flock behavior still needs actual deployment
  verification; local multiprocess regression alone does not establish it.
- If a physical host remains offline, its active leases are not stolen by
  other hosts: reconciliation is host-local. Its watchdog/restarted daemon
  must recover them, or an explicit audited reassignment is needed.
- Worker retry cap does not bound daemon infrastructure-error retries; the
  daemon deliberately reports/retries those until the infrastructure returns.
- GPU idle checks cannot reserve a card against an unrelated external
  scheduler; use only the authorized pool and verify process/memory/log state.
- `after_run` checks existence of predecessor completion, while resource idle
  checking supplies the second gate. Freeze the correct predecessor directory
  and host-to-rank mapping in the deployed plan.
- Final scoring should use the pinned common backend; workers explicitly pin
  difflib, while daemon finalization inherits daemon environment. A divergent
  daemon backend would fail mechanical score checks; pin it at launch too.

## Source bindings

```json
{
  "scripts/remote_frozen_queue.py": "90190c1ba4fd61935486bd32a4440ed0253371c25f1373d11e17268fb9717f6f",
  "scripts/run_remote_frozen_queue.sh": "d5a1869bb24319ae8e037b90f0eb7564ccefbbf329b44c21c0c4280a2fc982ab",
  "benchmark/LongBench/prefill_frozen_skip.py": "11daac6077e0985103b0f416c5749ae68ac0df8d1b5e1f259c6ce2cc584260a4",
  "benchmark/LongBench/export_frozen_skip.py": "10881896734c1f0c9a6fde1123b9c7b8f00427941d2709202f95ef9a165132ad",
  "tests/test_remote_frozen_queue.py": "02f60aa1fbbd39074aa0ac614ba5ac2c82b0b69cf3cfcfe65247e9bc28ef2ed4"
}
```
