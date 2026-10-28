# LongBench nominal K2 sweep (bounded remote runner)

Entry: `python -m benchmark.LongBench.run_budget_sweep` (Linux, one local GPU per job).
This adds orchestration, **not a new attention implementation or a measured result**.
It does not start a daemon or schedule future work. Default is dry-run.

## What is measured

- Current checkout's **LongBench-v1 protocol for explicitly selected Qwen3-8B or Qwen3-32B**, all rows in each selected local
  task JSONL, seed 42, BF16, the existing producer's split material/question prompting
  (not the full-prompt protocol proposed in an open PR). `model2maxlen.json` controls
  the actual input cap (current Qwen3-8B entry: 31,500); per-task generation lengths
  and prompt templates come from the same frozen config directory.
- LoDA fixes K1 (default **128 pages**, page size 64), scans **nominal final-token K2
  512 / 1024 / 2048**. Other fixed knobs: minmax/avg, nb32/nt-off/nl4096,
  tail subspace, cmp-ratio 4, sink4/SWA128, layer skip and kmeans explicitly off.
  Current far-skip `auto` resolution is reused and included in expected treatment;
  no changes to selectors, eviction or runtime code.
- Quest maps K2 to `quest_token_budget` (chunk16; first two layers remain dense),
  JSSA to `jssa_budget` (rank32), AsyncTLS to `asynctls_token_budget`
  (block64/top-kb128 fixed). Full runs once per task, not once per K2.
- These are **nominal parameter comparisons, not verified equal effective caps**.
  Protected regions, chunk rounding, layers, calibration and realized unique tokens
  differ. Retain producer budget telemetry; no new equality/performance claim.
  This is a fixed-K1 budget sensitivity comparison, not a complete two-level ablation.
- Default: the reference table’s 13 tasks × (4 sparse methods × 3 K2 points + Full) = 169 sequential jobs.
  Explicitly edit `tasks`/`methods` to limit the requested missing subset; no automatic
  matrix expansion, repetitions, calibrations or model downloads.

The default task set is narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa,
musique, gov_report, qmsum, multi_news, triviaqa, passage_retrieval_en, lcc,
repobench-p. It deliberately excludes trec, samsum and passage_count from the
16-task launcher. The requested RULER-10 32K/K512 reference is **not implemented**
by this LongBench runner; do not relabel existing RULER-11 averages as RULER-10.
The LongBench main-table K1024 anchor and multi-K2 sensitivity plans have separate
plan identities (`k2: [1024]` versus `[512, 1024, 2048]`). Full is once per task per
frozen plan; reuse across different plans is intentionally not assumed.

Set manifest `model` and `model_path` consistently. Models run in separate plans;
selecting 8B does not automatically run 32B. Architecture dimensions are checked
against the official [8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)
and [32B config](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json), then all
local file bytes are frozen. Supply model-specific calibration assets. A 32B job
requires a GPU with enough memory for its full weights/context; choose an explicit
appropriate `--min-free-mib` (40,000 MiB default is intended for 8B, not 32B).
No tensor-parallel or automatic model substitution is performed.

## Prepare, inspect, then explicitly execute

Use the intended existing inference Python environment. Do not prepare with one
interpreter and execute with another. Local model/tokenizer files and dependencies
must already be installed; runner installs nothing and forces HF offline modes.

```bash
# In a clean checkout of the desired revision. Keep manifests/results outside it.
cp docs/longbench_budget_sweep.example.json /your/work/budget-manifest.json
# Edit local paths, tasks and methods in /your/work/budget-manifest.json.
python -m benchmark.LongBench.run_budget_sweep \
  --prepare /your/work/budget-manifest.json --plan /your/work/budget-plan.json
python -m benchmark.LongBench.run_budget_sweep --plan /your/work/budget-plan.json

# Obtain full UUIDs from the site's inventory (or nvidia-smi -L).
# The local lock directory must already exist, with permissions for all cooperating
# users; every checkout/tool must use this SAME directory. Do not use an NFS mount.
python -m benchmark.LongBench.run_budget_sweep \
  --plan /your/work/budget-plan.json --execute \
  --output /your/work/budget-results --lock-dir /your/site/shared-gpu-locks \
  --gpus GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --wait-seconds 600 --idle-seconds 30 --poll-seconds 5 \
  --min-free-mib 40000 --max-used-mib 256 --max-util-percent 0 \
  --job-timeout-seconds 21600 --score-timeout-seconds 1800
```

`--gpus` is a mandatory explicit allowlist of physical UUIDs (no indices/MIG).
Thresholds are admission criteria, **not a guarantee against OOM**. Wait is bounded
per pending job, prediction and scoring have separate finite deadlines. Stops on
first failed attempt, preserving it; rerun the same command to resume. It never
kills another user's job, allocates dummy reservations, or changes compute mode.

Preparation hashes **all model/tokenizer files**, all config files, selected task
files, calibration assets, relevant local source (including ignored adapters), Git
SHA, Python executable, dependency versions and sanitized worker environment.
Execution rejects edits/additions to model/config inventories, source changes,
changed inputs or environment. It rechecks before/after each new job; hashing large
weights is deliberately conservative and can take time. Use immutable inputs during
execution; checks do not provide an adversarial filesystem snapshot against changes
that are reverted between checks. Prepare a new plan after changing a prerequisite;
existing plans are never overwritten. Plans/logs include local paths: do not publish
private machine inventories or dataset/model bytes.

## Current prerequisite gaps and partial execution

At inspected main `e6f0cc9e79f1212d04c3f91577f7ff33255dee1c`, CLI and launch scripts
reference `exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_patch.py` and
`exp/vendor_baselines/jssa_vendor/qwen3_jssa_patch.py`, but those files are not in
this committed tree. JSSA's `CALIB_COMPAT_PATCH.md` documents ignored third-party
and calibration assets; that is not proof they are absent from the execution host.
The runner checks the **actual host checkout**, records local adapter bytes, and
reports missing adapters/assets as blocked arms. It does not fabricate replacements.

Default execute rejects any blocked selected arm before GPU acquisition. Select
only ready methods in the manifest, or explicitly use `--allow-partial` to run ready
arms. A partial matrix retains blocked cells, reports `incomplete`, and exits 2;
never interpret it as a complete comparison. Restoring an adapter requires a newly
prepared identity. Dependency/config preflight also occurs before GPU acquisition.

## Completion, reuse and output

Existing historical tables/JSON summaries do not have this runner's complete input
and execution identity, so they are **not automatically counted as matching points**.
The search of `agent_doc/results/qwen3_8b_longbench_*.json` and current launch scripts
establishes existing nominal anchor summaries, not a fully matched 512/1024/2048
sweep with current code/input/config provenance. No old scores are relabelled.

Each task/arm gets an isolated `attempt-<uuid>` directory with full command logs,
identity, prediction, run-cap receipt and scorer output. New attempts never overwrite
failed or interrupted attempts. The existing `run_cap_manifest` schema/hash/row-count
validation is reused; expected source IDs/answers, model/cap/code, exact resolved
producer filename and TLI treatment are checked. Existing `benchmark.LongBench.eval`
performs scoring with `--require-runcap --expect-count`; the runner additionally
requires exactly one expected, finite [0,100] score. Only then is `complete.json`
atomically written, binding prediction, run-cap, TLI treatment (if applicable), and
score hashes. Restart skips only matching, revalidated completions. A prediction
without successful scoring is not complete. `failure.json` preserves errors.

`<output>/<plan-id>.summary.json` lists task/method/K2, status, score and attempt
paths; `k1`, protocol and nominal-budget caveat remain explicit. All scores are in
LongBench's 0–100 scale. Do not silently average a partial task set into the main
LongBench table or substitute this sensitivity sweep for the established main table.

## Lock and lifecycle boundary

The runner obtains a **UUID-scoped advisory flock**, then repeatedly queries
`nvidia-smi -q -x` for a stable idle window. Free and used memory, utilization,
compute **and graphics** process list, UUID and disabled MIG state must be known.
Missing command, malformed/unsupported telemetry or query failure prevents launch.
Locks use persistent inodes and are closed, never deleted or PID-based force-unlocked.
An independent Linux guardian inherits the lock FD, installs/rechecks parent-death
notification and supervises a separate worker process group. Normal exit, failure,
timeout, SIGINT/SIGTERM/SIGHUP and launcher death clean up managed descendants.

Advisory locks exclude only cooperating tools using the same local directory.
Noncooperating jobs can arrive after a check; this does **not** establish exclusive
hardware ownership or certified latency isolation. Coordinated site scheduling is
still required for performance claims. Simultaneous SIGKILL of guardian and launcher,
host failure, or deliberately daemonized/escaped descendants cannot be promised
universal cleanup; a surviving inherited worker FD retains the cooperative lease.
Only fixed repository producer/scorer commands are launched, never arbitrary shells.

## CPU validation

```bash
python -m unittest discover -s tests -p 'test_longbench_budget_sweep.py' -v
```

Tests use mocked NVIDIA XML and harmless CPU subprocesses, including parent death,
timeout, stale/incomplete/corrupt results and file identity changes. No GPU workload,
pretrained model, quality measurement or GPU performance gate was run for this change.
The repository's `test_e127_gates.py` and `test_tli_batch_gates.py` GPU hard gates
remain **unrun**, not passed. Real remote integration/quality acceptance is separate.
