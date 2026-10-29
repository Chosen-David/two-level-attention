# Prefill-frozen region skip with explicit Near staging

Branch: `explore/prefill-frozen-near-far-skip`, isolated from the concurrent main
worktree. This first implementation isolates physical KV eviction with native
Qwen3 SDPA; it does not add TLI L1/L2 selection or claim a speedup.

The cache is an actual linked chain of independently owned buffers:

```text
sink -> far -> near -> SWA ring
```

Dense one-shot prefill returns the original complete K/V to attention. Its last
256 queries are profiled using their own causal, page-aligned near/far boundaries,
sink4, near4096, SWA128, page64. Scores are head means, summarized by mean or P95
over query rows. Near and far get independent signals. Decisions freeze before
the first decode update. Default exploration tests epsilon 0.10 and 0.15. When
both regions qualify but their calibrated sum exceeds epsilon, only the lower
mass region is evicted; otherwise both may be deleted. This is a joint calibrated
budget, not a guarantee about future attention or accuracy.

Far-skip and both-skip release the evicted nodes' owned tensors; slices into a
full-history backup are never retained. Near-only skip instead renames Near to
an explicit attention-invisible staging node. Attention gathers only visible
nodes; staging is still counted in resident KV bytes.
SWA is a fixed-address ring; the near node also uses a bounded FIFO internally.
Absolute RoPE positions and logical sequence length continue increasing.

## Permanent eviction and migration

| Frozen decision | Retained nodes | Subsequent overflow |
|---|---|---|
| keep | sink, far, near, SWA | SWA -> near -> far |
| far skip | sink, near, SWA | SWA -> near; matured near KV discarded |
| near skip | sink, far, Near staging, SWA | SWA -> staging -> far; staging excluded from attention |
| both | sink, SWA | SWA overflow discarded permanently |

For **near-only skip**, Far remains complete and updates as the distance boundary
moves. Near KV remains in a bounded staging FIFO; it does not participate in
attention while near, and becomes attention-visible when it ages into Far.
Consequently this case saves attention work, **not resident KV capacity**.
SWA overflow is staged normally. The physical chain is
`sink -> far -> near_staging -> SWA`, whereas attention reads only sink/far/SWA.

Far-only still evicts far; near overflow into the absent far node is discarded.
Both-skip permanently evicts both regions and drops SWA overflow. The earlier
Near permanent-deletion implementation at commit486a406 was superseded by the
user's staging decision. Its run `frozen_near_far_20261029` was stopped with63
completed requests; those raw results remain isolated and are not reused.
Current protocol schema is `frozen-near-far-screen-v2-near-staging` and output
is `agent_doc/results/frozen_near_far_staging_20261029/`.

Supported: Qwen3 full attention, B=1, unpadded, inference, one-shot dense prefill,
single-token eager decode, request-local cache. Beam/reorder, rewind, chunked or
batched prefill, offload, gradients and CUDA graphs are rejected/unsupported.
Near staging is explicit; no additional full-history backup exists. Runtime concatenation temporaries
are excluded from the resident-node byte counts; no allocator or throughput
claim follows from those counts.

## End-to-end screen

Use exactly the existing five screening tasks: qasper, hotpotqa, musique,
gov_report, repobench-p. First exploration: 20 seeded-random paired rows per task,
100 requests total, 5 arms = 500 generated results. These are **not** the full
200/500-row screening datasets. Raw prompt context cap31500; repository prompts,
Qwen3 non-thinking chat template except repobench-p, task-specific official
generation caps, EOS, greedy, repository `sample_score` (repobench alias,
difflib backend). All arms freshly run; first prediction must agree because
prefill remains dense. Full node-cache generation is checked against native
DynamicCache on the first request of each worker.

Arms: full, far-only epsilon0.10, dual epsilon0.10, far-only epsilon0.15,
dual epsilon0.15. Report actual near/far/both activation, per-task scores,
AVG5, paired task-stratified request bootstrap (5000 repeats), physical node KV and attention-token
ratios (Near-only never counts as KV saving), and exact output agreement. A CI containing zero does not prove
equivalence. Activation-free requests do not validate skipped-path quality.

```sh
python -m pytest -q tests/test_frozen_region_cache.py
python scripts/supervise_frozen_skip.py --devices 0,1 --interval 3600
python -m benchmark.LongBench.analyze_frozen_skip agent_doc/results/frozen_near_far_staging_20261029
```

The **local** supervisor checks hourly, records PIDs/heartbeats/GPU state, resumes
unfinished request-arm checkpoints after exit, restarts its own workers only
after two hours without progress, and permits at most three restarts per rank.
It does not change algorithms or kill unrelated processes. After completion it
writes the paired report and exits. Its lock prevents a second supervisor for
the same run; rank locks prevent duplicate workers. Source/input identities
prevent incompatible result reuse.

## Completed exploratory result

Run `agent_doc/results/frozen_near_far_staging_20261029/` contains100 paired
requests (20/task),500 generated outputs, all five screening tasks including
Musique and HotpotQA. Full AVG5=53.8789; dual0.10=54.0548 (+.1759 pp;
95% CI [−.2380,+.6534]); dual0.15=53.3765 (−.5024 pp). Mean non-regression
in the small0.10 screen is not an equivalence result, and GovReport drops.
The0.15 policy has larger per-task regressions. No production recommendation
or TLI-champion integration follows from this prototype.

`REPORT.md`, scores/paired-difference/activation TSVs and compact predictions
are versioned. `raw_requests.tar.gz` restores complete per-layer traces under
`requests/`; `raw_index.json` pins each trace. Generation commit62f6fb4 and
source/data identities are in the manifests/provenance. All500 scores and18000
request-layer states passed mechanical validation;14 unit/integration tests
passed. Hourly supervision automatically completed and is no longer running.
