# Prefill-frozen near/far node deletion exploration

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

Each deleted node releases its owned tensors; slices into full-history K/V are
never retained. Attention gathers only nodes remaining in the linked chain.
SWA is a fixed-address ring; the near node also uses a bounded FIFO internally.
Absolute RoPE positions and logical sequence length continue increasing.

## Permanent eviction and migration

| Frozen decision | Retained nodes | Subsequent overflow |
|---|---|---|
| keep | sink, far, near, SWA | SWA -> near -> far |
| far skip | sink, near, SWA | SWA -> near; matured near KV discarded |
| near skip | sink, far, SWA | SWA overflow discarded permanently |
| both | sink, SWA | SWA overflow discarded permanently |

For **near skip**, far remains the live prefill prefix. Deleted near tokens do
not reappear when their absolute distance would put them in far, and subsequent
SWA overflow does not populate far. Thus this is **surviving-history far**, not
complete moving-distance far. Near-low observations before a token ages into far
cannot alone justify permanent eviction; actual autoregressive accuracy is the
test. Keeping every moving far token would require retaining/recomputing the
near values, which is outside the requested permanent-deletion semantics.

Supported: Qwen3 full attention, B=1, unpadded, inference, one-shot dense prefill,
single-token eager decode, request-local cache. Beam/reorder, rewind, chunked or
batched prefill, offload, gradients and CUDA graphs are rejected/unsupported.
No shadow near or full-history backup exists. Runtime concatenation temporaries
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
AVG5, paired task-stratified request bootstrap (5000 repeats), physical node KV
ratio, and exact output agreement. A CI containing zero does not prove
equivalence. Activation-free requests do not validate skipped-path quality.

```sh
python -m pytest -q tests/test_frozen_region_cache.py
python scripts/supervise_frozen_skip.py --devices 0,1 --interval 3600
python -m benchmark.LongBench.analyze_frozen_skip agent_doc/results/frozen_near_far_20261029
```

The **local** supervisor checks hourly, records PIDs/heartbeats/GPU state, resumes
unfinished request-arm checkpoints after exit, restarts its own workers only
after two hours without progress, and permits at most three restarts per rank.
It does not change algorithms or kill unrelated processes. After completion it
writes the paired report and exits. Its lock prevents a second supervisor for
the same run; rank locks prevent duplicate workers. Source/input identities
prevent incompatible result reuse.
