# Prefill-frozen Near/Far skip

Opt-in Qwen3 and GLM-4.7-Flash TLI eager inference reference. This does not change the default TLI
path or enable operator/graph optimizations. Supports unpadded batch size 1,
one-shot dense material-prefix prefill and single-token suffix/answer decode.

The Quest input is tokenized once after rendering. A task anchor or the final
100 characters, whichever starts later, defines the requested suffix boundary;
a BPE token crossing that boundary belongs wholly to the suffix. The canonical
invariant is `prefix_ids + suffix_ids == full_ids`. Only the material prefix is
densely prefilled, profiled and frozen. The suffix is then replayed token by
token through the frozen cache before ordinary answer generation. A full-prompt
prefill followed by prefix recomputation is not an equivalent protocol.

Prefill profiles the last min(256, material-prefix length) queries, with causal,
query-relative page boundaries. Each layer independently compares mean Near
and Far attention mass with **0.05**, strictly less than the threshold. Decisions
freeze for suffix replay and the entire answer decode, without recalibration.
Both regions may qualify independently; there is no combined mass budget.

The physical cache chain is `sink -> far -> near -> SWA`, with a linear Far
buffer and bounded Near/SWA rings. Geometry: sink 4, SWA 128, Near 4096,
page 64. Absolute token positions remain independent of packed cache offsets.

| Frozen gate | Decode behavior |
|---|---|
| Keep both | SWA overflow moves through Near to Far |
| Far skip | Release Far; discard tokens aging beyond Near |
| Near skip | Keep invisible Near staging; aging tokens enter visible Far |
| Both skip | Keep only sink/SWA; discard SWA overflow |

Near-only saves attention reads, **not resident KV capacity**. Scoring and
attention never read staging until it ages into Far. Far and both skip
physically reclaim the corresponding cache nodes.

`FrozenRegionCache` owns node lifetimes; `GateConfig`/`profile_regions`/
`decide_regions` calibrate the gates. The Qwen3 and GLM patches recognize the explicit
`tli_regions` cache marker before generic KV update. The benchmark installs
the patch, captures prefill profiles, freezes the cache and scopes eager
request guards inside inference mode. Other callers must perform the same
lifecycle; passing ordinary DynamicCache does not activate these gates.

The adapter calls the existing TLI compute_score/compute_mask on tail32 key
scratch and projects its selection onto visible positions. Final attention
uses full-width K/V. L1=128, L2=1024, Far=minmax, Near=avg, Near pages=32,
Near tokens=off. Projected scratch is temporary logical-length storage: this
is an accuracy reference, not a measured peak-memory or speed optimization.
Beam search, chunked prefill, batches, gradients and CUDA Graphs are unsupported.

Each suffix and answer-decode query records `attention_work`: logical position,
visible KV and actual unique valid attended keys per KV head and layer. TLI uses
the real post-L1/L2 `selected_per_kv_head`; the non-TLI reference uses the visible
cache size. K1=128 is only the coarse-candidate budget and K2=1024 only the fine
selection budget; neither is reported as actual attended-K. These decode counts
exclude dense prefill. The older `frozen_attention_token_ratio_mean` is only a
freeze-time visible-KV proxy, not a decode-work measurement.

## Run the validated comparison

```sh
TLI_SCORER_BACKEND=difflib python -m benchmark.LongBench.prefill_frozen_skip \
  --attention tli --thresholds 0.05 --samples 0 \
  --model /path/to/Qwen3-8B --data /path/to/LongBench/data \
  --device cuda:0 --verify-native-full \
  --out agent_doc/results/frozen_region_skip_RUN_ID
```

This runs gates-off (`full`), Far-only and independent Near/Far on all rows of
the 16 English/code LongBench V1 tasks: NarrativeQA, Qasper, MultiFieldQA-en,
HotpotQA, 2WikiMultihopQA, MuSiQue, GovReport, QMSum, MultiNews, TREC, TriviaQA,
SAMSum, PassageCount, PassageRetrieval-en, LCC and RepoBench-p. `full` means TLI
gates-off, not dense attention. `--samples 20` selects a smaller subset. For
disjoint workers use matching `--world-size N` and unique `--rank 0..N-1`; use
a fresh output directory when code, input or configuration changes.

The default 31,500-token budget truncates the raw prompt before chat-template
rendering; it is **not** a final 32,768-token input cap. TREC, TriviaQA, SAMSum,
LCC and RepoBench-p do not use the chat template. SAMSum uses EOS/newline stop
IDs with first-token EOS suppression. The runner pins operator mode off, fine
gather off, dense material-prefix prefill and sparse decode. Its generic
`--attention sdpa` mode is a separate reference experiment.

After all workers complete:

```sh
python -m benchmark.LongBench.validate_frozen_skip agent_doc/results/frozen_region_skip_RUN_ID \
  --data /path/to/LongBench/data
python -m benchmark.LongBench.analyze_frozen_skip agent_doc/results/frozen_region_skip_RUN_ID
python -m benchmark.LongBench.export_frozen_skip agent_doc/results/frozen_region_skip_RUN_ID
```

The validator reconstructs the same canonical token stream and verifies prefix
freeze, suffix replay, stop rules, per-step attended-K and cache invariants.
Analyzer/export use a task-stratified macro average over exactly the tasks in the
manifest (16 for the default run), not a hard-coded five-task average.

## Accepted evidence and limits

Historical **old one-shot full-prompt, five-task** evaluation: source
`3e4a6885604ac3e719ee92a4906f87b532535314`, 1300 paired inputs / 3900 generations /
64 completed shards. Dual AVG5 48.239706 -> 48.154300 (-0.085406 percentage
points); largest task decline Qasper -0.683604 points. Musique had no per-request
score decline. User accepted up to 3 points of task-mean/AVG5 degradation after
inspecting the results; this is engineering acceptance, not a pre-registered
equivalence test. Near-only activated in 19 requests, both-skip in 2; coverage
is limited. These figures are retained as historical evidence and are not Quest
prefix-freeze or 16-task main-protocol results.

Independent review checked stored predictions, scores, input identities and
cache/gate receipts, without independently rerunning GPU generation. Evidence
is retained under the project result directory
`agent_doc/results/frozen_near_far_full5_independent005_20261029/FINAL_REPORT_ZH.md`.
These historical quality figures belong to that frozen source, not a fresh
full evaluation of later main changes. Main integration checks cover the
four-state cache lifecycle, TLI selection parity, patched Qwen logits and
benchmark request guard/data coverage. No latency or whole-device memory claim.

## GLM-4.7-Flash

Use the same entry point with `--attention tli --model /path/to/GLM-4.7-Flash`.
The GLM adapter preserves native MLA projections and interleaved RoPE, then
routes the post-RoPE Q/K and full V to the region consumer before any generic
cache write. Prefill profiling uses the model attention scale, including any
configured scaling adjustment, rather than assuming `1/sqrt(head_dim)`.
The selector retains GLM's tail pairs `[208:224] + [240:256]`.

Only unpadded B1, one-shot prefill and single-token eager decode are supported.
GLM rejects active graph state, operator mode, outer batch contexts and external
masks before projection/cache mutation. Its ordinary dynamic far gate must be
disabled: the independent near/far gate owns the request decision. The generic
`--attention sdpa` segmented-cache reference remains Qwen-only. Transformers 5
integer query-length mask requests and the older tensor API are both accepted.

Qwen historical scores above do not establish GLM quality. Four-state routing,
cache lifetime and gates-off parity are integration checks; GLM task accuracy
requires a separately recorded evaluation.
