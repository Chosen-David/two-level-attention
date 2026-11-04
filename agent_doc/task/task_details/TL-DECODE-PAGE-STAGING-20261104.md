# TL-DECODE-PAGE-STAGING-20261104

User request: recurring actual-main performance and branch audit. Baseline:
`37a8e5d386fd80dbbef3d775e39990c27f790138`. Status: host regression complete;
native Torch, CUDA correctness and paired GPU performance **pending**.

## New evidence and mechanism

The new main retains all physical KV pages until request removal. Near/far gates
affect selection, not physical eviction. It must not inherit old ring-cache
memory-saving claims. `GraphRunner.step -> new_step -> _prepare_metadata ->
plan_common -> _prepare_common` is the cross-page route; eager planning also
uses it. Inside `_prepare_common`, `_sync_block_table` already writes each new
physical page ID into CPU staging. The subsequent `_write_block_table_row`
clears and rebuilds every layer's complete historical row unnecessarily.

Remove that one call. Keep allocation, append-only/capacity checks, the full-row
H2D copy, pinned buffers, fixed device addresses and all kernel/selector semantics.
`add_request` clears staging, `write_prefill` populates each layer, `freeze`
refreshes all layers, and `remove_request` clears the slot before reuse. No other
production mutations of `state.block_tables` were found in this cache class.

At 36 layers and 128K/64 pages, a single cross-page decode rebuilt 73,764 page
cells in addition to the 36 appended cells. Removing the rebuild eliminates
those 73,764 writes and 36 row fills; it does NOT eliminate the full H2D row copy
or `_sync_block_table`'s Python append-only validation scan. Vectorized Torch
reconstruction adds a temporary; a GPU scatter adds a kernel and requires
allocator metadata anyway. Reusing the existing incremental writer is narrower
and cheaper to maintain than either. Further H2D slicing/allocator redesign is
deferred until profiling distinguishes remaining cost.

Archived evidence (under `benchmark/efficiency/results/nfa_flashinfer_20261104/e2e`):
`ruler128k_b1_graph.json` and `arms/ruler128k_b1_graph/` NFA A/B raw JSONs.
H20-3e, Qwen3-8B, prompt length 130986, 512 generated tokens / 511 decode steps,
first 3 excluded. Both repeats have spikes at zero-based decode indices
22,86,150,214,278,342,406,470, exactly `(130986 + index) % 64 == 0`.
NFA A median/mean = 26.2200/28.5485 ms; boundary steps 169.10–172.54 ms.
NFA B median/mean = 26.1951/28.4343 ms; boundary steps 166.76–169.05 ms.
FlashInfer A median/mean = 14.0116/14.0142 ms. This is a source-supported
cross-page hypothesis, NOT a measured benefit from this patch or an explanation
of all steady-state regression. Median-derived throughput hides the periodic tax.

The archived producer records dirty head `c597e36e0c8979201cb4c56cafbb61795bf460f3`
plus diff SHA256 `f97c70179fa1a653231210c40e39b88429d1d940f7d56f1999f7b7d631ef1bf5`.
The connector cannot decode its `.diff.gz`; exact producer/main source identity
was not certified. Current-main source and host counterexample stand separately.
These are FlashInfer comparisons, not FA2 comparisons. Dynamic gate receipt is
13 near-skip layers, no far/both skip; do not infer no-skip from the separate
`enable_layer_skip: false` selector configuration field.

## Validation and reproduction

`tests/test_decode_page_staging_host.py`: actual source methods under a NumPy
facade, 3 methods pass in normal and `-O` execution. Nine layer/page configurations
(1/4/36 layers × page4/16/64), two slot lifetimes, two ragged/reordered requests,
20 steps per lifetime, multi-token updates and padding; independent metadata/slot
oracle. Capacity, backwards-length and non-append-only guards still reject.
Preserved baseline passes metadata checks but fails work-count test:
73,842 assignments vs expected 78. Candidate gives 78. This is host logic/work
count validation, not Torch timing, CUDA execution or measured memory reduction.

The added native regression in `tests/test_graph_ready_paged_region_cache.py`
uses real prefill/freeze, four gate combinations, request removal/readmission and
repeated page crossings. It and required E127/batch gates could not start locally:
`ModuleNotFoundError: No module named 'torch'`. Python syntax compilation passes.
No numerical tolerance changed; no GPU correctness/Graph/quality claim is made.

```bash
git show 37a8e5d386fd80dbbef3d775e39990c27f790138:sparse_attn/cache/paged_region_cache.py > /tmp/decode-staging-baseline.py
python tests/test_decode_page_staging_host.py
python -O tests/test_decode_page_staging_host.py
# Expected work-count failure, other host checks pass:
NFA_CACHE_SOURCE=/tmp/decode-staging-baseline.py python tests/test_decode_page_staging_host.py
# Run with actual project dependencies; skip is not success:
PYTHONPATH=. python -m pytest tests/test_graph_ready_paged_region_cache.py tests/test_multi_request_graph_runner.py
PYTHONPATH=. python exp/bench/test_e127_gates.py
PYTHONPATH=. python exp/bench/test_tli_batch_gates.py
```

GPU paired test: create two clean worktrees at the baseline SHA and this PR's
head. In each worktree run the following command, alternating base/candidate
ABBA in fresh processes on an otherwise idle GPU. Set MODEL_PATH/DATA_ROOT to
the same assets and OUT to a distinct absolute JSON path for each run.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmark/efficiency/benchmark_nfa_end_to_end.py \
  --model-path "$MODEL_PATH" --data-root "$DATA_ROOT" --output "$OUT" \
  --task niah_multikey_1 --context-length 131072 --batch-size 1 \
  --method nfa --execution graph --input-provenance native-ruler \
  --sample-index 0 --max-new-tokens 512 --warmup-steps 3 \
  --measurement-mode latency --gate-epsilon 0.05 --visible-len 8192
```

Repeat for eager, B2/B4 and 32K/64K; verify the runner's recorded tail/ratio4
configuration, input identity, gates and generated-token equality base/candidate.
Report raw `tpot_ms_all`, boundary/nonboundary mean/p95/p99, total decode time,
wall-clock request time and capture cost separately. Native regression must also
cover EOS, mixed request lengths and new request recapture. Accept only after
unchanged selections/outputs and a reproducible boundary/total latency reduction;
do not extrapolate this host edit to compact-selector quality or ring-wrap safety.

## Coverage, overlap and next candidates

Previous checked main: `ac9328776ef6fd888141b542fb586e3f0eb26fa9`.
This round reviewed new append-only lifecycle, CPU staging writers/consumers,
GraphRunner planning, current batch selector dispatch and new raw timing arrays.
Current paged selector uses batched request tensors and fused kernels; it is not
the historical per-request hybrid loop. Full-width fine selection/visible-bucket
cost remains a separate pending GPU hypothesis, not fixed by this change.

All 73 remote branch heads (72 development) and 39 open PRs were re-read.
No merged PRs returned. Development heads unchanged; ancestry recomputed against
current main (12 historical ancestors). Compact machine-readable coverage is in
the adjacent `.branches.json`; unchanged branch evidence is reused, not claimed
as newly GPU-verified. No activity inferred from names or commit dates.
Today's issue/inline review-comment queries and PR10/38/39 review lists are empty.

PR10 head `2449f56983d7e97c5fe1c4df300c2b50b0c16b81` remains an older ABI.
PR38 head `e26f2797dcad3dd279f353dcac0c8465415c833d` optimizes prefill; latest main
already implements layer-local prefill staging. This decode-only fix has no PR38
dependency. PR39 split-V write work is independent and is not replaced here.
`opt/l2-fine-topk` is a main ancestor; `opt/ako-pool-score` and `e175-skip-ring-kv`
retain 2/3 unique commits relative to both main and PR10, but their ring/legacy
ABI contracts conflict with current append-only residency. No unverified port.

Next: GPU verify this patch, then quantify residual full-row H2D and append-only
scan separately; resume unresolved wide selector/bucket GPU profiling. Do not
repeat the eliminated whole-row reconstruction analysis on unchanged code.
