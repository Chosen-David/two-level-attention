# TL-NFA-GEOMETRY-WORKSPACE-20261102

Source: user's recurring main/branch performance audit. Base:
`ae69e995acd7e9423d4512f8277aa789a5b310a5`; selector blob
`1706d571eb750709d838301bbd9a46201a0678f0`.
Status: host checked; native CUDA/Graph and performance pending. Serial audit.

## Mechanism and scope

Qwen/GLM write current KV, then call the current-query request-batched selector
for each layer. `_batched_geometry` made an int64 token axis `[S_cap]` and
`mid_mask`/`swa_force` bool outputs `[B,S_cap]`, where
`S_cap=cache.max_blocks*block_size`. The caller uses block masks, bypass tokens,
lengths, interval bounds and quotas only. Ranking uses lower/upper bounds;
protected sink/SWA IDs are assembled separately. None of the two token masks
is consumed. All 93 current `sparse_attn`/`tests` Python sources were searched;
the names and helper call occur only in this module. GitHub code search was
incomplete and was not used as proof of absence.

Delete just these dead allocations/comparisons and dictionary entries. Preserve
the private helper signature to avoid unrelated call changes. No new kernel,
workspace cache, synchronization, host gate, selection rule, numeric threshold
or public configuration. Ordinary tensor work occurs in eager and is recorded
by CUDA Graph; removing it is preferable to fusing/caching unused masks. This
does not fix remaining owner tables, fine-loop tails or both-skip work.

At B=32/S_cap=131072, the removed final masks total 8 MiB and the int64 axis
1 MiB. This is logical tensor payload, NOT GPU peak memory, HBM traffic or
measured speedup. Intermediates and allocator reuse are not counted; Graph
private pools may reuse storage across layers, so do not multiply by layers
and call that measured savings. No KV-residency savings are claimed.

## Evidence and overlap

- Baseline workspace regression fails: requested arange widths
  `[2048,131072,64]` include full token capacity.
- Candidate: two host tests pass normally and with `python -O`; 540 paired
  geometry/candidate-assembly cases have exactly equal retained tensors and
  candidate IDs/masks. Block sizes 4/8/64, zero/aligned/partial sinks, zero and
  nonzero windows, near widths, empty/short/ragged/boundary/capacity lengths,
  two KV-head rows and four region-mask combinations. NumPy executes actual
  function ASTs; not Torch/Triton/native selection or model validation.
- Added native paired geometry test: B=1/4/32, capacity 32/64/128K, eager plus
  captured replay after device lengths change. Fixed exact comparison, no
  relaxed tolerance. Skipped here: Torch, Triton and CUDA unavailable.
- PR #31 `c48a50ab` and #30 `0bd69289` retain the dead masks. PR #10
  `2449f569` has no current helper. #22/#23 old fine-output/constants work is
  not this change. Independent of #31's coordinate-pairing fix; preserve both
  if merging. No unmerged code dependency or branch transplantation.
- All 65 existing remote heads and 31 PR heads unchanged at initial refresh;
  all reviews and repository issue/inline comments empty. Reuse recorded
  branch-specific diffs, dates and ancestry: `opt/l2-fine-topk` is a main
  ancestor; `opt/ako-pool-score` and `e175-skip-ring-kv` are old hybrid/ring
  implementations. Git metadata does not establish live task activity.

## Reproduce and complete native acceptance

From this branch checkout, preserve the exact baseline outside the checkout:

```bash
git show ae69e995acd7e9423d4512f8277aa789a5b310a5:sparse_attn/indexer/nfa_layer_batch_selector.py > /tmp/nfa-geometry-before.py
NFA_SELECTOR_SOURCE=/tmp/nfa-geometry-before.py python tests/test_nfa_geometry_workspace.py -v
# Above intentionally fails the workspace regression on the old source.
NFA_GEOMETRY_BASELINE=/tmp/nfa-geometry-before.py python tests/test_nfa_geometry_workspace.py -v
NFA_GEOMETRY_BASELINE=/tmp/nfa-geometry-before.py python -O tests/test_nfa_geometry_workspace.py -v
python -m pytest -q tests/test_nfa_current_query_selector.py
```

First two paired commands automatically run the native geometry test when CUDA
is available; inspect skips rather than calling a skipped run native PASS.
The existing current-query suite covers noncontiguous request IDs, mixed gates,
first-seed consumption and replay with changed Q. It does not replace a paired
whole-model test or EOS/slot-reuse/repeated-wrap/new-owner capture acceptance.

For paired whole-model timing, use isolated base/candidate checkouts on the same
idle GPU with identical model/data/sample and warmup. Alternate A/B/B/A for at
least three rounds, both eager and graph, B=1/2/4 and 32/64/128K where feasible:

```bash
python benchmark/efficiency/benchmark_nfa_end_to_end.py \
  --model-path "$MODEL_PATH" --data-root "$DATA_ROOT" \
  --output "$OUTPUT_JSON" --task niah_single_1 \
  --context-length 32768 --batch-size 1 --method nfa --execution graph \
  --sample-index 0 --input-provenance "$INPUT_PROVENANCE" \
  --max-new-tokens 128 --warmup-steps 3 --visible-len 8192
```

This entry currently supports only B=1/2/4 and duplicates a prompt across the
batch; report that limitation. At this base, its `nfa_args()` explicitly sets
tail/ratio4 and K1=128/K2=1024; preserve this config in both arms. Also profile actual geometry CUDA
launches/allocations on B=32 and mixed unequal lengths. Compare generated
tokens, selected IDs/counts, eager/Graph agreement, whole-selector/model TPOT
and allocator peaks separately. A geometry microbenchmark is attribution, not
whole-model speedup. No native performance result has been obtained here.
