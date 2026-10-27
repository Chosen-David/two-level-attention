# Prefill pair tail cast — 2026-10-27

Main 17e56e8fe30cf8037caf3ce7b38040e754f0521e; PR10 parent
277c46480f90c82a4baf224e94ed41a6e63bd5f7. Main source blob
69eb1641deaebb96cdc8f17de6b5789e914088fe was fetched and matched locally.
Root CLAUDE unchanged (83dbec28), no nested AGENTS found. Read TASK, current
narrative, prior summary/linear-ID/capacity evidence. PR10 and other branch heads,
reviews/comments and main experiments initially unchanged; this run still analyzed code.
Isolated candidate directory preserves the previous PR snapshot. No guide edits.

## Pre-submit main advancement

Main advanced to8f213e48926431ea2d56cf3e7bc1a667d1105c74 during the run.
Integrated all4 upstream files by three-way merge: builder source release, stricter
source weakref test, E177 qualification and hybrid validation addendum. Preserve
PR fusion/workspace changes; resolve E177 wording by retaining both qualifications.
The observer/indexer source is unchanged in new main, so its baseline remains valid.
After integration:68 CPU passed,55 CUDA skipped in12.66s (integrated-cpu.txt),
including linear-gather and incremental-mask tests in addition to the original set.
New remote GPU reports are scoped to that main source, not this integrated PR.
Read add() source: release occurs after copy; source and destination of the layer
being copied still coexist. The upstream max(src,dst) phrase is not an exact peak
bound, and source allocator reservation/graph pools need actual GPU measurements.

## Actual path and candidate decisions

- qwen3_attn_patch prefill: clear -> observe_prefill_q -> calibrate_dyn_dgate,
  before attention. decode consumes selected pairs in _subspace_indices; clear
  resets the next prompt. The observer is disabled unless tli_static_pair=True.
- main observe_prefill_q line496 converts full [B,H,S,128] Q to FP32, then slices
  the last256 rows. PR10 had the same expression. Keep candidate: slice first,
  cast only [B,H,min(S,256),128]. abs/mean/topk/order and caching unchanged.
  Work of conversion falls from O(BHSD) to O(BH min(S,256)D), no new buffer cache.
  FP32 inputs already need no cast; no speed benefit is claimed for that case.
- This is prefill setup shared by subsequent eager/Graph decode, not a decode
  launch optimization. It neither changes ragged request policy nor batches gate
  decisions, and it does not reclaim any KV. Existing hybrid index/attention in
  main and PR10 remains a Python loop over request rows.
- Dynamic gate line560: explicit GQA repeated K plus full logits/probabilities.
  With B1/H32/Hkv8/S65536/D128/R256, FP32 repeated K alone is1GiB and each logits
  or probabilities tensor is2GiB (shape accounting, NOT measured peak memory).
  Grouped contraction can avoid repeated K but changes GEMM grouping and requires
  GPU threshold-boundary tests; streaming softmax changes reduction order. Defer.
  Existing decode coarse scores aggregate heads and cannot replace this per-head
  causal softmax contract. Never substitute historical gate quality for validation.

## Implementation choice and validation

Use PyTorch basic slicing (a view) followed by the existing conversion; no custom
Triton kernel is needed to avoid processing discarded data. Primary reference:
https://docs.pytorch.org/docs/2.14/tensor_view.html (checked2026-10-27; local runtime
is Torch2.8.0+cpu, not the docs version). A fused abs/reduction would add maintenance
and different reduction-order risks; no evidence justifies it in this run.

61 CPU passed,30 CUDA skipped in10.76s (cpu.txt). New observer matrix contributes
18 CPU /18 CUDA; covers fp32/fp16/bf16, contiguous/transposed/strided layouts,
length1/255/256/257/1025/4097, B1/2, tied channels, irrelevant NaN prefix, same
request no-op, disabled flag and two request resets. Exact selected IDs, zero
numerical tolerance. Adjacent batch/hybrid tests retain mixed routing coverage.
The CPU dispatch probe records conversion input shape, not inferred timing.

Frozen main method transplanted via AST into the same runtime fails the copy
scope test as expected: [1,4,4097,128] instead of [1,4,256,128] (baseline.txt).
Initial instrumentation required handling aten.to.dtype for inference tensors;
that test harness issue was fixed before the reported run, without changing gates.

CPU paired cast backing storage for transposed bf16 Q [2,4,4097,128]:
16781312 ->1048576 bytes, identical pair IDs (storage.json). This is only the
FP32 cast allocation, not total peak, CUDA allocated/reserved, KV savings or speed.
For illustrative B1/H32/S65536/D128, cast payload is1GiB ->4MiB by shape formula;
this larger configuration was NOT executed. GPU benchmark refuses CPU fallback.
Syntax checks passed. No independent review, CUDA execution, E127 GPU promotion,
full model quality or latency result claimed. Keep PR's pending GPU status.

## Reproduce and next acceptance

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_prefill_pair_tail.py tests/test_tli_batch_parallel.py tests/test_hybrid_batch_cache.py
python -m exp.bench.prefill_pair_tail_storage
python -m exp.bench.prefill_pair_tail_compare --lengths 32768 131072 --batch 1 --heads 32 --repeats 10 --out /tmp/pair-tail-gpu.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

GPU microbench alternates arms, retains raw event/wall samples and peak allocated
increments after warmup, with a frozen old observer and fresh candidate state per
invocation. It measures observer only, without graph replay or full prefill.
Remote acceptance must run the CUDA cases and existing full-model eager/Graph
request-reset gates with static_pair explicitly enabled, paired same prompts and
same production flags. Default static_pair=False receives no claimed benefit.
Next candidates: per-head gate temporary elimination (numerical contract first),
conditional page refresh (same-position/SWA0 counterexample), and deduplicated
linear IDs (zero-score fill and attention order contract). No repeated old fix.
