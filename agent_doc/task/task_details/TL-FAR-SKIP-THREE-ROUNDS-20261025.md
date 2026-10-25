# TL-FAR-SKIP-THREE-ROUNDS-20261025

## Plan (frozen before implementation)

User requests three additional iterations and PR delivery covering eager and
CUDA Graph. Parent PR #7 head: 3e55156bf375a2f925172b3e6ec1da74a383f17e.
No merge/main push. Strict far exclusion, near-page quota, protected tokens,
GQA probability pooling, stable ties and physical bounded KV remain mandatory.
Do not substitute legacy boundary semantics or change quantization to gain speed.

1. CODE-O001: current SDPA represents G query heads as separate heads. Math
   GQA repeats gathered K/V. Reinterpret G as independent noncausal queries;
   compare outputs, operation counts and paired fresh-step CPU timings.
2. CODE-O002: Python lengths/dynamic shapes prevent capture of compact eager.
   Implement a fixed-capacity, tensor-position selector/cache bridge in existing
   Qwen/TLIStaticCache graph entry, initially rebuilding bounded near summaries.
   Compare decisions/outputs to the dynamic eager oracle across page/ring wraps.
3. CODE-O003: profile the fixed path, replace redundant bounded summary rebuild
   with request-local page summaries and compact fine candidates if warranted.
   Compare round 2/3 plus original parent under identical inputs. Retain negative
   findings; CPU timing is never CUDA validation.

Target entry: Qwen3 compact far-off, B=1, append-only decode, DynamicCache eager
and TLIStaticCache staged/whole-model capture. Existing graph benchmark is
exp/bench/e152v4_graph_latency.py; current pred_ruler_batch has no static/graph
entry despite its historical descriptions. Mixed far-on layers retain the old
graph path. Frozen gates/cache/dimensions are request-local and revalidated at
setup. Recapture on a new request avoids stale gate/layout pointers.

Environment: Python 3.12.14, torch 2.8.0+cpu, transformers 4.56.2, no visible
CUDA device. Reuse installed research-implement-optimize workflow and official
PyTorch tools. No added kernel DSL. GPU correctness/latency remain explicit gates.
Baseline snapshots in ../iter-base; existing results searched in
far_skip_optimized_20261025, reused as context only because code changes.

Candidates: ordinary SDPA query-axis grouping (same semantics; portable),
fixed bounded graph selector (static control/data-driven state; padding tax),
immutable complete-page memoization (setup cost and bounded O(near/bs*H*d)
storage). Full-history masks/summaries rejected because their work scales with S.
Handwritten Triton fusion deferred without a device to validate lowering/latency.
The core patterns are cyclic append storage, immutable page memoization and
stable top-k; no pointer-based LRU is needed for monotonic windows.

Primary sources checked 2026-10-25:
- https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://docs.pytorch.org/docs/2.8/notes/cuda.html
- https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html

## Progress (append-only)

- Setup: PR still open at expected parent. No GPU; runnable CPU oracles and
  skipped real-CUDA capture/replay probes will be separated in the report.

- Round 1 (`d7fab93f2142c07acd2cb222aa191c7de0221d14`): changed SDPA layout
  from [Hkv,G,1,D] to [Hkv,1,G,D]. CPU profiler verifies repeat_interleave 2→0
  and allocation traffic 33,693,908→8,382,780 bytes. Final paired eager medians
  at S=8192/32768,n=4096 are 10.62→6.67 / 10.08→6.29 ms versus parent. CPU
  holdouts: S=64,n=1024 0.369→0.221 ms; S=8192,n=1024 2.952→1.011 ms.
  Retain; same noncausal attention semantics, floating-point backend changes.
- Round 2 (`ed308eac7c2da6c0c2814db6cff149ab8f48189e`): fixed tensor-position
  selector/ring adapter, with bounded all-near summary/fine pooling. This is a
  functional graph candidate, not a speedup: CPU prototype 19.18 ms versus
  dynamic 5.79 ms. Retained only for further optimization and reproducibility.
- Round 3: immutable page summary ring, refresh newest complete page (at most
  one becomes eligible per token), gather only selected pages plus partial
  boundary fragments, reuse concatenation buffers. GraphRunner writes changed
  mask intervals and fills persistent inputs; explicit cache/RoPE positions and
  per-request recapture prevent stale metadata/gate/address replay.
  Final same-input fixed comparison: round2→round3 18.51→7.68 ms at S8192,n4096;
  holdout S32768,n1024 8.05→5.03 ms. Setup grows 2.24→7.84 ms because page
  statistics are seeded once. Record both setup and steady state; no hidden work.
- Strong runnable same-semantic library baseline: ordinary eager PyTorch/SDPA.
  Final fixed path is still slower on CPU: 7.52 vs dynamic 5.14 ms in a separate
  paired run. Therefore dynamic eager stays the normal DynamicCache path.
  Fixed-path timing includes graph-style input-buffer copies even without capture.
- Final 119 tests + 205 subtests pass; 5 actual-CUDA tests skip. Final normal
  timing runs compare 840 fresh-step outputs (max absolute error 1.133e-6),
  fixed comparisons additionally assert every selected token ID. No GPU or
  pretrained-quality result. Existing unrelated main AST test failures remain
  outside scope (see v1 record); this is not a whole-repository green claim.
- Tests cover FP32/BF16 CPU selection, avg/minmax, quantization, zero/all page
  budgets, short overlaps, unaligned near lengths, bs=1, poisoned unused slots,
  exact stable ties/invalid priority, ring wraps, real Qwen static versus
  DynamicCache, request/gate changes, input validation before mutation, one-page
  summary reduction and no scalar readback. CUDA probes include changing-input
  capture/replay of IDs/output and whole all-far-off Qwen request recapture.
- Artifacts: agent_doc/results/far_skip_three_rounds_20261025/{summary,validation}.json
  bind final source hashes and raw receipts. Summary regeneration checks hashes
  and recomputes medians. Profiling runs are not used for speed ratios; positive
  self allocation counts are traffic, not peak/net memory or GPU launch counts.
  Profiler warned preexisting allocations' deallocations were untracked.
- Manual implementation review plus executable checks only; no independent
  reviewer/verification agent ran. Remaining GPU gates: capture on target Torch/
  CUDA, mixed far-on/off model decode, gate/layout request changes, SDPA dispatch,
  GPU peak/reserved memory, launch trace, latency and model quality.
- User subsequently requested two more subspace iterations. These three rounds
  remain a frozen evidence checkpoint; new subspace results use a new run folder.

## Complexity and remaining costs

Let n=near length, b=page size, p=selected pages, K=K2, d=subspace width and D=full
width. Both adapters retain O((n+w+s+b)HkvD) KV and O(n/b*Hkv*d) page statistics.
Eager refreshes new complete pages only; fixed graph refreshes one b-token page
per step to avoid host control flow. Fixed L2 is O(Hq*(pb+2b+s+w)*d), replacing
the round-2 O(Hq*(n+b+s+w)*d) pool, plus stable selection. Final attention stays
O(Hq*K*D). Library top-k internals have backend-dependent complexity; no optimal
asymptotic claim. Graph may still pay padded lanes, multiple framework launches,
int64 tie-key construction and masked SDPA dispatch. Short requests retain the
same fixed-reservation memory tradeoff as before; prefill peak is unchanged.

## Reproduction

Use the commands/configs embedded in each raw JSON; checkout the parent for
`far_skip_compare --baseline-root`, round2 commit for `far_skip_graph_compare
--baseline-root`. Run one process at a time, same CPU thread/device and fresh
inputs. `python -m exp.bench.summarize_far_skip_three_rounds` verifies current
bindings; after subsequent edits use `--source-root <round3 checkout>`.
Target-GPU commands and mask/cache lifecycle are in docs/far_skip.md.
