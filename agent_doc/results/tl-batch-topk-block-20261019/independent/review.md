# Independent review: non-64 candidate-topk guard

Verdict: no blocking finding. This narrow configuration guard is suitable for source release without a GPU, provided release notes explicitly retain the GPU/Graph/model/performance validation gap. It does not certify GPU numerical equivalence or historical experiments.

Reviewed baseline `4f577242bcd88aa30e2eb25107961030fd3c4aa9` and the single production diff in `sparse_attn/ops/batch_select.py`. No production files, tests, task index or human guides were edited by this reviewer.

## Source reasoning

- `_zero_pad_pool` uses 64 in block offsets, token slices and padding width; `nbad_f_l`/`nbad_n_l` also subtract candidate counts multiplied by 64. The branch was not valid for arbitrary block sizes. The new predicate prevents all candidate-only sorting, gather, topk and padding code from running for non-64 sizes, retaining the pre-existing full-width L2 route.
- `_gate_ok` allows positive aligned non-64 configurations; the production caller in `eager_prefill.py` filters group ends by `_bs`, not 64. Thus this is a reachable configuration defect, not merely a synthetic unreachable case. The existing CUDA gate is unchanged; CPU checks intentionally substitute it and the score kernels.
- The graph signature includes `bs` in geometry and the effective `use_cand_topk` boolean in modes. A non-64 env toggle now correctly shares the same full-width signature; 64 still gets different signatures. `require_graph=True` demands graph execution, not candidate-topk specialization, so full-width graph routing remains a valid required-graph result.
- The patch does not change K2 formulas: clamp K2 to row width; subtract protected sink/SWA; reserve near by `nb_near * bs * gamma`; clamp far to pool width; let near receive remaining budget subject to near width. The full-width branch retains the source selector's pool starvation semantics. Protected tokens can exceed a deliberately undersized K2; the checks do not mislabel that existing behavior a new violation.
- `bs == 64` behavior and env-off behavior are unchanged relative to baseline. Shared-GQA index optimization is orthogonal and retains its own geometry guard.

## Independent CPU evidence

`check_cpu.py` loads the baseline source directly from local Git and compares it with the changed module. Ordinary Python and `-O` each pass:

- 50 small block/budget/shared-index cases: bs in 16/32/64/128/256; K2 in 0, 2*bs, 2*bs+1, 512, 2048; shared indices off/on. Non-64 enabled output equals baseline full-width output bit-for-bit; all default paths and the enabled 64 path equal their respective baseline paths. Also checks protected regions and selection-count bounds.
- Five host-only graph-signature cases verify effective mode and key equality/inequality, using a CPU body executor in place of CUDA graph capture/replay.
- `git diff --check` passes.

These are finite host/glue checks with deterministic fake scores, not real Triton scoring, CUDA tie-breaking, capture/replay or model tests. CPU candidate-vs-full-width parity at bs64 is intentionally not used as a substitute for CUDA tie-order validation. Existing bs64 zero-padding/tie limitations remain outside this fix. Non-64 performance may change because it now takes full-width topk; no throughput claim is made.

Release boundary: approve the local guard and its regression coverage; keep real-device Graph parity, actual-model quality/performance and impacted historical-grid reassessment explicitly pending. The new guard does not broaden supported GPU kernels or alter their arithmetic.
