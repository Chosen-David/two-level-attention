# Remove dead SWA probability fill — 2026-10-29
Main a2e7b67b4e5c131cfa169283f7c66d6fb8892b00, indexer blob e4d76e0bb1e81b358bb99247a684ae8c7a579a26. Candidate depends on PR14 c0f73564adf712541cbce9e869d099a842a0fe9a and preserves both dtype guards. No other production changes.

## Actual dataflow
compute_mask_graph creates p via fused GQA mean or Torch fallback. It then writes p[SWA]=1 and immediately produces mid_scored=p.masked_fill(~mid,-inf). p has no later consumer or external owner; final selection separately unions swa_force. Geometry builds mid with token<swa_lo and SWA with token>=swa_lo. Therefore every overwritten p position is unconditionally replaced by -inf before top-k. Removing that intermediate fill preserves mid_scored even for NaN/Inf, empty intervals and zero window.
The full-width softmax and protected-token contribution to its denominator remain. This is not the unsafe protected-score deletion rejected previously. L2 bypass returns earlier; skip_far is excluded by this function's entry guard. Benefit is scoped to execution of this helper (including captured replay), not all eager/skip/non-skip paths.
Existing mean kernel can force SWA itself, but no forcing is needed at all here. Removing the redundant call is simpler than expanding that kernel's dynamic geometry interface. One Torch masked_fill_ call removed, no claimed measured GPU launch/time/memory gain; p capacity unchanged.

## Validation
288 pairs execute actual geometry and masking AST fragments with a NumPy facade: capacities1/129/8192, lengths0/1/cap-1/cap, sink0/4/128, window0/1/128/cap+1, rows1/8; values include NaN and ±Inf. Masked outputs byte-identical. Normal/-O identical; overlap mutation demonstrates removal would fail without the proven disjointness. Exact source comparison confirms only the one fill statement removed from PR14; compile passes.
No Torch/Triton/CUDA available. This is fragment-level host evidence, not full selector, CUDA lifetime/capture, attention output or speed measurement.

## Reproduction and pending native acceptance
```sh
git show c0f73564adf712541cbce9e869d099a842a0fe9a:sparse_attn/indexer/tli_indexer.py > /tmp/swa-baseline.py
python agent_doc/results/graph-swa-fill-20261029/probe.py /tmp/swa-baseline.py sparse_attn/indexer/tli_indexer.py
python -O agent_doc/results/graph-swa-fill-20261029/probe.py /tmp/swa-baseline.py sparse_attn/indexer/tli_indexer.py
# Actual tensor fragments, unexecuted locally:
python agent_doc/results/graph-swa-fill-20261029/probe.py /tmp/swa-baseline.py sparse_attn/indexer/tli_indexer.py cuda
bash scripts/test_tli_semantics.sh --cuda
python exp/bench/test_tli_batch_gates.py
```
Full selector/output and paired eager/Graph performance still require identical parent/candidate workloads, B1/2/10/32, ragged lengths, frozen gates/EOS, ring wrap and fresh request recapture. Do not infer these pass from fragment tests. Preserve BF16 and non-BF16 dtype guard tests from PR14; no tolerance change.

26 branches/14 open PRs checked. Main, external branch heads, reviews and discussions unchanged; exact-head branch scope/ancestry evidence reused, previous PR14 addition included. PR10/13 lack the new full-width path and do not cover this fill; update PR14 rather than transplant old selector implementations. No main push/merge.
