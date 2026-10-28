# Main fusion regression repair — 2026-10-28
Input main fb335e5ceb742b31d83206c670a10286c7fba182 (merge opt/l2-fine-topk); PR10 baseline32e68114a11053a8d867497370417194920b782b. Only main delta since8f213e4 is tli_indexer.py. Three-way merge of that file with PR10 completed without conflicts; previous candidate-load, subspace and geometry work retained. This commit has both PR10 and latest main as parents. No main push or PR merge.

## Mechanisms and scope
1. Main now dispatches the previously audited tiled softmax by default. Each1024-slot program normalizes independently; R2048 sums~2 and R4224 sums5, and selected top1 differs. The exact merged kernel blob950c53ed27a884f0cc220a82226c64440674e202 matches the prior concurrent branch. Restore reference Torch softmax/nan_to_num/dtype-rounded GQA mean by having its wrapper return None. This removes the invalid path, not an accepted acceleration. The dormant kernel is retained for subsequent isolated redevelopment; it must not be enabled until global per-query normalization, rounding and quality are validated.
2. New compact sentinel fusion writes sanitized IDs but reads MIDM from raw inactive IDs first. Normalize idx before MIDM load, preserving stable tail and MID output. 48 interpreter pairs: fixed all exact vs stable argsort reference; original differs on MID output in36 cases with inactive slots. Probe pads MID backing to expose logical out-of-domain reads safely; this is not an actual device OOB experiment.
3. New TLI_L2_BYPASS branch runs before compact and lost its sanitizer. Restore where(keep,ids,S_cap-1) before scatter. Four actual bypass-body CPU cases on baseline:2 incorrectly include inactive in-range token9;2 reject inactive out-of-range999. Fixed cases all exact vs expected protection+valid-candidate set. This optional branch changes budget semantics; no quality claim.
Keep main's direct score outputs, non-power-of-two L1 support, compact fusion and mask changes. Do not replace the whole newly merged implementation with historical code.

## Validation
- CPU integration:62 passed,10 CUDA skipped (hybrid batch/cache, geometry lifecycle, shape contracts,16 new guard cases).
- Final guard file after adding native CUDA tests:16 passed,3 CUDA skipped. These16 overlap the62; do not sum them.
- Compact:48 CPU Triton interpreter exact pairs. Explicit NumPy2.5 interpreter loop-index compatibility shim, unchanged kernel executable AST apart from intended repair.
- Main softmax counterexamples: R1024/2048/4224, G4;2 failing normalization/top1 cases.
- First CPU collection attempt lacked exp namespace in isolated snapshot; rerun with existing PR10 snapshot on secondary PYTHONPATH passed. No code workaround.
- No GPU available. CUDA correctness/capture, native memory safety, latency, peak memory and end-to-end quality remain pending. Disabling wrong fusion can cost performance versus invalid main; no speedup claimed.

Reproduce on checkout:
```sh
PYTHONPATH=. python -m pytest -q tests/test_main_fusion_guards.py tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py tests/test_graph_geometry_request_cache.py tests/test_tli_runtime_shape_contract.py
TRITON_INTERPRET=1 python agent_doc/results/main-fusion-fix-20261028/probe_compact.py
TRITON_INTERPRET=1 python agent_doc/results/main-fusion-fix-20261028/probe_softmax.py
PYTHONPATH=. python exp/bench/test_e127_gates.py indexer
PYTHONPATH=. python exp/bench/test_tli_batch_gates.py
```
Run pytest on CUDA for the3 new native compact eager/capture checks; run under compute-sanitizer for actual device memory access. Real model eager/Graph paired quality and latency must use identical champion flags and input/gate sets. Prior gate quality does not accept compact-selector semantics.

## Branch coverage
All24 heads refreshed.12 PRs remain open and unchanged; no new issue/inline comments or reviews. Main alone advanced; opt/l2-fine-topk is now a main ancestor, no longer an unmerged candidate. Remaining no-PR branch identities and development mapping reuse branch-audit-20261028; old ancestor relations remain valid because main advanced by merge. opt/ako-pool-score constructor error remains isolated there. No new experimental-result files in this main delta.

