# TL-FINE-DIRECT-OUTPUT-20261101

Base main dbbba689e19d0d1507d2005a7a4fb4227d281559; indexer blob b58d68607928411dc0f53c50cc12e3b80d8afd9a.

Problem: compute_score's full-width Triton fine branch allocates score_fine[1,Hq,S], then _fast_score_fine allocates another[Hq,S], and assignment copies the whole result. The helper already supports out= and strides. Pass score_fine[0] directly and remove the assignment. Same kernel, input, output dtype, strides, grid, arithmetic and rounding; no new kernel. Keep initial -inf allocation and segmented dgate_short, projection, candidate and einsum branches unchanged.

Static routing: paged_region_tli.region_selection requires TLI_FINE_GATHER=0 and rejects candidate scores, so its eligible non-projection full-width branch can benefit. Both-skip bypasses scoring; dgate_short takes another branch. Paged entry remains B1 eager, not a batch/Graph completion. Other callers of compute_score may use this branch; do not claim compute_mask_graph or exact batch benefits. The helper's optional out support pre-existed and is tested under capture below.

Confirmed source work removed on successful fast path: one temporary allocation and one full-output copy. For Hq32,S131072,BF16 that temporary payload is8MiB and the copy logically reads+writes16MiB. These are analytical payload counts, NOT measured GPU allocation peak, HBM traffic, latency or speedup. Initial score_fine fill is deliberately retained for segmented/fallback semantics. Destination pointer differs; alignment/allocator effects still need native measurement.

Correctness: destination view is contiguous[Hq,S], same dtype and strides as helper-owned output. _score_fine_kernel covers all t<T for each h, including tile tails. Existing except branch overwrites the full destination if a synchronous helper exception occurs. This does not promise recovery from an asynchronous CUDA device fault.

Tests: tests/test_tli_fine_direct_output.py extracts the actual production try/except block.72 host cases (4 widths,2 groups,3 dtype labels, success/before-write/partial-write exception), ordinary and -O. Baseline creates one temporary and successful-path copy; candidate creates zero temporary and zero successful-path copy. All final arrays agree with independent deterministic expected arrays;48 injected failure cases still perform full fallback overwrite. Baseline fails all72 new no-temporary assertions; candidate passes. Host dtype labels do not establish numerical dtype accuracy. An initial test extraction matched coarse-avg's use of the same helper; corrected to select the k_qat caller before collecting results.

Native CUDA test added but NOT RUN here (torch/Triton absent): exact helper output and actual caller parity for FP16/BF16/FP32, odd/tile-boundary widths, output identity, repeated capture/replay with changed K. Commands from repo root:

```bash
PYTHONPATH=. .venv/bin/python tests/test_tli_fine_direct_output.py -v
TLI_NATIVE_BENCH=1 PYTHONPATH=. .venv/bin/python tests/test_tli_fine_direct_output.py -v
```

Second command adds8K/32K/128K odd-width cases and alternating paired CUDA-event scorer+copy vs direct-output samples. This is a scorer microbenchmark, not selector/TPOT; full paged keep/near/far selected-ID/output and complete-step latency still required before a speed claim. No tolerance relaxed. For host baseline reproduction, save base-commit tli_indexer.py to a separate file, set TLI_SCORER_SOURCE to that file and TLI_EXPECT_LEGACY=1; unset legacy expectation for the regression.

Remote context: all53 branches refreshed,52 exact nonmain ancestry/date/unique-commit and source classifications reused from gqa-forced-load-audit-20261101 because heads/main unchanged. All21 PRs open/unmerged and review timelines empty. PR10 actual2449f569, opt/ako-pool-scorea8eb0f08, and PR1 actual217d99bb callers freshly read: they retain the same assignment; PR10/ako have the existing out helper. No same narrowly scoped fix identified. Independent small change based on current main, no dependency on outdated ring/linear integration; don't overwrite those branches. l2-fine-topk remains historical main ancestor, e175 old ABI/GPU caveats retained. No new experiment quality acceptance, scientific narrative or human-guide edit.

Decision: submit isolated draft for native validation. Pending prior lifetime, summary and batch compact candidates are not mixed into this fix. Local detailed paired outputs/coverage retained in agent_doc/results/fine-direct-output-20261101; remote validation summary accompanies this task.
