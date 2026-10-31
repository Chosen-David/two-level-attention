# TL-PAGED-POOL-REPORT-20261031

## Problem and change
Main dbbba689e19d0d1507d2005a7a4fb4227d281559 cache blob e2c393317a4490838707662c1df20cd857f4a969: PagedRegionCache.report sums layer.report() legacy B1 pages. Batch write_request populates kv_pool/block_tables, not pages, so active_request_ids can coexist with kv_bytes=0. Physical pointer buffers _k_ptr/_v_ptr are omitted too. This is owned-tensor accounting, not a driver/allocator measurement.

Add layer.report_pool(), a single scan of independently owned physical pages plus currently held logical and physical pointer tensors. Global report totals come from these pools; retain legacy layers detail and per-request report API, expose pools explicitly. Legacy pages also occur in kv_pool and are counted once. No KV mutation, synchronization or new GPU kernel. CPU metadata traversal O(total resident pages + layers); per-request report's existing pool-membership scan is a separate deferred optimization. No inference-speed claim.

## Evidence
Source-bound actual class methods with NumPy tensor facade:18 configurations (page4/16/64 × batch1/4/32 × layers1/3),126 checks. Baseline48 failures; patched0 failures, ordinary and -O. Checks include batch bytes, physical-pointer bytes, removal/empty pools, legacy no-double-count and logical pointers. These are host accounting/lifecycle checks, not native Torch/CUDA evidence. Added native regression covers CPU and CUDA when available, unequal K/V widths, request-slot reuse and legacy path. No tolerances changed.

Local files: agent_doc/results/pool-report-audit-20261031/{baseline.py,fixed.py,probe.py,before.json,after.json,after-optimized.json,native_test.py,ancestry.json}. Local audit artifacts are retained separately; reproduction of native regression requires only this PR.

Native validation pending (local Torch/Triton/pytest/GPU absent):
```sh
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_paged_region_batch_cache.py tests/test_paged_region_cache.py
```
No measured latency, allocator peak/reserved memory, quality or GPU speed result. A corrected byte report must not be presented as a measured memory saving.

## Integration and coverage
Current CLAUDE83dbec/TASK3b617c and previous coverage read.52 branches refreshed; all51 nonmain exact main+head pairs unchanged, so unique commits/dates/relations and actual-diff/evidence classifications reused. Priority l2-fine-topk historical ancestor; ako-pool-score/e175 candidates retain existing overlap/conflict/GPU gates.20 PRs all open with unchanged heads,20 fresh review/discussion timelines empty; no new committed experimental evidence or inferred live jobs. PR10 head2449f56983d7e97c5fe1c4df300c2b50b0c16b81 does not cover this new paged pool report. PR19 import repair, PR20 duplicate-write guard independent; no dependency on unmerged PRs. Isolated branch based on main, no main push or self-merge. No human guide or manuscript edits.

Production route remains Qwen B1 eager; this accounting repair does not add batch attention or Graph support. Current paged direct consumer still uses legacy pointer_tables, not the batch physical table; no historical hybrid-loop assumption. No external kernel needed: scalar tensor metadata is host-readable and pool ownership supplies the authoritative tensor set.

Next: native test on actual dependency stack; inspect layer-wide freeze failure consistency separately. Scientific performance acceptance remains pending; this is an engineering accounting fix.
