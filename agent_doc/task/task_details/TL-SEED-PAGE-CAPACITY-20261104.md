# TL-SEED-PAGE-CAPACITY-20261104
Status: implementation and host regression complete; native CPU/CUDA, Graph and model acceptance pending.

Baseline: 40756b6c925ad0caa5741ccbc305299fb8189567.
Route: prefill_frozen_request -> cache.freeze -> _max_resident_pages_for_layer / _resident_page_ranges_for_layer -> _batch_reclaim -> deficit check; GraphRunner marks seeds consumed after all layers, then commit_step releases seed allowance with the existing CUDA event.

Problem: the seed allowance filters absolute token IDs by resident_tokens (4228 for default far-skip), while residency correctly filters by logical_length. At prompt length32768,64 seed tokens8192..12224 spaced64 occupy64 distant pages. Far-skip/near-keep residency is131 pages; baseline budgets69 and freeze can raise resident page accounting is inconsistent. Fix budgets133 (steady bound69+64), then returns to69 after consumption. The bound deliberately double-counts overlap, preserving existing conservative allocation policy.

Change: align seed validity with logical_length. No selector, gate, score, Graph pointers, event ordering or steady bound changed. This is correctness, not a measured speedup or VRAM saving. Additional seed slots are necessary temporary residency, not a regression to full-history retention. Existing set-of-page counting is already O(seed_count); no new kernel/library needed. Do not optimize repeated steady commit bookkeeping in this patch.

Host evidence: actual bound method AST executed; explicit32K counterexample plus96 page-set configurations across lengths7/257/32768/65537,page16/64/128,four skip states,first/consumed,duplicates and invalid IDs. Candidate two host tests pass normal/-O; baseline both fail. Baseline also admits some beyond-length IDs for non-far-skip due to growth reserve; aligned filter rejects them. Python compile passes. No Torch/Triton here: two native tests skipped. Tests include optional real-cache freeze/KV-preservation/release/slot-reuse on CPU and CUDA; they do not run attention or Graph replay.

Reproduce from candidate checkout:
```bash
PYTHONPATH=. python tests/test_seed_page_capacity.py
PYTHONPATH=. python -O tests/test_seed_page_capacity.py
git show 40756b6c925ad0caa5741ccbc305299fb8189567:sparse_attn/cache/paged_region_cache.py > /tmp/nfa-seed-before.py
PYTHONPATH=. NFA_CACHE_SOURCE=/tmp/nfa-seed-before.py python tests/test_seed_page_capacity.py
PYTHONPATH=. compute-sanitizer --tool memcheck python tests/test_seed_page_capacity.py
PYTHONPATH=. python -m pytest -q tests/test_frozen_region_runtime.py tests/test_multi_request_graph_runner.py
```
Baseline command is expected to fail. On GPU verify native tests are executed, not skipped. Required follow-up: paired real model prefill/freeze/first-decode/steady-decode for mixed gates, ragged B1/4/32, EOS and repeated slot reuse/new capture; exact seed support and KV values; event-safe release; no thresholds relaxed. Measure full-prefill wall time and allocated/reserved/peak separately; baseline failing freeze is a correctness failure, not a denominator for speedup.

Overlap: PR31..36 retain wrong seed filter; PR27..30 predate this seed allowance (PR27 independently changes near-only residency/both-skip bound). No dependency on PR27 or PR36. PR10 and priority development heads unchanged old ABI/history; no transplant. Apply future bound refactors while preserving this absolute-position filter and seed release allowance. All70 existing heads and36PRs checked unchanged, reviews/comment increments empty; exact branch classifications reused. No main push/merge or guide/manuscript change.
