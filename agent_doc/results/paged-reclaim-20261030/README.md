# Paged KV reclaim audit — 2026-10-30 22:38 CST round

Base main: bc52d8677544ca390efb5375afedf5d6f4d3e3f4; prior main: 8ff7e78574a0dd8246ac1a5b677bd625939d4dc8.
Candidate branch: perf/paged-reclaim-fastpath-20261030.

## Change and actual path
Qwen/GLM paged consumer -> paged_region_tli.decode -> update_for_paged_decode -> PagedRegionLayer.update -> _reclaim. This opt-in path is eager B1, not CUDA Graph or complete-batch indexing. When skip_far=False, intervals(resident=True) cover [0, logical_length): sink, far, near (including invisible near-only staging), SWA. Scanning every page cannot remove anything. Return before constructing spans/list when far is kept. Far-skip and both-skip retain the existing reclaim algorithm, free-ID heap and pointer-table invalidation.

Existing direct-page Triton projection/attention kernels are unchanged. No new kernel, allocator, dependency, GPU launch or arithmetic. A boundary-indexed reclaim data structure would add maintenance and rollback complexity; the invariant guard removes this no-op case with two executable lines. This does not remove the full logical projection/mask/sort/synchronization costs.

## Evidence and limits
Actual-source metadata tests: 2 unittest methods, 198 subcases (192 all-gate/boundary cases plus 6 no-scan cases), ordinary and -O pass. Base main passes residency checks but fails the six no-scan cases.
Additional local actual-source NumPy tensor-facade probe: 24 configurations / 2016 decode appends / 2 injected-copy rollback cases pass for baseline and candidate in ordinary and -O modes. KV contents, copies, clones, reclamation counts match; only far-kept page visits disappear. These are NOT native Torch or CUDA tests.
The update still clones old K/V tail slots before appending to an existing page. At page=64, 192 appends produce 378 clone calls and384 copy_ calls per layer in this host trace. Existing bytewise rollback tests explicitly include unused tail slots, so backups were retained. No GPU byte/launch profiler measurements claimed.

Paired CPU timing runs actual _reclaim Python code on page metadata; no tensor facade operations occur in the timed region. Six configurations (8K/32K/128K, all-kept/near-only), 30 observations each arm, alternating ABBA/BAAB, 100 calls per observation. First run baseline medians: 8K 91–100us, 32K348–366us, 128K1656–1702us; candidate 0.08–0.11us. Raw timing attached. This is isolated CPU metadata cost, NOT GPU speedup, TPOT, end-to-end or memory measurements. Python dicts contain placeholder payloads; far-kept reclamation never accesses payloads. New CLI rerun also recorded independently.

Local environment: no torch, pytest, triton or GPU. Native tests and model/GPU acceptance pending. No quality or memory claim; no threshold changed.

## Fresh remote coverage and integration
All44 branches including main and all17 PRs read; all17 review/discussion timelines empty. Branch heads/dates/unique commits/ancestry in ancestry.json. Other branch heads unchanged; actual-code evidence reused by exact SHA. main advanced one commit; all branch ancestry recomputed. No inferred active developer processes.
PR10 remains2449f56983d7e97c5fe1c4df300c2b50b0c16b81 (31ahead204behind); all362 changed paths reviewed for overlap, no paged files. PR12/13/16/17 file inventories also have no paged changes. No same-topic open PR; no unmerged dependency.
opt/l2-fine-topk56d503a is main ancestor. opt/ako-pool-score a8eb0f0 (2ahead214behind) and e175-skip-ring-kv9b6e792 (3ahead231behind) retain prior unmerged/validation status. paged1b7dbd6, incremental0516c3e, GLMe5d0fd9 are historical main ancestors. physicale55349f remains12ahead28behind; aggregate verdict/raw availability limitations unchanged.
New main bc52d86 explicitly propagates device revisions to Qwen and hybrid row views and admits managed-capture/inference-device geometry reuse; 472 actual cache-helper row/step checks plus capture isolation controls pass on host facades. This covers PR17's actual caller failure mechanism; PR17 remains open/unmerged and its old tuple patch must not be transplanted over new helper semantics. Native Torch/CUDA validation still pending. FIFO capacity8 still does not guarantee reuse at B>8, already a recorded candidate.
Current hybrid_batch_cache.py still loops over rows for indexer and attention; exact rejects mixed ring. This is a fresh code check, not an assumption from history.

## Reproduction / pending acceptance
```sh
python tests/test_paged_reclaim_host.py
python -O tests/test_paged_reclaim_host.py
git show bc52d8677544ca390efb5375afedf5d6f4d3e3f4:sparse_attn/ops/paged_region_cache.py > /tmp/tli-paged-reclaim-baseline.py
python exp/bench/bench_paged_reclaim_metadata.py --baseline /tmp/tli-paged-reclaim-baseline.py
python agent_doc/results/paged-reclaim-20261030/probe_append_host.py
TLI_PAGED_RECLAIM_SOURCE=/tmp/tli-paged-reclaim-baseline.py python agent_doc/results/paged-reclaim-20261030/probe_append_host.py
.venv/bin/python -m pytest -q tests/test_paged_region_cache.py tests/test_paged_region_tli.py
.venv/bin/python exp/bench/check_paged_region_kernels.py --out /tmp/paged-kernels.json
.venv/bin/python exp/bench/check_paged_region_same_input.py --model "$TLI_MODEL_PATH" --length 32768 --steps 129 --modes keep near far both --out /tmp/paged-same-input.json
.venv/bin/python exp/bench/check_paged_region_model_correctness.py --model "$TLI_MODEL_PATH" --length 32768 --steps 129 --out /tmp/paged-model.json
```
Set TLI_MODEL_PATH to the same local model checkpoint for both revisions. On a GPU, run the existing paged model/same-input/kernel checks on baseline and candidate in isolated checkouts with identical model/config/seed, all four frozen gate combinations, long decode crossing repeated page boundaries and new requests. Record selected IDs/output at existing tolerances, whole-request wall/TPOT and allocator allocated/reserved/peak separately. Do not infer GPU or model gains from this host benchmark. Graph/batch paths are unchanged and not claimed as accelerated.

