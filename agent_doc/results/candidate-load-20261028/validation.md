# Candidate-prefix loads — 2026-10-28

Main8f213e4, parent PR10 3404da6. Root CLAUDE83dbec28, TASK/current narrative and prior coverage reviewed. All12 PR heads and concurrent branches unchanged. No nested rules in unchanged main. Kernel unchanged in PR12 and concurrent L2 branch; their downstream softmax work is not imported. Isolated local branch codex/candidate-valid-load-20261028; only allowlisted delta published.

Main _gather_fine_cand and compute_mask_graph both invoke _fast_score_fine_cand. CLEN already defines the valid per-head prefix: eager builder places L1+shared tokens there; graph uses stable compaction. Old kernel read CAND and K for all R lanes, then overwrote invalid scores with -inf. Move CLEN load earlier and predicate both loads with r<R && r<CLEN. Keep output stores, query loads, reduction order, dtype, launch geometry and all fallback/mask/softmax/topk code unchanged. Scope includes eligible eager and CUDA Graph indexer paths, including skip/non-skip when routed here; compact ring workspace has a different scorer and is not changed.

No new kernel or dependency. This reuses the existing kernel contract. Masked lanes still execute arithmetic with zero K and still store -inf. It does NOT remove launches, all inactive arithmetic, EOS work, or output buffers. Invalid sentinel reads may hit cache: reduced predicated lanes are not measured DRAM bytes or a speedup. All-valid extra predicate overhead and register/codegen behavior require GPU measurement; reject this delta if target workload regresses.

Validation:48 exact frozen-old/new Triton3.4 CPU interpreter pairs, fp32/fp16, D32/128, G1/4, R65/129/4224, token/head-major layouts, three heads with CLEN0/half/full. Same outputs at rtol0/atol0 and invalid outputs negative infinity. pairs.json is raw scoped evidence.49 CPU regressions passed,12 CUDA skipped (batch parallel, runtime shape contract, hybrid batch cache). E127 indexer and batch gates attempted, unavailable on CPU Torch; see gpu-gates.txt. No real GPU/BF16/capture/allocator/performance acceptance. Prior PR evidence is not inherited as GPU acceptance for this changed kernel.

Remote reproduction (from repository root with Triton/CUDA):
```
python agent_doc/results/candidate-load-20261028/probe.py --bench
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
OMP_NUM_THREADS=1 python -m pytest -q tests/test_tli_batch_parallel.py tests/test_tli_runtime_shape_contract.py tests/test_hybrid_batch_cache.py
```
Probe pairs frozen parent kernel and current kernel with exact outputs, then reports eager and CUDA Graph microbench timings; includes BF16 on CUDA. Also require model-level eager/Graph B1/2/8/32, ragged lengths, skip mixtures, EOS, repeated wrap and fresh capture. Add all-valid-only and empty-heavy production distributions to acceptance and run interleaved repeats; the mixed-head microbench alone is insufficient. No GPU latency, memory or quality improvement asserted. No main push or merge.
