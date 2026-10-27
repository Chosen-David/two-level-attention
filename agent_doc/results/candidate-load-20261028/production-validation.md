# Production tile / codegen follow-up, 2026-10-28

Main `8f213e48926431ea2d56cf3e7bc1a667d1105c74`; PR10 parent `89aaaa7f541a5838051ce6f6a45feb594f167806`. Main indexer blob `69eb1641deaebb96cdc8f17de6b5789e914088fe`, PR indexer blob `47a02e188471190197c80f2fbf5f0964119201ec` matched local files. Re-read root CLAUDE, TASK, narrative, candidate evidence and prior coverage. All 12 PR heads, main, concurrent branches, reviews/discussion and main experiment tree unchanged. No production code changes in this follow-up.

## Actual main path and coverage correction

- Main `_TILE_FINE` at line 2327 is D32 `(16,1)`, D64/128 `(64,4)`. `_fast_score_fine_cand` at 2382 reads that table. The first probe fixed `(64,4)` for every D: its 48 exact pairs remain valid for those explicit launches, but did **not** verify the production D32 tile. Do not relabel them as production-wrapper evidence.
- `_gather_fine_cand` at 1847 and `compute_mask_graph` at 4399 call the same wrapper. Graph's stable compaction precedes the scorer, so CLEN is a valid prefix, not just a count over arbitrary holes. Eager already omits redundant post-Triton masking; Graph still masks after scoring. That masking/fusion candidate remains deferred against overlapping `opt/l2-fine-topk@56d503a` and its documented softmax error. PR12's unchanged head does not resolve this tile validation gap.
- Main mixed ring/linear decode still loops through `layer.rows` in `hybrid_batch_cache.py:192`. The linear row calls this scorer through graph selection; ring workspace uses its separate implementation. This is not complete-batch kernel evidence, EOS pruning, or a new batch-parallel implementation.
- Compared the frozen baseline's executable AST against main (excluding indentation-only docstring differences): identical. The production kernel is unchanged from PR10 `89aaaa7` throughout these tests.

## Executed evidence

`production_probe.py` calls the **actual wrapper** for both frozen baseline and candidate, temporarily substituting only its kernel binding, restored in `finally`. It exercises table lookup, output allocation, strided inputs and real launch arguments. **216 CPU Triton-interpreter pairs are exact at rtol=0/atol=0**, inactive outputs all `-inf`: fp32/fp16; D32/64/128; G1/4; token/head-major K; nontrivial Q row stride; R=BT+1,2BT+1,4224; empty/mixed/full candidate occupancy across Hkv=3. Raw records and source/harness SHA256 bindings are in `production-pairs.json`. This is wrapper/kernel correctness coverage, not execution through a whole Qwen forward or actual capture.

`compile_production.py`: **72 offline compilations** (36 old/new pairs), SM80/SM90, fp32/fp16/bf16, D32/64/128, G1/4, real tile/warp table. Uses dynamic stride arguments **without runtime alignment hints**, so resource counts are scoped generic specializations, not guaranteed counts for a captured production workload. Default compiler fusion retained. `production-compile.json` records raw ptxas logs, PTX hashes and source identities. All 72 compiled, zero spill stores/loads and zero stack frame; shared allocation unchanged per pair. Registers increase in 9/36 pairs, decrease in 22, unchanged in 5. For example, generic SM90 D64/G1 fp16/bf16 rises 115 to 128. Do not convert these counts into occupancy or latency claims.

Representative regenerated SM90/bf16/D32/G4 PTX: CLEN feeds comparisons combined with R; CAND `ld.global.b64` and K `ld.global.b16` use those predicates. Four Q `ld.global.b32` loads remain unconditional, reductions remain, and OUT still writes `-inf` lanes. **Static predicated instruction counts are not dynamic transactions or DRAM bytes.** The initial standalone ptxas invocation used `sm_90` against generated `sm_90a` and failed; fixed by reading `.target` from the emitted PTX, then reran all 72 successfully.

`--bench` was attempted here and correctly rejected CPU-only execution. No GPU timing, BF16 numerical execution, CUDA Graph replay, end-to-end correctness or physical memory measurement was performed. Prior 49 CPU tests/12 CUDA skips remain historical scoped results; no production code changed in this follow-up.

## Candidate disposition / next scope

The load predicate is already in PR10, so do not duplicate it. Fully inactive tiles still load Q and run the GQA reductions. A program-uniform device branch could skip that work while writing `-inf`; it preserves fixed grid/address ownership, but adds branch/control overhead and does not eliminate output stores or launches. Defer that additional kernel variant until target-GPU all-valid versus empty-heavy measurements justify it. Existing fused scorer is the appropriate baseline; a new gather tensor or external attention library would add materialization/change the selector contract. No new dependency or production mutation is warranted by these offline results alone.

## Remote acceptance

From repository root, with dependencies installed:
```bash
PYTHONPATH=. TRITON_INTERPRET=1 python agent_doc/results/candidate-load-20261028/production_probe.py
PYTHONPATH=. python agent_doc/results/candidate-load-20261028/compile_production.py
PYTHONPATH=. python agent_doc/results/candidate-load-20261028/production_probe.py --bench --repeats 6 --out /tmp/candidate-production-gpu.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

The new CUDA microbenchmark includes BF16, empty/mixed/full occupancy, T=32768, production tiles, exact output checks, preallocated outputs and alternating old/new order per repeat for eager and Graph timing. It replaces the original fixed-tile, one-order microbenchmark as the acceptance starting point. It does not replace production Hkv/batch/length distributions, selector/top-k checks, Nsight traffic/launch measurements or model-level B1/2/8/32, ragged, gate, EOS, repeated wrap and new-request capture gates. Reject/regate the load-predicate change on relevant GPU regressions. No speedup, memory saving or quality improvement claimed.
