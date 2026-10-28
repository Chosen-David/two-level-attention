# Request geometry cache — 2026-10-27

Main8f213e48926431ea2d56cf3e7bc1a667d1105c74; PR10 parent19e69bcaf7324a283e3885809785f77a237a7d1b.
Applicable CLAUDE, human guide (read only), TASK and prior coverage reviewed.
Main and PR10 both use the same nine-entry global geometry cache policy.
Concurrent opt/l2-fine-topk56d503a changes L2 selection, not this cache policy;
do not transplant its unmerged work. Latest main/PR refs rechecked before push.

## Main finding and route

HybridBatchRunner.scope -> batch_decode_scope(ctx) -> try_hybrid_batch_decode
row loop -> compute_mask_graph -> _graph_geo_lookup/store. Geometry includes
capacity-width mid/SWA masks and page-pool masks, independent of layer for equal
geometry keys. Store clears the whole global dictionary when its length exceeds8.
At B>9 non-skip rows, the next layer reaches rows already evicted: no cross-layer
reuse. Actual-main two-layer tiny-Qwen probe at B1/2/8/9/10/32, two steps each,
shows layer1 builds0 at B<=9, but10/32 at B10/32, in addition to layer0's B builds.
Baseline probe/receipt are included; execute it on main8f213e4, not the candidate.

Change: store geometry on the current BatchCtx, bounded by max(9,B); only evict
when adding a new key to a full dictionary. Keep no-context fallback bounded9.
Version and geometry-key validation remain intact. Inference tensors without a
version counter still bypass caching. Different contexts cannot retain/reuse one
another's geometry. No scoring/reduction arithmetic or kernel changed; existing
Torch geometry and GPU consumers are reused. Request ownership avoids leaving
finished batch geometry under a global owner.

Tradeoff: B>9 retains up to B geometry entries instead of9; live cached mask
storage scales O(B*S_cap), avoiding their repeated rebuilds across layers. This
is not a blanket memory reduction claim. No new full-history KV is allocated.
The cache releases with its request owner; real CUDA private-pool/allocator peak
and latency still require validation. An unbounded global cache was rejected.

## Evidence and limits

25 CPU tests passed,9 CUDA skipped in91.32s (cpu.txt), Torch2.8.0+cpu,
Transformers4.56.2. Twelve new full-model pairs cover B1/2/8/9/10/32 × fp32/bf16,
two fresh request configurations (all linear, then mixed ring/linear),35steps each,
ragged lengths, EOS, multiple ring wraps. All840 paired logits comparisons are
exact (rtol=0/atol=0). In all-linear B32 two-layer case builds64->32 per step;
B<=9 remainsB. Candidate geometry-owner weakrefs are dead after runner/context
release while the model remains alive. Additional test checks nested contexts,
version changes, geometry keys, bounded insertion and inference-mode bypass.
Adjacent batch-indexer regression:31 CPU passed,6 CUDA skipped in3.55s;
combined56 passed/15 skipped. Three CUDA paired whole-model capture/replay/recapture tests are present but
not run. Actual GPU speed/memory and real-model quality are unmeasured.

The scope is eager/hybrid warmup sharing. compute_mask_graph still unconditionally
rebuilds geometry during capture (despite its comment describing per-layer reuse),
so this patch does NOT eliminate duplicated captured geometry kernels. Safely
sharing captured geometry needs a capture identity/lifecycle contract; reusing
warmup geometry without recording its producers is incorrect. This remains a
separate GPU-gated candidate. Do not claim CUDA Graph speedup for this patch.

Existing E127/indexer and batch GPU gates were attempted but unavailable: CPU
Torch cannot allocate CUDA tensors; batch gate explicitly requires CUDA. E127
also reports the partial local snapshot's missing optional layer-mask fixture.
These are NOT passing gates. CUDA benchmark explicitly refuses CPU fallback.
See unavailable.txt. CPU tests use deterministic dynamic gates from the existing
tiny-model fixture and do not substitute for RULER strict-selector quality.

## Other main hotspots reviewed (not changed)

- _prepare_index_base pads/copies full-D K and gathers coarse dimensions before
  _stat_full_kernel, which already accepts dimension IDs and masked zero loads.
  Direct-source cold initialization is a candidate; preserve packed-cu/padding,
  projection, MoBA and numerical contracts before integration. Startup work is
  distinct from steady incremental decode; do not claim per-token savings yet.
- HybridBatchRunner.inject still creates/copies small token/length/alive device
  tensors outside replay. Packed persistent input staging needs measured host/H2D
  overhead and careful dtype/address/host-buffer reuse handling.
- Native TLIStaticLayer.update has clamp+2gather+2where+2scatter. Standard
  compact register_patch patches ALL Qwen attention layers, bypassing this native
  layer update via hybrid rows; do not multiply this fallback cost by all layers.
  No GPU kernel launch counts or model time were measured for this observation.

## Remote reproduction

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_graph_geometry_request_cache.py tests/test_hybrid_batch_cache.py tests/test_tli_batch_parallel.py
python -m exp.bench.geometry_cache_compare --mode both --batches 2 8 10 32 --steps 40 --repeats 5 --out /tmp/geometry-paired.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

The paired CUDA driver alternates old/new arms with identical seeded tiny Qwen
models and EOS inputs, records raw event/wall time for inject+forward/replay,
checks exact logits, and binds indexer SHA256. Setup is excluded; it is not a
production-model TPOT claim. Run full production eager/Graph end-to-end and
allocator measurements separately before promotion. Only PR branch is pushed;
remote team reviews and merges.
