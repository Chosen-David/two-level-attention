# Experimental exact linear batch route

This integration calls batched production tail32 incremental feature/summary
maintenance, then batched minmax/average coarse scoring, native rounded GQA mean, two L1 pool
TopKs, candidate expansion and stable compaction across B*H, fine score, exact
candidate normalization/ranking, protected/selected attention and merge. It does
not call the old slot-softmax parallel path. Production BF16 D128, tail32, block64, same-subspace/no-quant maintenance uses
one B*H kernel. Other previously allowed maintenance configurations keep an
explicit row fallback. Initialization and refill still use row-owned host loops.

Explicit configuration:

```
TLI_BATCH_INDEXER=exact
TLI_OPERATOR_MODE=serial
TLI_SPARSE_SELECTION=1
TLI_GRAPH_INC=1
TLI_OPERATOR_BACKEND=simt
TLI_OPERATOR_SELECTED_STAGES=1
```

Scope is B>=2 fixed-address graph-state inputs, linear shared K/V, BF16 D128,
no skip/dynamic gate/projection, fixed production near geometry, K2=1024,
R>=1024, pinned Torch 2.8 sparse-normalization capacity/alignment and ranking
capacity<65536. Unsupported configurations raise rather than produce a legacy
fallback labelled exact. Hybrid ring storage and overlap are rejected.

Each source indexer owns independent row indexers. Outside capture, one warmup
binds their persistent `_ginc['k_qat']` to views of a `[B,S,Hkv,Dfine]` arena.
Warmup also binds k_min/k_max/k_avg to persistent summary arenas. Only
initialization/refill copies full fine/summary storage. Warmup canonicalizes each current block with the primary single-row kernel;
steady production steps update these views with one batched kernel and skip all
writes for dead rows. The min/max/FP32 mean reduction and final storage-block
zero padding remain the primary kernel's operations. Bulk query projection, scores and candidate
geometry feed a BatchedCandidateInputs directly to the consumers, with no
unpack/restack of candidate outputs or full-history tensors. Raw K/V and graph input buffers are retained against allocator
address reuse by the current state and each captured Graph using that state. Static KV writes use the accepted `append_kv_batch` primitive.

Lifecycle:

* Every row scope restores the shared graph state in `finally`, including errors.
* Dead rows retain their valid length and cannot reactivate in the same session.
  Lengths can repeat during warmup or advance by one for live rows.
* Request refill calls `indexer.clear()`, which invalidates row summaries and
  clears a device ready flag retained by captured graphs. Warmup is mandatory
  before replay; warmup cannot occur inside capture.
* With identical storage/config, warmup restores the same arena/summary addresses;
  the existing graph may be reused after that warmup. The ready flag does not
  permanently invalidate every old graph in this same-storage case.
* Storage/config replacement requires clear, warmup and recapture. Old state is
  retained with a false ready flag, so its stale graph cannot silently consume
  replacement storage. Each captured Graph retains its address-owning state;
  obsolete states and KV can be released when their last Graph is deleted.
* Cache reorder is unsupported, matching TLIStaticLayer. Raw external in-place
  permutations are outside the fixed-owner API and must not bypass its lifecycle.

The batch producer retains primary pool clipping, bypass-page deduplication,
sentinel redirection and stable prefix/tail order. It uses the primary Torch mean
of model-rounded per-query scores, not the optional fused sequential mean from
coarse kernels. Batch geometry uses mutation-version/capture-token/stream keys
with a separate namespace, never full token-domain masks. GPU TopK tie equivalence
and complete new-producer capture behavior remain native validation gates.

CPU tests cover tensor geometry/selection vs individual primary producers,
random/tie/nonfinite score pools, source-method combinations, empty/short pools,
metadata rejection, state isolation/restoration, fake-owner arena
reuse/clear/replacement and one-call-per-batch consumer scheduling. They do not
establish native numerical equivalence, capture safety or performance. GPU tests
must compare each row with the single-request reference, cross page boundaries,
EOS/dead rows, same-storage clear+warmup with old/new graphs and changed-storage
retirement. Negative device-assert tests need isolated processes. No speedup or
full-batch completion is claimed until native/model tests and independent review.

The source indexer exposes `_exact_batch_maintenance_route` (`batched_tail32` or
`row_fallback:*`) and `_exact_batch_maintenance_calls`. Counters record Python
invocations, including capture, rather than CUDA Graph replays. The batch kernel
uses 64-bit batch/head address products. Immutable feature indices are checked
at warmup; the exact owner checks dynamic positions before maintenance. Native
maintenance and full-entry tests are pending execution for this revision; CPU
structure/schema tests do not establish precision or performance.

Exact capture must use `capture_graph()` from `ops.cuda_graph_state`; raw
`torch.cuda.graph` is rejected for this route. The managed scope pins each
address-owning state on the Graph itself. Replacing storage after clear leaves
an old live Graph holding its old false ready flag and KV/feature/summary
buffers. Deleting that Graph releases the obsolete state; the indexer keeps no
permanent retired-state list. Same-storage clear/warmup still reuses the same
state, so the old Graph remains valid after successful warmup. CPU weakref/GC
contracts verify ownership and release; native lifecycle coverage remains queued.

Graph-quality integration preserves the complete denominator domain: selected
L1/bypass candidates first remove protected duplicates, then append every live
sink/SWA token before stable compaction and fine scoring. This matches both the
single-row Graph producer and eager reference; adding protection only at final
attention is insufficient.

Normalization uses each request's device live length, including its native
softmax alignment/reduction order. Query-head blocks map back to their request;
flattening B*H never shares denominators or GQA groups. Initial live lengths
must all be at least 9217 and no greater than capacity; admission checks these
outside capture before allocating index arenas. Replay retains the existing
monotonic lengths/dead-row constraints. Explicit disabled live normalization or
non-owner statistics are rejected. Exact mode enables live/owner defaults for
its single-row references too; other low-level legacy defaults are unchanged.

This remains explicit no-skip Qwen batch configuration, not the dynamic-gate
configuration used by the ring experiment. It does not change the paper budget
or claim dynamic GLM B>1 support. Capture receipts identify actual batch
maintenance, complete domain, live owner normalization, ranking and attention.
New native validation is required for this integration; previous branch results
are not evidence for this revised normalization domain.

For attribution diagnostics only, `TLI_EXACT_BATCH_REFERENCE=1` keeps the model
batch shape but calls the established single-row indexer and attention within
independent row scopes. It shares admission and Graph address ownership with the
exact route, but does not execute the batched candidate/scoring/maintenance path.
Its receipt is `linear_batch_reference_simt`. The mode is part of the storage
configuration signature, so changing it requires clear and recapture. Agreement
with this control cannot by itself resolve a difference from independent B1 model
requests; model batch shape also changes other arithmetic paths.

The default `TLI_EXACT_ATTENTION=ordered` consumer forms the sorted unique
union of selected, sink and SWA IDs and invokes the original eager attention
reduction (BW256, four warps, unchanged BF16 multiplication rounding). Its
receipt is `exact_batch_ordered_eager`. The split consumer remains available only through explicit `TLI_EXACT_ATTENTION=split`
for diagnostics; its observed B4 token drift in run 0750 makes it unsuitable
as a quality reference. Ordered same-batch parity has scoped real B2/B4/B8
evidence (0680/0790/0970); this does not establish whole-benchmark quality.
The fused batch route itself still requires `TLI_BATCH_INDEXER=exact`, serial
SIMT, and supported no-gate Qwen geometry. Unsupported requests raise; GLM
dynamic gating at B>1 remains unsupported.

The real batch harness supports `--eager-batch-reference` for same-shape
correctness, and `--legacy-batch-reference` for native Graph versus fused Graph
comparisons. `--cycles` alternates reference/candidate execution order and
records each pair's request indices. B1 uses the native Graph with no batch
fusion. Timing includes prefill, cache conversion and capture in total request
time; the first pair may contain compilation/setup costs and must be reported
separately from later pairs, with all raw measurements retained.

For sequential unpadded prefills, the harness and RULER fixed-cache batch entry
retain only completed B1 caches
until all rows finish. `consume_prefill_rows` then allocates each static layer,
copies every row and releases that source layer before allocating the next.
It preserves KV values and zero padding and consumes the input caches. It is
an inference, same-stream transfer; sources cannot be reused afterward, and
an allocation failure propagates instead of returning a partial cache. This
avoids an entire batch allocation beside subsequent prefill activations. It
does not guarantee B8 fits: run 0850 failed in prefill and run 0900 failed
during conversion under the default allocator. Run 0970 independently accepts
the original eight Qwen inputs with native `expandable_segments:True`: same-B
eager and ordered Graph produced identical 128 tokens. That scoped allocator
configuration is required for the current B8 matrix; it is not a general memory
guarantee or a full benchmark quality result. The RULER entry integration has
CPU call-path coverage; its native GPU integration check remains pending.
