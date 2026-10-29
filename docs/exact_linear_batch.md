# Experimental exact linear batch route

This integration keeps per-request incremental index-feature maintenance, then
calls batched minmax/average coarse scoring, native rounded GQA mean, two L1 pool
TopKs, candidate expansion and stable compaction across B*H, fine score, exact
candidate normalization/ranking, protected/selected attention and merge. It does
not call the old slot-softmax parallel path. Row-owned incremental maintenance
still loops; this is not fully parallel cache/index maintenance.

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
initialization/refill copies full fine/summary storage. Each step's remaining row
loop updates these views in place. Bulk query projection, scores and candidate
geometry feed a BatchedCandidateInputs directly to the consumers, with no
unpack/restack of candidate outputs or full-history tensors. Raw K/V and graph input buffers are retained against allocator
address reuse. Static KV writes use the accepted `append_kv_batch` primitive.

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
  replacement storage. Retired states must remain alive while graphs reference
  them. This initial prototype retains them for the source indexer's lifetime.
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
