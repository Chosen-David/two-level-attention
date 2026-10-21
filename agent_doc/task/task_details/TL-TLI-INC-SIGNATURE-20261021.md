# TL-TLI-INC-SIGNATURE-20261021

## Scope and baseline

Base: `e6faba35342dbc21414e0212b27b68d209726b96`.
This change only guards compatibility of TLI incremental index buffers. It does not change selection probabilities, budgets, protected tokens, kernels, or model defaults. It assumes append-only K and stable indexer configuration; it does not provide arbitrary runtime reconfiguration support for projection/kmeans caches.

## Failure and fix

Before this change, increasing sequence length with a matching subspace key could reuse buffers with a different device, dtype, KV-head count, or page size. Existing buffer allocations retain their old metadata. In particular, a new single-head tail can broadcast into old multi-head buffers; dtype changes can cast the tail while retaining the old prefix.

Store `(device, dtype, Hkv, D, page_size)` after full index preparation and compare it before incremental dispatch. Missing signatures also rebuild. Sequence length, allocation identity and strides are excluded, preserving ordinary append/reallocation eligibility. This adds host metadata comparisons, not tensor reductions or device synchronization.

## Validation

A dependency-light harness executes extracted production methods; it is not a numerical tensor emulator. Before the production fix, four metadata-change cases and the missing-signature case incorrectly dispatched incrementally (five failures). The initial dependency-light suite passed 20 tests in normal Python and `-O`. A subsequent isolated CPU environment (Python 3.12, torch 2.8.0+cpu, einops 0.8.2, NumPy 2.5.4) enabled real tensor regression. The final suite passes 42 tests in each mode across:

- `tests.test_tli_runtime_shape_contract`
- `tests.test_tli_debug_contract`
- `tests.test_explicit_near_page_budget`
- `tests.test_batch_decode_context`
- `tests.test_tli_incremental_metadata`
- `sparse_attn.ops.tests.test_graph_cache_schema`
- `sparse_attn.ops.tests.test_graph_required_dispatch`
- `sparse_attn.ops.tests.test_batch_explicit_quota_guard`

Coverage includes full-preparation signature storage, matching growth, equal/shrinking lengths, empty cache, each changing metadata field, and prior public shape contracts. Independent reviewers inspected the diff and reran the host suite. The real-tensor test loads production TLIIndexer and compares index tensors against a fresh full preparation with exact equality. It covers head count increase/decrease, float32 to float64, page size changes, clear(), and valid append with same/full L2 width and quantization on/off. Replacing only the two changed methods with their baseline implementations reproduces two assertion failures (head shape/dtype) and two runtime errors (head growth/page size). The fixed tensor suite passes all five test methods.

Device changes are covered only by host metadata dispatch; CPU tensor equivalence is not a CUDA device-migration test. Graph tests use mocked CUDA APIs and do not certify capture/replay. Model quality, CUDA kernels, latency and memory have not been tested. No GPU speedup is claimed. The initial system interpreter lacked Torch; the isolated CPU installation resolved that test dependency without GPU execution or model downloads.

## Performance audit findings, not optimizations or measurements

At this base, Qwen `_tl_decode_batched` iterates requests, clears the shared indexer, and rebuilds each row. The batch-prefill selector Graph gate rejects explicit nl/nb settings and subpage sink; that gate does not describe every possible host Graph integration. E156 candidate construction includes data-dependent `nonzero`/`.item()`, then reconstructs full-width scores before normalization/selection. These are source-level work/synchronization findings, not proof of current host-run slowness.

The Graph cache clears all entries at capacity. A selective eviction policy is a separate candidate requiring lifecycle/replay testing; it is not included here. Historical B=1 stage timings cannot quantify current large-batch Graph performance. Current run configuration, capture/replay receipts and matched quality/request timings remain necessary.

## Publication

Prepared for an independent repair branch and draft PR only. Main is not a publication target for this change. GPU/Torch verification remains open.
