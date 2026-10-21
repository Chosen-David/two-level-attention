# TL-TLI-INC-SIGNATURE-20261021

## Scope and baseline

Base: `e6faba35342dbc21414e0212b27b68d209726b96`.
This change only guards compatibility of TLI incremental index buffers. It does not change selection probabilities, budgets, protected tokens, kernels, or model defaults. It assumes append-only K and stable indexer configuration; it does not provide arbitrary runtime reconfiguration support for projection/kmeans caches.

## Failure and fix

Before this change, increasing sequence length with a matching subspace key could reuse buffers with a different device, dtype, KV-head count, or page size. Existing buffer allocations retain their old metadata. In particular, a new single-head tail can broadcast into old multi-head buffers; dtype changes can cast the tail while retaining the old prefix.

Store `(device, dtype, Hkv, D, page_size)` after full index preparation and compare it before incremental dispatch. Missing signatures also rebuild. Sequence length, allocation identity and strides are excluded, preserving ordinary append/reallocation eligibility. This adds host metadata comparisons, not tensor reductions or device synchronization.

## Validation

A dependency-light harness executes extracted production methods; it is not a numerical tensor emulator. Before the production fix, four metadata-change cases and the missing-signature case incorrectly dispatched incrementally (five failures). The initial dependency-light suite passed 20 tests in normal Python and `-O`. A subsequent isolated CPU environment (Python 3.12, torch 2.8.0+cpu, einops 0.8.2, NumPy 2.5.4) enabled real tensor regression. The first-commit suite passed 42 tests in each mode across:

- `tests.test_tli_runtime_shape_contract`
- `tests.test_tli_debug_contract`
- `tests.test_explicit_near_page_budget`
- `tests.test_batch_decode_context`
- `tests.test_tli_incremental_metadata`
- `sparse_attn.ops.tests.test_graph_cache_schema`
- `sparse_attn.ops.tests.test_graph_required_dispatch`
- `sparse_attn.ops.tests.test_batch_explicit_quota_guard`

Coverage includes full-preparation signature storage, matching growth, equal/shrinking lengths, empty cache, each changing metadata field, and prior public shape contracts. Independent reviewers inspected the diff and reran the host suite. The real-tensor test loads production TLIIndexer and compares index tensors against a fresh full preparation with exact equality. It covers head count increase/decrease, float32 to float64, page size changes, clear(), and valid append with same/full L2 width and quantization on/off. Replacing only the two changed methods with their baseline implementations reproduces two assertion failures (head shape/dtype) and two runtime errors (head growth/page size). The first-commit tensor suite passed all five test methods; the follow-up below extends it to seven.

Device changes are covered only by host metadata dispatch; CPU tensor equivalence is not a CUDA device-migration test. Graph tests use mocked CUDA APIs and do not certify capture/replay. Model quality, CUDA kernels, latency and memory have not been tested. No GPU speedup is claimed. The initial system interpreter lacked Torch; the isolated CPU installation resolved that test dependency without GPU execution or model downloads.

## Performance audit findings, not optimizations or measurements

At this base, Qwen `_tl_decode_batched` iterates requests, clears the shared indexer, and rebuilds each row. The batch-prefill selector Graph gate rejects explicit nl/nb settings and subpage sink; that gate does not describe every possible host Graph integration. E156 candidate construction includes data-dependent `nonzero`/`.item()`, then reconstructs full-width scores before normalization/selection. These are source-level work/synchronization findings, not proof of current host-run slowness.

The Graph cache clears all entries at capacity. A selective eviction policy is a separate candidate requiring lifecycle/replay testing; it is not included here. Historical B=1 stage timings cannot quantify current large-batch Graph performance. Current run configuration, capture/replay receipts and matched quality/request timings remain necessary.

## Publication

Prepared for an independent repair branch and draft PR only. Main is not a publication target for this change. CUDA verification remains open; CPU Torch coverage is recorded above.


## Follow-up: avoid rebuilding a complete previous page

Incremental preparation used `(S_prev - 1) // page_size` as the first affected page. At an exact page boundary this includes the previous complete page, even though append-only inputs cannot change it. Use `S_prev // page_size`: a partial previous page is still rebuilt, while an aligned prefix is reused. This does not change allocation identity, metadata invalidation, selector budgets or Graph cache policy.

The real CPU test observes the actual statistics input and quantization calls for 32 combinations: previous length 63/64/65/128, append 1/81, L2 same/full, quantization off/on. Before this change, 16 aligned cases fail the reduced-work assertion; nonaligned cases already pass. Afterward all cases also match fresh full preparation exactly. Example at page size 64 and append 64→65: padded statistics input 128→64 positions, quantized input 65→1 token when quantization is enabled. These are measured operation input lengths, not latency or GPU throughput measurements. The reduction occurs at page-aligned updates; it does not remove per-row batch rebuilding.

Current explicit nl/nb, pooled L2 and sink4/SWA128 receive an additional CPU score/mask exact-parity test at aligned and nonaligned prefixes. The final primary suite is now **44 tests in normal Python and 44 under -O**, including the earlier 42 tests.

### Broader gate coverage and remaining requirements

The earlier 42-test report was affected-unit coverage, not a claim that every repository gate passed. The historical E127 HINTS list was inspected. Its old worktree/no-push instructions belong to that previous task; this repair is scoped to the current draft PR instruction. Applicable runnable CPU gates were also executed normally and under -O:

- near-SWA boundary: 3/3 each.
- gamma-off: 6/6 each; an initially missing sparse-checkout JSON dependency was restored from the exact tracked HEAD before rerunning.
- cluster tail coverage: 10/10 each.
- near-kmeans sample guard: 5/5 each.

The C3 historical comparison cannot import its absent `/tmp/c3_old_b10` baseline. The separate, explicitly nongating near-SWA red/green script reports four SKIPs because `/tmp/near_fix_v2` is absent; neither is called a pass. No fixture was fabricated. RULER publication tests, the repository-wide test suite, GPU greedy tests and CUDA output/performance gates were not executed. No GitHub Actions workflow definitions are present in the inspected tree; remote check status is reported separately.

The minimal existing CUDA correctness command for this incremental change, to run only in an authorized GPU environment, is:

```sh
python exp/bench/test_e127_gates.py indexer
```

This is the existing multi-step incremental-vs-full index/mask gate (including page boundaries), not a new matrix. It remains unrun here and does not substitute for current-method large-batch Graph quality and timing receipts.

### Deferred performance changes

- Per-row batch rebuild: avoiding it needs independent per-request state and lifecycle guarantees for request reorder/termination; CPU equivalence alone cannot establish current host throughput or memory cost.
- Candidate synchronization/full-width reconstruction: removing synchronization or changing reduction widths can change capture behavior, ordering and rounding/tie selection. Preserve the current numerical tree until matched CUDA parity and profiling are available.
- Whole-cache Graph eviction: changing it alters live graph/buffer lifetimes and retained memory. No current host reuse distribution or GPU memory trace is available; leave the policy unchanged in this PR.

Draft status remains appropriate: CPU correctness and reduced work are verified, while fused GPU behavior, model outputs and speed are not.


## Follow-up: remove overwritten growth-buffer zero fills

Four growth allocations (min/max/optional mean and fine K buffers) now use `new_empty`. Every exposed statistic row is initialized by prefix copy plus tail write; every exposed fine token is initialized by prefix copy plus `[lo:S_now]` overwrite. Spare capacity is never exposed and is written before subsequent growth returns it. Genuine zero padding and the leading cumulative-length zero remain unchanged. This does not change Graph policy or eliminate required prefix copying.

A real CPU probe at S=64→65, Hkv=4, tail32, float32, with average statistics enabled observed six `new_zeros` calls before versus two afterward: 199688 versus 131080 zero-filled bytes, removing **68608 bytes** of redundant initialization in this synthetic case. This is an operation-volume measurement, not a timing or GPU bandwidth result.

The new test poisons every growth `new_empty` buffer with NaN and checks 12 configurations (three initial boundary lengths, same/full L2, quantization off/on), each through four successive lengths. It compares each visible index tensor exactly with fresh full preparation, verifies expected allocation counts, and observes poisoned spare capacity surviving until overwritten. The baseline fails 12 allocation subcases plus the spare-poison check; the fix passes. Current suite: **45 normal and 45 optimized-Python tests**, independently rerun. CUDA behavior and latency remain unverified. Earlier incomplete historical comparisons remain incomplete.
