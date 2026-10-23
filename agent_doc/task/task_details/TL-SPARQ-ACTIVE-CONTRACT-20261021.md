# TL-SPARQ-ACTIVE-CONTRACT-20261021

Second requested review round; distinct from first-round FASA masking and TLI growth work. Base publication tree: main 5964816457d016b4f74062e2b6ac43bf5685d074. Active entrypoints import qwen3_sparq_patch.py; the parallel qwen3_sparq_patch_gptfast.py contains receipt/guard logic but is not their route.

## Confirmed defects and fix

The three existing baseline integration failures are missing route/identity instrumentation and missing prefill-only rejection, not proof that SparQ attention math failed. Wire counters and identity into the active adapter. Successful native prefill and sparse decode increment separate counters; baseline_request now rejects a request that never reaches sparse decode.

The active wrapper ignored nontrivial attention masks and accepted missing decode cache/position. Reject unsupported batch sizes, padded/custom decode masks, batched contexts, missing cache/position, and use_cache=False before projection/cache mutation. Shared guard alone is insufficient because a BatchCtx intentionally relaxes padding for other adapters. Installation validates positive integer parameters, full Qwen3 coverage and duplicate installation before mutating layers. Reject a derived r exceeding head_dim; do not silently clamp or alter valid r/k configurations.

## Method / official source / adaptation distinction

Paper: https://arxiv.org/html/2312.04985v5 , Sections 3–4 / Algorithm1, read locally for this audit. Official implementation: graphcore-research/sparq-gpt-fast f2eeb9d10e79d3ffecc48015e55157bab0f6e269, sparq.py. Both vendored sparq.py and active sparq_attention_official.py SHA256 b0f8929b7fd7e77c808fd999e6af3412740e16d90e01d5c9efce52ae599e33c3; BSD-3 license SHA256 dc2fcddedde170f348996b8f2a13058a48e0e6cd8cc1dd9a39e6a3f11afffec8.

Preserve fixed upstream top-r proxy, top-k full-dimensional attention and running mean-V compensation. Qwen3 q/k normalization, supplied RoPE, DynamicCache and GQA expansion remain adapter behavior; full KV stays resident. Paper v5 Algorithm1 includes a local-window term absent from this pinned implementation. Per-query-head selection is not certification of all paper GQA variants. No local-window/protected-set or selection change is introduced under this bugfix; comparisons must retain the pinned implementation/adapter identity instead of claiming every paper configuration is reproduced. No historical run is retroactively invalidated or relabelled.

## Tests and remaining scope

Python3.12, PyTorch2.8.0+cpu, Transformers4.56.2, offline flags. Ordinary and -O each pass33 methods:17 baseline integration (including tiny randomly initialized HF Qwen3) plus16 active SparQ lifecycle tests. Full-budget parity, sparse finite outputs, route counts, prefill-only failure, invalid installation without mutation, seven invalid decode cases, sequential/interleaved/cropped/reused caches, weakrefs, fp32/fp16/bf16, strided tensors and no-host-scalar runningmean guards remain covered. Updated legal lifecycle fixtures carry explicit positions; no-cache decode now intentionally fails before state change.

Before fix, original three receipt tests reproduce1failure+2errors. New invalid-install test triggersZeroDivisionError; the seven invalid decode cases are wrongly accepted by old source. These are CPU evidence, not pretrained quality or GPU performance. Do not count the33 methods as33newtests or count subcases as independent experiments.

Command: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python [-O] -m unittest tests.test_baseline_integration sparse_attn.ops.tests.test_sparq_request_lifecycle

Performance audit retained: GQA expansion and full-resident KV remain; candidate nonzero/item synchronization, duplicate L1 construction/full-width score reconstruction and Graph eviction require matched device profiling/state guarantees before change. This patch makes no latency claim. Full repository, real model generation, CUDA Graph, GPU timing and all-baseline semantics remain unverified.
