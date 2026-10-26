# TL-FASA-BATCH-MASK-20261021

## Scope and source identity

User-requested baseline semantic audit. Local tested base: e6faba35342dbc21414e0212b27b68d209726b96.
Remote read subsequently found 5964816457d016b4f74062e2b6ac43bf5685d074 (async TLI candidate guard); used as the publication base tree; its async TLI files are retained unchanged. No production defaults, experiments, budgets or SGLang files changed.

Paper: FASA arXiv:2602.03152v2, Sections 4.1–4.3 and B.4 (local section reading, not a new full-paper review), https://arxiv.org/html/2602.03152v2 . The paper selects tokens using calibrated head-specific frequency channels, then applies full-dimensional attention to selected original KV. It does not license attending padding.
Official source: wangyifei0047/FASA-ICLR2026 commit 96c00c46db12ccee56104175470aa29579d66c58, monkey_patch/frequency/utils.py lines 91–125. The vendored core checksum is 52eb64f73ef24a05e64cee0a009b337157afb44ce2b4a66bfbe126077596569a; Apache-2.0 LICENSE checksum 1ac51f2e6ab5961c87397376c7e7baa3a5d392aa85c7f22e00f6bad01ae9cd10. Core source/checksum inspected; the Qwen3 adapter is not claimed byte-identical to upstream.

The adapter intentionally preserves Qwen3 q/k normalization, supplied RoPE and attention scaling; it retains dense prefill and layer 0, repeats KV for per-query-head selection, and adds a batched left-padding interface. These adaptations are distinguished from the official token selector, not treated wholesale as bugs. Calibration/run identity, generation protocol, other baseline adapters, real model quality and GPU timing are not certified here.

## Two connected defects and minimal correction

1. batch_valid_len_from_mask sliced [B,1,1,S] to [B,1,S] despite expecting [B,S]. Real tensors returned nested lengths such as [[1],[2]], rejecting a legal context [1,2] for both additive and boolean masks. Explicitly select both singleton axes. The prior mock ignored the indexing key and concealed this defect.
2. FASA gathered the selected boolean mask correctly but added it to logits. False columns remained attendable. Use masked_fill(False, -inf); preserve additive bias behavior. When the batch-wide token budget exceeds one row's valid length, selected padding must still receive zero final probability. Comments now state this limitation of top-k selection explicitly.

Before correction, real CPU tests reproduced a boolean/additive numerical mismatch (maximum absolute difference 0.672) and the legal-context length rejection. No specific historical prediction has been bound to this trigger, so no historical result is declared invalid.

## Validation and limitations

Environment: Python 3.12, torch 2.8.0+cpu, transformers 4.56.2; offline flags enabled; no model download or GPU execution.

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python [-O] -m unittest   tests.test_fasa_boolean_mask tests.test_batch_decode_context
```

Nine methods pass normally and under -O: actual 4D bool/additive lengths, incorrect shape/context rejection, selected-padding zero probability, short row/budget overflow, selected-order mask gathering, exact bool/additive equality, CPU SDPA parity, additive bias and unmasked behavior, context isolation. Production FASA helper bodies are AST-loaded to avoid upstream module-level CUDA allocation; this is not full-model/GPU integration.

Broader baseline integration: 24 methods total (these nine plus existing integration suite), 21 pass and three SparQ tests fail in both modes: missing routes, missing _baseline_stats, and missing expected no-sparse rejection. A separate pristine e6faba3 worktree reproduces exactly those three failures in the 15-method integration suite. Do not claim repository-wide success. An independent reviewer also observed the pre-existing decode-padding AST harness missing batch_decode_ctx (64 Qwen subcase errors), outside this patch. GPU, Graph capture/replay, model outputs, throughput and all-baseline semantic certification remain untested.

Git command-line authentication is unavailable; publication uses the connected official GitHub Git-data interface on a dedicated branch, with fresh head checks and readback. No credential extraction, proxy change or attachment-upload workaround is used. Publication status is reported separately.

Independent read-only review reran the nine tests in normal and -O modes and accepted both fixes. Additional independent gradient probes confirmed zero padding value gradients; no GPU or historical-run certification.
