# Qwen3 Quest/SparQ Transformers API compatibility

Tested source base: `65ed618008cccc71e6e24b575d24cb6bb2e9d447` (2026-11-10).

## Confirmed failure and bounded fix

On Transformers 5.22.2 / Torch 2.8.0 CPU, the prior baseline integration suite had 4 failures and 11 passes: both adapters passed the removed `cache_position` as a fifth native attention positional argument. New targeted tests on pristine base reproduced 6 failures and 1 pass. Native prefill now forwards an explicit position only by keyword, retaining HF4 compatibility.

HF5 provides decode `position_ids` instead. Recovery accepts only a single integer row equal to the contiguous physical positions derived from that layer's cache length; offsets, missing positions, malformed rows and custom masks are rejected before cache mutation. Quest reuses the existing idempotent Llama rotary-signature bridge, without modifying official selection math. Duplicate output_attentions forwarding is removed.

## Actual CPU verification

- HF5 5.22.2, normal and `-O`: 25 passed, 2 deselected, 14 subtests.
- HF4 4.56.2, normal and `-O`: 24 passed, 3 deselected, 14 subtests.
- Suites: `tests/test_qwen_vendor_hf_api.py`, `tests/test_baseline_integration.py`, `exp/vendor_baselines/quest_vendor/test_qwen3_quest_patch.py`, `tests/test_llama_vendor_adapters.py`.
- HF5 selection excludes `asynctls`; HF4 excludes `asynctls and base_prompt`.
- Real tiny GQA Qwen3 models exercise ordinary and decoupled head dimensions, prefill plus three decode steps, full-budget native-logit parity and all-layer cache lengths. Negative positions and masked calls preserve caches. Shared Quest RoPE behavior is covered across Qwen/Llama adapters.
- Independent read-only review accepted the physical-position guard and compatibility approach; final CPU reruns and fresh-main check precede publication.

## Explicit remaining limits

HF5 fallback adds an arange allocation and `torch.equal` check, potentially a device synchronization on CUDA. This is an API correctness fix, not an optimization or CUDA Graph compatibility claim. No GPU, quality corpus, or performance run was started. Historical scientific results are not reclassified without evidence of using this failing API version.

Current-main AsyncTLS Llama smoke fixture uses the obsolete global channel schema; PR43 separately addresses the integration. HF4's excluded prompt test hits an environment datasets/huggingface_hub incompatibility, not this adapter path. Required E127 gate cannot run with CPU-only Torch. The tracked TLI batch gate imports removed `sparse_attn.indexer.tli_indexer`; the README-named NFA batch gate is absent. These gates are not reported as passed. CI status must be checked separately; no check means no CI pass.
