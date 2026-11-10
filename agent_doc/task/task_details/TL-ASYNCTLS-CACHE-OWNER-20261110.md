# TL-ASYNCTLS-CACHE-OWNER-20261110

## Scope and source
- Tested main: `65ed618008cccc71e6e24b575d24cb6bb2e9d447`.
- Continues the lifecycle finding on PR #43 comment 6762856112. PR #43 installation validation and #41 padding remain separate; no duplicated scoring or installation changes.
- Qwen3/Llama/GLM AsyncTLS semantic-reproduction adapters keep M_(t-1) on attention modules, while HF keeps K/V on cache objects. An A prefill, same-length B prefill, A decode consumes B state. Existing code also checks missing state only after KV append.
- Actual CPU tiny model calls, not AST-only proxy: interleaved cache, crop, uninitialized nonempty cache and failure/retry regressions fail on main. Normal serial and full-budget native-dense comparisons pass before/after.
- Reference implementation contracts inspected: installed Transformers 4.56.2 and 5.22.2 `cache_utils.py` (DynamicCache/layers/crop); model-specific native attention used as dense reference. This is lifecycle safety, not a new claim of paper/official AsyncTLS equivalence.

## Change
Weak cache ownership and expected layer length are checked before projections/KV update. Clear state at step start, commit only after successful projection; exceptions cannot leave a usable stale block set. Non-weakref cache raises before state mutation. Fresh prefill or empty cache begins/rebinds a request. No score, budget, channels, INT4, selected-token or current-token protection math changes. No strong cache reference or new tensor comparison/synchronization.

**Per-layer guard, not a whole-model transaction.** After a later-layer failure, earlier layers may already have advanced. Retrying can further advance earlier layers before the failed layer rejects. Discard that failed cache and start a fresh request/prefill; no rollback guarantee. Same-length in-place cache edits, beam reorder, external crop-then-regrow and arbitrary external mutations are unsupported and not detected merely by owner/length. Interleaved live requests are explicitly rejected, not newly supported.

## Verification
CPU Torch 2.8.0+cpu, no GPU invocation.
- Initial regression on main: 12 failed / 12 passed; tests do real tiny Qwen/Llama/GLM forwards. Final expanded suite on main: 18 failed / 12 passed (three failures additionally test the new helper contract).
- Candidate HF5: 30 passed normally and under `-O`; HF4: 20 passed / 10 GLM skipped (GLM model unavailable in HF4), normally and under `-O`.
- Covers equal-length request switch, crop, missing state before update, weakref lifetime, non-weakref no mutation, sparse serial determinism, single-token start, empty-cache reuse, multistep full-budget native dense logit parity, failure/retry and late-layer failure/fresh-prefill recovery.
- Adjacent `test_baseline_integration.py` + `test_llama_vendor_adapters.py`: baseline and candidate same 5 failures / 15 passes. Four HF5 Quest/SparQ API cases already covered by PR #44; one obsolete global-channel AsyncTLS fixture covered by PR #43. Candidate plus new suite: 45 pass / 5 fail. Not an all-suite pass.
- Required E127: 6 failures (CPU-only CUDA and removed/renamed symbols/modules); retained TLI batch gate collection fails on deleted `sparse_attn.indexer.tli_indexer`. README-named `test_nfa_batch_gates.py` is absent. Neither gate claimed passed.
- Independent read-only parent review checked guard order, weakref, exception and per-layer boundaries. Syntax/diff checks pass. Fresh remote main is rechecked before publication.

## Remaining validation
No GPU, Graph, throughput, memory peak, corpus quality or new experiments. Existing host `.tolist/.item` paths remain; this is not Graph support. No claim that sequential historical result arms were affected, no inferred score delta and no deletion/invalidation of unrelated historical results. Optional future GPU acceptance (only with explicit authorization): run this lifecycle suite on a CUDA-enabled, device-parametrized tiny fixture and compare the same source/config/inputs; performance needs a separately matched protocol, not CPU timings.
