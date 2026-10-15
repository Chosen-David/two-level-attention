# Registered baseline integration — 2026-10-15

Scope: S-T023 and E3, user-requested missing baseline integration. No GPU jobs,
weights, paid calls or new experiment matrix were started. This is engineering
candidate evidence, not acceptance of a paper quality/speed point.

Safe default: `python -m sparse_attn.baseline_runtime` prints a dependency-light
plan. The new `ffd_vendor` and restored `sparq_vendor` benchmark routes require
`--baseline-execute`; absent that flag they fail before model loading. Both
LongBench and RULER keep their existing datasets, generation loops and scoring.
No new scorer or result conversion is used.

| Method | Actual integration | This change's runnable validation | Remaining boundary |
|---|---|---|---|
| FullKV (`none`) | existing native model | affected CPU regression | backend/hardware-specific quality and time require real run |
| Quest vendor | Qwen3; native prefill and first two layers, quality-mask sparse decode | tiny actual HF Qwen3 full-budget parity; explicit missing-position/custom-mask rejection | full QK mask is not efficient sparse KV reads; GPU/quality not measured |
| FASA vendor | Qwen3; calibration-dependent sparse decode, native prefill/first layer | shared decode-mask guard; existing official core byte match | no calibration/GPU run; not replacing frequency calibration with low-frequency tail |
| SparQ vendor | **restored missing files**; Qwen3 with official gpt-fast per-query-head core, native prefill | tiny actual HF Qwen3 prefill/cached decode, full-budget parity, nontrivial sparse route | adapter candidate; full resident KV, mean-V recomputation cost; GPU/quality unverified |
| MInference vendor | existing Qwen3 sparse prefill/native dense decode | exact pattern coverage/invalid starting layer rejection; official core byte match | compiled deps/GPU and model quality not tested here |
| FFD vendor | **new native Llama loader**, official model and Q2+FP8 cache, top-delta sparse decode | cache injection/route/invalid-input gates using CPU doubles, NOT actual FFD kernel | official model/kernel not executed; CUDA/Triton dependencies required; Qwen3/MLA explicitly unsupported |
| ClusterKV | no local faithful runtime adapter | source/task inventory only | official efficiency dependencies, model/offload lifecycle absent; blocked, not substituted with ccluster |
| HISA | official TransArch/TileLang source fetched | identity resolved: Xu et al., arXiv2603.28458v4 | DSA/Sparse-MLA indexer; no current Qwen3/HF integration |
| HiP / MoBA | conditional candidates retained | no new integration | supported model/training conditions unaccepted; `tli_moba` is not official MoBA |

## FFD identity and routing

FFD means *Faster Than Flash: Exploiting Attention Sparsity for Efficient
Long-Context Decoding*, official
[qluoluo/faster-flash-decoding](https://github.com/qluoluo/faster-flash-decoding/tree/ca09458ab1536328e5f46502b27891412733cd6d),
package 0.2.0, fixed `ca09458ab1536328e5f46502b27891412733cd6d`, Apache-2.0.
The shipped `ffd_core` files are unchanged and hash-checked before loading;
`UPSTREAM.json` and original LICENSE accompany them. Upstream `oc_model.py`
explicitly raises NotImplementedError for Qwen2/3. We do not replace Qwen3 with
LlamaAttention or claim preservation of Qwen3 Q/K norm through such replacement.

Use `--method ffd_vendor --baseline-execute --ffd_delta 5 --ffd_cache_length 131200`
only with a locally available Llama checkpoint, existing benchmark task/config,
and a separately authorized hardware run. No execution command was launched.
Requires transformers 4.57.3 and upstream CUDA/Triton dependencies. GQA Llama
requires FlashAttention: the upstream SDPA fallback does not enable GQA, so this
combination is rejected before loading weights. Initial
prefill must contain at least 128 tokens; B=1, unpadded, initial full prefill then
single-token continuation only. Cache size must be a multiple of 128 and cover
all input/generated tokens. RULER's existing Qwen-specific long-context gate is
not repurposed: this native Llama candidate is currently limited to <=32K RULER,
no Qwen YaRN. Unsupported inputs raise, including dense cache continuation,
chunked cached prefill, mutated configuration and unexpected dense decode.

Per-forward hooks supply a fresh official QuantizedKVCache and check upstream
route counts against every layer. A request with zero sparse decode calls is
rejected before a prediction receipt is published. The official dense prefill
backend (FlashAttention or upstream SDPA fallback) is recorded, rather than
silently labeling every prefill FlashAttention. Graph mode stays at the fixed
upstream/default settings; no new full-model graph integration is claimed.

`delta` is passed unchanged to upstream **log2-scaled** scores (`m_global-delta`);
the natural-log offset is delta*ln(2). This is not fixed top-k. Q2 key scanning
and FP8 residual are part of this external baseline; they do not revive the
project's canceled quantization research line.

## Comparison and timing contract

Native system and controlled final-unique-K comparisons remain separate.
FFD preserves top-delta variable budget; a fixed-K control was **not** invented
by toggling its experimental top-k threshold option. FFD same-K cells remain
unavailable until a separately observed/frozen budget-matched protocol exists.
SparQ exposes k=1024 and r=clip(round(D/(context_length/(2*k))),1,D), with the
context-derived values recorded. Its query-head K and GQA union K differ;
actual per-group unique support is not measured here and remains null. The
legacy outer `budget` field is also null for new baseline receipts; an empty
TLI metrics collector is not reported as a measured zero-token budget.

New FFD/SparQ `baseline_receipt` records actual route counts and synchronized
host request wall around the existing generation context (cache creation,
quantization/index/mean work and final synchronization included). Tokenization,
loading and scoring remain outside. It is not TPOT, kernel latency, scientific
acceptance or the Qwen `paper_capture` contract. Do not combine it with
`--paper-capture timing`, which rejects unknown vendor forwards. No successful
receipt is created on model/guard/sync failure. CPU doubles prove control flow,
not the CUDA kernel or HF FFD generate/cache compatibility.

## Existing vendor sources

FASA `fasa_utils.py` matches official commit
`96c00c46db12ccee56104175470aa29579d66c58` byte-for-byte. MInference
`minference/modules/minference_forward.py` matches tag v0.1.6,
`d542c489a173795f5781409cb0307f4bef0f2b73`, byte-for-byte. `SOURCE_CHECK.json`
records the limited matched files and licenses. This does not establish every
historical adapter file's original provenance. Quest is a modified Qwen3
adapter of the pinned upstream quality path, not an unmodified vendor file.

FASA's selected-K mask cannot use a prefix slice of a nontrivial source mask;
the supported B1 unpadded decode now rejects such masks. Quest no longer accepts
missing decode positions. MInference rejects all-dense starting layers and
missing head patterns instead of allowing upstream default pattern substitution.

Tests: `tests/test_baseline_integration.py`, normal and `-O`; logs and retained
first-failure regression evidence in `exp/paper/evidence/baseline-integration/`.

## Source/cache follow-up

See [FOLLOWUP_20261015.md](FOLLOWUP_20261015.md) for the subsequent FFD
cache ownership/layer-state repair, actual official source/submodule downloads,
resolved HISA identity, and exact remaining ClusterKV/HiP/MoBA boundaries.
Earlier first-integration test evidence is historical, not the latest source hash.
