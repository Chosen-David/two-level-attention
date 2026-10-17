# AsyncTLS / FFD bounded integration — 2026-10-17

## Sources and decisions

- AsyncTLS arXiv2604.07815v1, §4.1/4.2, existing 11-page PDF. Official abstract/HTML and public title/author/code searches checked 2026-10-17; no attributable author repository identified. This is a bounded search, not proof none exists.
- FFD official https://github.com/qluoluo/faster-flash-decoding at ca09458ab1536328e5f46502b27891412733cd6d, Apache-2.0. Existing 12-file vendor manifest verified; native source stays unchanged.
- Adopt existing FFD adapter, not duplicate implementation. Add append-position/mask validation before cache mutation. Upstream pyproject dependencies are incorrectly under project.urls; explicit environment requirements are in exp/vendor_baselines/ffd_runtime_requirements.txt. This file is a provisioning specification, not an installed/fully resolved environment.
- Adapt research/docs/AsyncTLS_对手baseline阅读与复现方案_20261013_by_deepseek.md: equations and paper parameters useful. Reject its claim that omitting asynchronous previous-step selection cannot change outputs, and defer its proposed expanded matrix. Current user permits no GPU/matrix expansion.
- agent_doc/guide/paper_figs/evaluation/evaluation.md:129 requires evidence before adding runtime claims; no invented native AsyncTLS implementation. Human GUIDE containing credentials was not printed; no guide edits.
- Prior results searched in agent_doc/results for FFD/AsyncTLS: figure plans/gap inventories only, no usable real-model acceptance evidence. Existing FakeFFD tests reused as host-contract tests only.

## Actual entries

`--method ffd_vendor --baseline-execute` already connects LongBench and RULER to official native Llama. Qwen/MLA rejected. Initial prefill >=128, batch1 unpadded, then single tokens. Native dense prefill + Q2/FP8 variable top-delta decode; no same-K claim. `use_full_graph=False` still permits upstream block-wise CUDA graphs. Local CPU environment cannot execute its kernels (no CUDA/Triton/FlashAttention and transformers4.57.1 vs required4.57.3). No model download or dependency installation performed.

`--method asynctls_component --baseline-execute --asynctls_calibration FILE --asynctls_blocks 128 --asynctls_tokens 1024` connects the existing LongBench/RULER Qwen3 entry to a new **paper-derived synchronous §4.1 component reproduction**. For the paper's RULER token setting pass 512 explicitly. No batch run is authorized by this example. Native dense prefill; full Q/K normalization and RoPE; group block score, per-Q-head softmax then group mean, full-dimensional selected-KV output. Output treatment name is `asynctls_component_sync_<identity>`; receipt remains candidate and quality_accepted=false. All checkpoint bytes are hashed once before adapter binding.

No offload, previous-step block state, resident-cache delta transfer, asynchronous overlap, compressed physical index, or efficient sparse kernel. The component rebuilds metadata and fake-quantizes in FP32; its latency/memory is **not** AsyncTLS system performance. Different current-step versus lagged blocks may change quality. Partial last block is retained; paper's floor expression is not used to silently drop valid tokens.

Explicit unverified assumptions: symmetric per-token int4 [-7,7] with round-to-even and reconstructed floating storage; softmax denominator covers selected blocks; all layers active; complete last block. Paper does not supply enough implementation detail to establish these choices as author behavior. No sink/SWA override is added. Sliding-window configurations, padding, external populated caches and chunked continuation are rejected. Failed cache use is rejected thereafter.

## Calibration contract and remaining blocker

No real calibration artifact is available or manufactured. Execution fails without it. JSON schema:

```json
{
  "schema": "asynctls-component-calibration-v1",
  "provenance": {
    "producer": "exact producer code revision and invocation",
    "held_out_input_sha256": "64 lowercase hexadecimal digits",
    "score_rule": "mean_qheads(max_abs_q * max_abs_k); post-RoPE; held-out"
  },
  "effective_rope": {"rope_theta":1000000.0,"rope_scaling":null},
  "checkpoint_files": {"config.json": "sha256", "model.safetensors": "sha256"},
  "model_shape": {"model_type":"qwen3","num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128},
  "channels": {"0": [[0,1,2]]}
}
```

Shape/channels above are illustrative, not Qwen3-8B calibration. Exact effective RoPE (including any YaRN changes), loaded shape, **every layer and KV head**, unique in-range channels and the exact local config/all weight-file hashes are required. Use `checkpoint_files(local_model_directory)` to compute the mapping without loading a model. Compute held-out post-RoPE Q/K maxima per layer/head/channel, mean query-head products per KV group, then select top dc (paper GQA dc=32). Do not calibrate on evaluation examples. The loader validates identities/structure, not truth of declared held-out provenance; independent data review remains mandatory. No existing trace lacking required Q coverage is implicitly accepted.

## Verification scope

CPU synthetic tensor oracle plus actual randomly initialized tiny Qwen3 (saved/loaded locally, no pretrained download) tests establish component wiring and dense-budget parity, request reset, position/mask rejection, not task accuracy. FFD uses control-flow doubles; no real Llama/kernel/quality acceptance. Tests run both normally and with `-O`; independent review required before dev publication. Full inherited baseline suite has pre-existing SparQ receipt failures on main; preserve and report separately rather than fix unrelated code.

Paper metadata is independently published on main; all engineering changes only original dev. Real checkpoint/calibration, GPU numerics/quality/performance and full AsyncTLS system remain open. No CI result is inferred from local tests.

Final producer checks: component 4/4 normal and 4/4 `-O`, FFD host contract 6/6 in each mode. Component adds a separate scalar L1/L2 selected-token oracle (GQA + partial final page). Exact source hashes/raw logs: `agent_doc/results/asynctls-ffd-20261017/`. Unchanged main3628501 full suite reproduces 1 failure + 2 errors in 15 tests (SparQ missing receipts); no unrelated repair or green-suite claim. No tracked GitHub workflow or remote CI success established.

## Scope revision v2: paper-derived lagged/offload candidate (2026-10-17)

User subsequently authorized §4.2 implementation; the synchronous component above remains a reference. `--method asynctls_paper --asynctls_backend cpu|cuda --asynctls_capacity 131200` reuses its frozen calibration/checkpoint/RoPE, entry guards and benchmark request receipts. CUDA is explicit opt-in, not executed or validated here. New layer state stores full host KV, device block bounds and immutable selected resident blocks. Decode consumes previous-step selected blocks plus the newest-token candidate; it prepares the next selection, reuses unchanged resident blocks, appends growing tails from the device token, and copies only newly selected blocks. A dedicated CUDA stream/event and tensor stream records protect lifetime; request cleanup synchronizes owned transfers, including failures. HF offloading/sliding caches, replacement/crop/mutation, noncontiguous positions, padding and capacity overflow fail closed. Fixed pinned capacity avoids reallocating buffers during pending transfers.

This is a **paper-derived reproduction candidate**, not official code or accepted full AsyncTLS. Current-token candidacy (not forced final selection), complete tails and inherited symmetric fake-int4 remain explicit assumptions. Host `.tolist()` selection barriers, Python loops, reconstructed quantization and duplicated final-prefill Q are reference overhead; no packed-int4 storage, measured overlap, speed or memory claim. CPU synthetic and locally generated tiny-Qwen tests do not establish GPU numerics or task quality. Source, logs and independent review binding: `agent_doc/results/asynctls-offload-20261017/contract.json`.

### Existing TIA route audit and reuse boundary

Existing `TIAIndexer.compute_mask` lines 115–122 implements previous-coarse-mask exchange and clear/reset. `prepare_index` uses fixed tail channels (cmp4: [48:64]+[112:128]), asymmetric min/max fake-int4, and forced tail window; complete K/V/index tensors remain on their device. No host store, resident delta transfer or prefetch stream exists there. Preserve this actual historical route and its protocols; replacing its mask behavior would change old experiment identity. The new transport cannot reuse a nonexistent old transport; reuse is at the existing baseline/calibration/receipt infrastructure. Do not equate the routes or label TIA official AsyncTLS.

Historical Qwen result files `exp/results_ruler/Qwen3-8B/none_20260202_173101.txt` and `tia_64_128_512_c4_async_20260208_145311.txt` report MK2 32K .9820/.9860 matching the paper. Their names/scores do not prove historical producer revision or complete protocol identity. Old lm-eval and current KVCacheFactory routes differ in sample count/max generation/stopping; do not silently change settings, merge results or infer causal explanations from those differences. Freeze producer SHA, checkpoint/RoPE, prompts/tokenization, dataset bytes/sample IDs, generation settings, calibration and receipt before any fair comparison.

### Single bounded execution handoff

Existing authorized executor first checks idle resources and current jobs; do not preempt work, download weights, start new paid resources or automatically run experiments. Safe CPU dry-run: `python -m sparse_attn.asynctls_offload` prints scope only. Real held-out calibration and GPU validation remain blocked, not fabricated. Upon separate execution authorization, use existing benchmark CLI/protocol with the explicit method/backend/calibration above and a small existing-input smoke before the approved matrix. No new model matrix.

User's latest scope: **RULER quality/reporting 32K only**; LongBench uses existing actual task lengths. Preserve existing 64/128K negative quality evidence and disclose in supplement; no new longer quality matrix. Kernel lengths 32/64/128K; E2E 32/64/96K. Report matched-batch latency separately from capacity-limited throughput. Never combine 128K speed and 32K quality into a 128K lossless claim. No 1M runs. Receipt status remains candidate/quality_accepted=false until independent scientific acceptance.

FFD original publisher PDF is now archived on main57b30397 (PMLR306 ICML2026, SHA256 6fa54f7eb14c0f9648a6b0fbb9cfd61889ab6b0d0c218a4df02ede96580d05e9); the earlier HTTP403/missing-source statements above are historical. Full real-model/GPU acceptance remains open. Inherited SparQ full-suite failures are unchanged and no remote CI success is established.
