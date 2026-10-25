# Compact far skipping and KV eviction

This opt-in Qwen3 decode path omits far keys from **page statistics, L1 scoring,
L2 scoring, and final attention**. It bypasses the full-history index/mask path
and the dense fallback in `eager_decoding_attn`. It provides a dynamic eager
selector and a fixed-shape selector wired into the Qwen3/TLIStaticCache whole-model
CUDA Graph entry. Both use gathered SDPA; neither is a fused selection kernel.
CUDA capture/replay, GPU latency and pretrained-model quality have not been
validated on hardware in this environment. The real-CUDA tests are present but skipped.

## Enable

Add these flags to an existing Qwen3 TLI invocation with a working static layer
skip mask or dynamic gate:

```bash
--tli_far_skip_backend compact --tli_far_skip_cache evict \
--tli_near_len 4096 --tli_near_pages 32 --tli_near_tokens off \
--tli_boundary_bypass true --tli_far_select 4bit --tli_near_select 4bit
```

For request-local dynamic calibration, also use:

```bash
--tli_enable_layer_skip false --tli_dyn_dgate --tli_dyn_dgate_eps 0.05
```

Use model `eval()` plus `torch.inference_mode()`/`no_grad()`, HF 4.56.2
`DynamicCache` for ordinary eager, batch size 1, single-token decode, and no padding. SDPA with an
unpadded query supplies the required `attention_mask=None` to the attention
module. An explicit non-None decode mask is rejected, including custom bias.
The model head dimension must be 128 and native sliding-window layers must be
disabled. The `tli_swa_size` protected window is still supported.

`retain` is an ablation: it runs the same compact selector and attention but
keeps the ordinary full KV cache. `evict` additionally releases far storage.
Defaults remain `legacy`/`retain`, with unchanged default treatment identities.
Only layers with a frozen/selected `skip_far=True` use the new compute path;
other layers still execute the historical TLI implementation and keep full KV.
A short prompt with no far interval does not activate the existing dynamic gate.

## Selection and storage contract

For logical length `S`, page size `b`, protected sink `s`, window `w`, and fixed
near length `n >= b`, the near page boundary is
`max(s // b, (S - w - n) // b) * b`.

- Full near pages compete for the unchanged `near_pages` cap using avg/minmax
  scores, averaged over the query heads in each KV group.
- Partial protection-boundary pages contribute only their unprotected **near**
  tokens to L2. Tokens outside sink/near/window never enter a candidate buffer.
- L2 normalizes over candidate and protected tokens per query head, averages
  probabilities per KV group, then fills the remaining total K2 budget from
  near candidates. It does not enlarge the near interval or page cap to fill K2.
- Final attention gathers the selected original-width K/V and calls SDPA.
  Eligible CUDA inputs can use fused attention; even a math fallback receives
  only selected KV. No full-history
  score tensor, boolean mask, or attention fallback is allocated on this path.
- Ties choose the earlier absolute page/token. Optional L2 quantization is a
  scoring approximation, not compressed KV storage.

This is a strict far-off semantic contract, **not bitwise parity with the old
gate**: the old sink-boundary bypass can admit non-sink far tokens. Those tokens
are excluded here. Stable tie-breaking and arithmetic ordering also differ.
Existing dynamic calibration is unchanged; its unaligned far endpoint can
conservatively include up to `b-1` tokens in the aligned near region.

After the gate freezes, each evicted layer owns fixed-address K/V buffers of
`s + n + w + b - 1` slots: a fixed sink prefix plus a cyclic suffix. Absolute
position `t >= s` maps to `s + (t-s) % (n+w+b-1)`. Only live near/window positions
are read, including when the tail wraps. Unused/expired slots are not keys for
attention. In DynamicCache eager, `get_seq_length()` returns logical `S`, so HF generation and
RoPE keep absolute positions. Keys are not re-rotated.

The first eviction copies the retained intervals and releases the old cache
reference. Subsequent appends write exactly one K token and one V token in
place; they neither allocate nor copy old KV. This saves long-context active
decode KV memory, **not the prefill peak**. Fixed reservation can use more space
than a short full cache when a static gate activates before far exists. CUDA
reserved allocator memory need not fall. External references to old KV or
autograd graphs would prevent release; the runtime requires inference and
rejects differentiable cached KV.

The optimized eager selector also:

- caches immutable complete-page statistics in bounded cyclic slots; each new
  complete near page is reduced once, while query-dependent scores are always
  recomputed;
- gathers token and subspace indices in one operation, with no full-D temporary;
- uses exact score/position integer keys for stable top-k (FP16/BF16/FP32), then
  sorts only selected indices. FP64 retains the general stable-sort path;
- constructs candidates in absolute order and reuses range/cat buffers, removing
  candidate/output full sorts and repeated metadata allocation. All-selected
  pages and zero middle-token budgets bypass unnecessary scoring.
- represents each KV group's query heads as independent query rows in noncausal
  SDPA, avoiding the math GQA backend's K/V repeat. Shared K/V and zero-dropout
  attention semantics are unchanged; backend rounding need not be bitwise identical.

`FarSkipWorkspace` is per-request, append-only, sequential, single-stream state.
It must not be shared between requests or used after modifying existing K or
dimension indices. The model bridge resets it at prefill and `indexer.clear()`.
Selections returned with an explicit workspace borrow its position buffer until
the next selection; clone IDs if they must outlive that call. This eager version
still has multiple framework kernels: it does not claim a fused L2 kernel or
zero temporary allocations. Capturing this dynamic eager selector directly is
unsupported; use the separate fixed path below.

Eviction is irreversible within a request. Changing the gate back to far-on,
changing geometry/cache mode, cropping/rewinding, beams/batching, offload, and
legacy tuple export are rejected. Start a fresh `DynamicCache` for a new
request. A compact layer's `reset()` clears its state and requires prefill and
calibration again. Cache sharing/serialization and speculative decoding are not
supported. Arbitrary HF StaticCache, batch>1 compact decode, chunked/sparse prefill, projection,
per-query-head selection, clustering, async top-k, sigma and MoBA combinations
are outside this implementation and fail explicitly.

## Fixed shapes and CUDA Graph

The existing `exp/bench/e152v4_graph_latency.py` now calls
`fill_static_from_dynamic(cache, dyn, [length], model=model)`. For compact+evict,
this freezes the per-layer gates and replaces only far-off static layers with
bounded rings; far-on layers keep the original full static cache and graph
route. Release `dyn` and other full-cache references after setup. Refill and
**recapture at every request boundary**: gates, dimensions and addresses can change.
Legacy-mode setup/capture reuse is unaffected.

The early Qwen attention hook consumes this cache before ordinary full-width
index preparation. `FarSkipStaticLayer.update` uses device `cache_position`
indices for its two in-place writes. Selector boundaries derive from that tensor;
no host length, `.item()`, dynamic `nonzero`, or changing output shape is used.
This static adapter's host `get_seq_length()` is a setup hint, not a replay
counter: pass explicit `position_ids` and `cache_position`, as GraphRunner does.
The single-stream caller must supply the same position for warmup/capture or
advance by exactly one token. Jumps, rewinds and speculative decode require refill.

The fixed selector seeds bounded complete-page summaries during setup, then
refreshes only the newest complete page each step. It gathers selected full
pages plus at most two padded partial fragments for L2, and at most K2 slots
for original-width attention. Invalid lanes are redirected and zeroed before
arithmetic and excluded by masks. Stable top-k ranks every valid lane ahead of
padding, even at negative infinity. Outputs borrow scratch until the next call.

The graph entry supports unpadded B=1 only. It registers the identity of its
causal additive mask in `cache._far_skip_attention_mask`; external callers must
construct that mask from the same position, as GraphRunner does. Arbitrary
custom bias, padding, or mask mutation violating that contract is unsupported.
The driver writes only changed mask intervals and fills persistent scalar/token
buffers, rather than constructing four device tensors and rewriting S_cap every
step. Far-on layers still incur their historical full-width work.

Example (existing target-GPU model installation required):

```bash
python -m exp.bench.e152v4_graph_latency --method tli --model-path /models/Qwen3-8B \
  --context-length 32768 --max-gen 128 --graph-mode graph --out /tmp/far-graph.json \
  --tli_far_skip_backend compact --tli_far_skip_cache evict \
  --tli_near_len 4096 --tli_near_pages 32 --tli_near_tokens off \
  --tli_enable_layer_skip false --tli_dyn_dgate --tli_dyn_dgate_eps 0.05
```

Repeat with `--graph-mode eager` and compare generated tokens. The staged path
uses the same fixed selector; ordinary DynamicCache eager remains separately
optimized. Mixed-gate full-model GPU execution remains an unrun gate, even though
CPU setup/routing and all-far-off random-weight Qwen outputs pass.

## Validate and measure

```bash
python -m pytest -q tests/test_far_skip_compact.py tests/test_far_skip_optimized.py tests/test_far_skip_graph.py
python -m exp.bench.far_skip_decode --device cuda --legacy \
  --lengths 8192 32768 65536 --warmup 8 --steps 64
```

The benchmark emits raw per-step wall/CUDA-event times, first-decode conversion
time, actual KV storage bytes and GPU allocated peak. It covers cache update,
selection and attention for one synthetic layer. The historical `legacy` arm
has the boundary/tie differences above and is not an equality oracle. It is not
an end-to-end model throughput or quality experiment. Run RULER/LongBench and
full-model latency on target GPUs before enabling this path by default.

CPU correctness includes independent scalar page/dense-probability selection
oracles, a dense masked attention oracle, far NaN poisoning, exhaustive short
page/protection boundaries, repeated cache rollover, actual storage sizes,
request gate isolation, real random-weight Qwen3 prefill/decode and HF
`generate()` parity between `retain` and `evict`. The CUDA BF16 integration test
is skipped explicitly when CUDA is unavailable.

The original v1 CPU smoke is preserved at
`agent_doc/results/far_skip_20261025/cpu_smoke.json`.
With prompt 8192, near 1024, sink/window 128/128 and 10 appended tokens, logical
length is 8202 in both modes; physical KV length is 8202 vs 1290. Actual FP32 KV
storage is 33,595,392 vs 5,283,840 bytes (84.27% lower for this one layer).
That receipt describes v1's variable-sized cache, not the current ring capacity.
The optimized implementation's paired timings, memory tradeoffs, source hashes
and operator profile are in `agent_doc/results/far_skip_optimized_20261025/`.
CPU timings must not be used as GPU speedup evidence.

To compare with v1 using identical inputs and check every output:

```bash
git worktree add --detach /tmp/far-v1 3c95ea167d66b462e2c64840adcc7c532a5115bd
python -m exp.bench.far_skip_compare --baseline-root /tmp/far-v1 --device cuda
```

Alternating paired runs include cache update, selection and attention; first
decode is reported separately from steady state. Use `--device cpu` only for
CPU results. `far_skip_profile` provides CPU operator/allocation accounting;
its event counts are not GPU launch counts.

The three subsequent iterations are recorded in
`agent_doc/results/far_skip_three_rounds_20261025/`. They include the slower
round-2 fixed prototype, round-3 comparisons, independent-size holdouts and final
source-bound receipts. `far_skip_graph_compare --baseline-root PATH` compares
fixed paths; add `--device cuda --capture` for real capture/replay. Without a
baseline root it compares fixed shapes against current dynamic eager (this mode
deliberately rejects `--capture`). CPU fixed-shape timing does **not** measure
CUDA Graph performance.
