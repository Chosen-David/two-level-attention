# Compact far skipping and KV eviction

This opt-in Qwen3 decode path omits far keys from **page statistics, L1 scoring,
L2 scoring, and final attention**. It bypasses the full-history index/mask path
and the dense fallback in `eager_decoding_attn`. It uses eager PyTorch selection
and gathered SDPA with bounded tensor shapes, not fused selection or CUDA Graph.
GPU latency and pretrained-model quality have not been validated.

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
`DynamicCache`, batch size 1, single-token decode, and no padding. SDPA with an
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
attention. `get_seq_length()` still returns logical `S`, so HF generation and
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

`FarSkipWorkspace` is per-request, append-only, sequential, single-stream state.
It must not be shared between requests or used after modifying existing K or
dimension indices. The model bridge resets it at prefill and `indexer.clear()`.
Selections returned with an explicit workspace borrow its position buffer until
the next selection; clone IDs if they must outlive that call. This eager version
still has multiple framework kernels: it does not claim a fused L2 kernel or
zero temporary allocations, and CUDA Graph remains unsupported.

Eviction is irreversible within a request. Changing the gate back to far-on,
changing geometry/cache mode, cropping/rewinding, beams/batching, offload, and
legacy tuple export are rejected. Start a fresh `DynamicCache` for a new
request. A compact layer's `reset()` clears its state and requires prefill and
calibration again. Cache sharing/serialization and speculative decoding are not
supported. StaticCache, CUDA Graph, chunked/sparse prefill, projection,
per-query-head selection, clustering, async top-k, sigma and MoBA combinations
are outside this implementation and fail explicitly.

## Validate and measure

```bash
python -m pytest -q tests/test_far_skip_compact.py tests/test_far_skip_optimized.py
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
