# NFA paged cache reference

This opt-in eager B1 implementation keeps the existing frozen ring cache as its
correctness and performance reference. It does not change the default cache.
No performance or accuracy improvement is implied by the data structure.

`PagedRegionCache` uses request-local logical pages, each mapped to independently
allocated K/V tensors. K and V may have different channel widths. Aging changes
region boundaries; no old KV is copied from SWA to Near or Near to Far. The first
prefill copies the complete native KV into pages and returns the original KV to
full attention. That conversion overhead must be included in measurements.

| Frozen gate | Retention | Attention visibility |
|---|---|---|
| Keep | all pages | Sink, Far, Near, SWA |
| Near only | all pages | Sink, Far, SWA; aging restores Far visibility |
| Far only | Sink and Near/SWA pages | Sink, Near, SWA |
| Both | Sink and SWA pages | Sink, SWA |

Mixed pages are retained while any token needs storage. The inactive positions
inside them cannot participate in attention. A wholly unnecessary page releases
its tensor references to the Torch allocator and its integer ID to the free-ID
heap. This reduces active KV allocations; cached allocator reservations are not
claimed as released driver memory. Report inactive page slots separately.

The adapter reuses `TLIIndexer.compute_score/compute_mask` on low-dimensional
logical scratch to match the current tail-subspace selector. Final CUDA attention
reads selected full-dimensional K/V directly via page pointer tables. It does not
construct a contiguous full-width historical KV buffer. Scratch, table creation,
selection-index materialization and host synchronization remain real overheads.

The initial implementation is unpadded single-request/single-token eager decode,
with one-shot dense prefill and immutable gates. It has no Graph, batching, beam,
rewind, prefix sharing, quantized KV or offload support. GLM's expanded unequal
K/V widths are representable, but real GLM deployment requires a separate smoke;
this is not a latent-MLA compressed-cache implementation.

The isolated correctness and ABBA model scripts are
`exp/bench/check_paged_region_kernels.py` and `check_paged_region_model.py`.
The model script uses identical known decode input replay for each arm and never
labels it generated-answer throughput. Both arms warm up before measured ABBA
cycles. Compare every forced gate state, retain failures/slowdowns and do not
replace the ring default unless evidence supports that choice. Current measurements
remain pending independent acceptance in `agent_doc/results/nfa_paged_cache_20261030`.

CUDA pages are owned by the stream that first writes them. Updates, freezing and
pointer-table reads on another stream are rejected before state mutation; stream
handoff and CUDA Graph lifetime management are not implemented.

The TLI adapter matches the ring reference's activation-dtype Q scaling. For
fewer than 16,384 visible tokens, it additionally rounds normalized probabilities
to activation dtype, matching ring's dense arithmetic. This requires a second
page-read pass; its cost is included in timing. The standalone kernel's default
remains FP32 online softmax for comparison against SDPA. Neither setting promises
bitwise equivalence across reduction trees.

## 四态硬门与异常恢复

- Both-skip 的合法支持集只剩 Sink+SWA，直接构造受保护集，不创建 tail32 投影，
  不调用 L1、L2 或选择器 top-k。最终 attention 仍按页表直读这些物理槽位。
- 冻结后的通用 HF `Cache.update` 会拒绝 decode，防止 native attention 把单个新
  token 错当完整历史；只有显式的直接分页消费者可以调用
  `update_for_paged_decode`。
- 页写入在全部 K/V 拷贝成功后才提交逻辑长度和物理回收。中途异常会恢复被覆盖
  槽位、空闲页 ID、页表缓存与计数器，调用方可用同一位置安全重试。
- `check_paged_region_model_correctness.py` 在 freeze 前后读取 CUDA
  `memory_allocated`，并要求 Paged 臂的下降量严格等于实际整页 KV 释放字节；
  不把 caching allocator 的 reserved memory 或理论 mask 字节当成显存释放。
