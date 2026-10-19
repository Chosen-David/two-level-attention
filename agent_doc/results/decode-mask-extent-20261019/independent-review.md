# Independent review: decode mask extent

Date: 2026-10-19. Baseline: `5549339870004cb9ffb5dba6f20a88c89ac48547`.

## Verdict

PASS for the bounded metadata correctness fix. No blocking defect found in the six-line addition at `sparse_attn/ops/eager_decoding.py:355-360`. This is not a GPU/Triton execution or performance sign-off.

## Independent reasoning and call-chain review

- Llama3 and Qwen3 decode patches first unpad K/V when needed, prepare a single-sequence length array, call `indexer.prepare_mask`, and pass the returned mask/block size into `eager_decoding_attn`. DeepSeek V2 follows the same consumer chain with `allow_kv_hetero=True`. `Indexer.prepare_mask` returns `compute_mask` plus `get_block_size`; the consumer must enforce that this returned extent covers the actual K/V tensors.
- The dense consumer computes logits against all `k` tokens, then uses `torch.where` with an expanded block mask. At baseline a token-axis length of one broadcasts across all logits. An independent baseline import reproduced silent all-key attention: Q/K zero, V token values `[0,1,2]`, a one-slot true mask and block size one returned all ones instead of rejecting an incomplete mask.
- The new inequality uses only shape metadata and the existing validated positive block size. Rank validation already occurs earlier, so reading `mask.shape[-1]` is safe. V token length is already required to equal K token length. No allocation, value read, host/device synchronization, or kernel dispatch is introduced by this predicate for normal integer block sizes.
- Public dispatch and both direct dense/gather entry points call this validator first. Therefore the malformed short extent cannot reach either broadcasting or fused launch through these supported consumer entry points. The internal fused helper is not an independent raw block-mask API.
- The strict less-than comparison preserves complete masks and ceil-block/padding masks. Gather still routes excess padding to dense, where the token axis is cropped. The existing cu-seqlens contract is unchanged; these tests use `[0,T]` and do not claim to validate arbitrary malformed length arrays.

## Executed verification

Environment: PyTorch `2.8.0+cpu`; CUDA unavailable.

1. `python -m unittest discover -s sparse_attn/ops/tests -p 'test_eager_*.py' -v`: 19 tests passed, 5.866 seconds.
2. `PYTHONPATH=. python agent_doc/results/decode-mask-extent-20261019/independent_probe.py`: 48 independently constructed valid cases and 12 invalid extent cases passed. Details are preserved in `independent-probe.json`.
3. `git diff --check`: passed.

The independent oracle explicitly enumerates selected logical token indices for each query head and computes selected-token attention in float64; it does not reuse production dense/repeat/einsum/topk packing. It uses nonzero random Q/K/V and custom scale 0.37, two KV heads and six query heads, different selections per row, both shared and per-query-head selection, homogeneous and heterogeneous V widths, and a gapped mask with nonzero storage offset. This distinguishes head mapping and selection correctness from the existing uniform-zero-Q regression.

Valid cases cover T=5/65/16384, block sizes 1/2/64, exact coverage, a partial final block, and three additional true padding blocks. Both fast settings are exercised. Eight cases prove actual CPU torch-gather execution using a wrapped `topk` call, not merely `FAST=1`; long padded cases prove fallback without topk. Triton is explicitly disabled in the independent probe. Every result is compared to the independent oracle with atol/rtol 2e-6, and deferred error flags are checked.

Invalid cases cover zero slots, singleton-token and singleton-block undersizing, and a long mask one token short, through all three public/direct consumer entry points. Spies reject any attempted `topk` or `where` computation. The baseline singleton reproducer is loaded from the pinned commit in memory; current code rejects it with the targeted ValueError.

## Limits and nonblocking note

- No CUDA device, Triton kernel, sanitizer, GPU synchronization timing, model benchmark, quality metric, or speed claim was verified.
- The validator docstring's historical numbered list does not yet enumerate this new extent condition; the adjacent code comment and specific error explain it. This is a documentation nit, not a correctness blocker.
- No implementation, test, TASK, guide, or unrelated repository content was modified by this reviewer. Only this review and the independent probe/script result were added; no commit or push was performed.
