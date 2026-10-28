# Subspace before padding — 2026-10-27

Base main `8f213e48926431ea2d56cf3e7bc1a667d1105c74`; PR10 parent
`c96752e0d4c7441193a968bf94e21e4fd5bdad7e`. Isolated snapshot:
`candidate-subspace-pad-20261027`. Main indexer Git blob
`69eb1641deaebb96cdc8f17de6b5789e914088fe` was re-fetched and matched locally.
PR10's corresponding preparation differs only by a comment. CLAUDE, task,
current narrative and prior coverage reviewed; guide remains untouched.

## Mechanism and scope

`prepare_index -> _prepare_index_full -> _prepare_index_base` currently does
full-D `pad_tensor(k_stat)` followed by `pad_k[..., idx_sub]`. Full-stat Triton
fusion occurs only AFTER both copies. Its input is already the selected space.
Qwen D128/tail32 therefore zeros/writes a full-history 128-dimensional temporary
only to select32 dimensions. MLA D192/rope64 repeats full preparation each eager
decode step because both incremental paths reject D!=128. Qwen incremental
steady state does not repeatedly pay this initialization cost; graph warmup and
new-request rebuild do. Hybrid linear rows call it through the per-row indexer;
compact ring summary maintenance is a separate path and is not changed.

Move selection before padding when a subspace exists and MoBA is disabled.
`pad_tensor` still inserts zeros at the SAME packed sequence boundaries. The
existing `_fast_stat_full` kernel and all reduction orders are unchanged. MoBA
needs an independent full-D mean and keeps its original order/storage. Full
mode, L2 features/quantization and projection inputs preserve their contracts.
This reuses mature existing PyTorch indexing and Triton statistics rather than
introducing another specialized copy kernel. It reduces padding bytes; it does
not claim fewer launches. The selected source temporary remains live until the
function returns; overall allocator peak must be measured separately.

Alternatives considered: direct unpadded K into `_stat_full_kernel` could also
remove padding, since the kernel supports IDXSUB and masked zero loads. However
its wrapper uses floor(S/BS), and packed inputs need per-sequence block mapping:
for lengths33+31+7, existing output has3 blocks whereas ceil(total/64)=2.
Simply switching the pointer or checking tensor batch=1 breaks that contract.
Defer this larger change until a guarded packed-aware route and GPU pairs exist.

## Main/PR/concurrency coverage

Fresh read: PR11 `dc5acb8` adds a graph boundary-bypass guard and separate batch/
producer fixes. It does not optimize this preparation. Avoid duplicating it;
23:30 integration is a separate task. PR10 had no comments/reviews and remained
unmerged. `opt/l2-fine-topk`56d503a has identical base-preparation code to main;
`opt/ako-pool-score`a8eb0f0 differs there only in a CUDA capture guard. No competing
padding optimization was found. Mixed hybrid index/attention still loops rows
at `hybrid_batch_cache.py:201`, verified against current sources.

Capture geometry was also traced: `compute_mask_graph` bypasses cache lookup
for EVERY captured layer, contrary to its first-layer-only comment. The previous
request-cache patch fixes eager B>9 thrashing but not this. Reusing warmup cache
entries would omit live-length geometry producers from capture. Stream pointer
alone cannot distinguish successive captures on the same stream. Keep this as
an explicit capture-identity/lifetime candidate, not a claimed Graph speedup.

## Validation and limits

`cpu.txt`: **201 passed,24 skipped**, Torch2.8.0+cpu / Transformers4.56.2.
135 frozen-main dictionary pairs cover9 modes ×3 dtypes ×5 layouts: aligned,
partial tail, packed, effective length, non-contiguous last dimension. Modes:
tail/full/MLA/MoBA/quantized/full-L2/eager projection/lazy projection/static pair.
Twelve public prepare tests additionally run two requests ×17 growing steps
(408 dictionary pairs), exercising full->incremental and clear/reset.
All comparisons use rtol=atol=0. Adjacent hybrid whole-model, batch-indexer and
far-skip subspace regressions are included. Twelve new CUDA cases cover eager,
capture, four changed-input replays and fresh capture at another length; all
were skipped here, as were12 existing CUDA cases. No tolerance was relaxed.

`interpreter.json`:30 pairs run the existing Triton statistic kernel with old
and new input layouts (fp32/fp16 ×tail/MLA/full ×5 sequence layouts). Inputs,
strides, min/max/avg outputs all exactly match. Triton3.4 interpreter is CPU
execution, NOT CUDA correctness, BF16 validation or performance evidence.
No kernel body changed. `storage.json` observes actual CPU padding backing:
for S1025,Hkv8,bf16, tail buffer2,228,224 ->557,056 bytes; MLA3,342,336
->1,114,112 bytes; full/MoBA unchanged. At128K,Hkv8,bf16, the corresponding
padding capacities are **formula-only**256->64 MiB and384->128 MiB.
These are one temporary buffer, NOT KV residency savings or measured GPU peaks.

E127/indexer and batch gates were invoked but fail because CUDA is unavailable;
the partial local snapshot also lacks the optional static layer-mask fixture.
The benchmark refuses CPU fallback. See `gpu-unavailable.txt`. Evidence is
pending actual GPU gates, timing/allocator measurements and independent result
verification; self-checks are not independent scientific acceptance. No new
quality result or strict-selector acceptance is claimed.

## Remote paired commands

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_subspace_before_pad.py tests/test_hybrid_batch_cache.py tests/test_tli_batch_parallel.py tests/test_far_skip_subspace.py
python -m exp.bench.subspace_pad_compare --dtype bfloat16 --lengths 32769 131073 --repeats 20 --out /tmp/subspace-pad-bf16.json
python -m exp.bench.subspace_pad_compare --dtype float16 --lengths 32769 131073 --repeats 20 --out /tmp/subspace-pad-fp16.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

Benchmark uses exact frozen main preparation and candidate, alternates arms,
checks identical outputs, requires the fused statistics implementation and
records raw CUDA event/wall times and peak allocated deltas with source hash.
It measures full preparation including metadata synchronization; setup is outside
the timed region. It is not production TPOT. Remote end-to-end measurement must
separate Qwen initialization/capture from incremental decode and MLA full rebuild.
Reference function SHA256 before making its utils import absolute:
`ea7d28c043e781fe2dd08569dcd944a2995b24d1c564cdbe3b8985e52006d0d1`.

Only PR10 is updated; main is not pushed or merged. Roll back this preparation
reordering independently if paired GPU results regress; parent c967 remains the
prior baseline. All earlier PR10 work and latest main changes are preserved.
