# TL-NFA-STATS-AVERAGE-20261103

Source: recurring main/remote-branch performance audit. Base main
`ae69e995acd7e9423d4512f8277aa789a5b310a5`, selector blob
`1706d571eb750709d838301bbd9a46201a0678f0`. Serial implementation.
Status: host checks passed; native CUDA/Graph, full selector/model correctness
and performance pending. No performance or scientific verdict issued.

## Actual production mechanism

Qwen/GLM write current KV, call the current-query request-batched selector for
each layer, then consume selected physical indices in gather attention.
`_page_stats_all_layers` launches `_nfa_page_stats_kernel` over logical blocks
and request/KV-head rows, producing min/max/sum `[B,KT,Hkv,d]`. Skipped-region
and padded tokens are masked before K loads. Min/max include masked zeros;
the average divisor is the fixed block width, not the visible-token count.
This change preserves both details and the existing reduction tree.

Main stores sum in the KV dtype, then executes
`(k_sum.to(torch.float32) / bs).to(out_dtype)`. For BF16/FP16 this creates
three eager tensor passes (also recorded during CUDA Graph capture) over the
summary, although the FP32 sum was already live in the statistics program.
FP32 needs only the division pass because casts are no-ops.

Fuse normalization into the existing statistics store and allocate its third
output directly as `k_avg`. Preserve the intermediate low-precision sum cast:
`round_dtype(sum_fp32) -> float32 -> divide BS -> round_dtype`.
Do not change it to an FP32 mean: with 64 FP16 keys equal to 2048, the old sum
overflows to infinity while direct FP32 normalization yields 2048. Repairing
that inherited overflow would be a separate numerical/quality change.

No additional kernel or dependency; function return shape/dtype and consumer
stay identical. Reusing the existing reduction/store is simpler than a new
pointwise kernel, persistent summary cache, or compiler-wide fusion. All region
gates remain device-driven; no capture-time host shortcut or mutable cached
summary is introduced. Eager and Graph execute the same store expression.

Logical accounting only: N=B*KT*Hkv*d. Low-precision original normalization
passes read/write a total of 20*N bytes; fusion removes these separate tensor
passes. This is not measured HBM traffic, allocator peak or latency; cache
effects, compiler scheduling and graph memory reuse require native profiling.
No KV residency or release behavior is changed.

## Evidence, overlap, limitations

- Old wrapper regression fails with events `[stats,cast,divide,cast]`.
  New wrapper has `[stats]`; actual wrapper AST is executed with a tracing
  backend (not native kernels).
- Actual fused store AST executes against a NumPy cast facade for 165,549
  FP32 input sums across three output dtypes and BS=32/64/128: 1,489,941 paired
  scalar comparisons, exact non-NaN FP32 bit patterns including signed zeros
  and infinities, identical NaN masks. Inputs include all 65,536 BF16 patterns,
  100,000 seeded arbitrary FP32 bit patterns and 13 boundary values.
  This tests the store contract, not Triton reductions or CUDA conversions.
- Two host test methods pass normally and under `python -O`. Python source
  compilation passes. Native test skipped: no Torch, Triton or GPU here.
- Added native baseline/candidate paired stats test for BF16/FP16/FP32,
  B=1/4/32, BS=32/64/128, d=31/32/64, layer offset, permuted request slots,
  empty/unequal lengths, mixed four gates, moving boundaries, page-table reuse,
  eager and captured replay. Its 27 configurations are unexecuted here.
  Full cache lifecycle, EOS, seed use, recapture and model output acceptance
  are separate remaining gates, not established by a statistics-only test.
- Fresh PR #31 c48a50ab and #32 4dc6d7fe retain the old normalization;
  #10 actual 2449f569 has no current selector module. #31 coordinate pairing
  and #32 dead geometry removal are independent; preserve them when integrated.
  No unmerged dependency. Old hybrid/ring summary kernels use another ABI;
  no unverified branch transplantation.
- All 93 main Python files under sparse_attn/tests were fetched and searched:
  the private statistics kernel and wrapper are referenced only in this
  selector module. Public wrapper return values remain min/max/average.
- Initial all 66 heads/65 development branches and 32 PR heads unchanged;
  all PRs open/unmerged, reviews and repository comments empty. Exact dates,
  unique commits, ancestry and prior code/experiment evidence reused.
  opt/l2-fine-topk is a main ancestor; opt/ako-pool-score and e175-skip-ring-kv
  remain old hybrid/ring implementations. Git metadata does not establish live
  remote task activity. No new committed experiment evidence.

## Native acceptance commands

From this branch in a complete checkout with dependencies installed:

```bash
git show ae69e995acd7e9423d4512f8277aa789a5b310a5:sparse_attn/indexer/nfa_layer_batch_selector.py > /tmp/nfa-stats-before.py
NFA_SELECTOR_SOURCE=/tmp/nfa-stats-before.py python tests/test_nfa_page_stats_average.py AverageHostTest.test_wrapper_has_no_post_launch_tensor_pass -v
# The previous command intentionally fails on the old extra tensor passes.
PYTHONPATH=. NFA_STATS_BASELINE=/tmp/nfa-stats-before.py python tests/test_nfa_page_stats_average.py -v
PYTHONPATH=. NFA_STATS_BASELINE=/tmp/nfa-stats-before.py python -O tests/test_nfa_page_stats_average.py -v
python -m pytest -q tests/test_nfa_current_query_selector.py
```

Inspect skips; a skipped native test is not PASS. Require exact summary values,
min/max, selected IDs/counts and generated tokens; do not loosen tolerances.
Inspect generated code to confirm low-precision round trip survives compiler
optimization, especially FP16 overflow and BF16 subnormal boundaries.

For paired whole-model performance, use isolated main/candidate checkouts on
the same idle GPU, identical model/data/sample and explicit production config.
Alternate A/B/B/A for at least three rounds, eager and graph, B=1/2/4 and
32/64/128K where feasible. Run this command in each checkout with a unique
OUTPUT_JSON (MODEL_PATH/DATA_ROOT point to the same local assets):

```bash
python benchmark/efficiency/benchmark_nfa_end_to_end.py \
  --model-path "$MODEL_PATH" --data-root "$DATA_ROOT" \
  --output "$OUTPUT_JSON" --task niah_single_1 \
  --context-length 32768 --batch-size 1 --method nfa --execution graph \
  --sample-index 0 --input-provenance "$INPUT_PROVENANCE" \
  --max-new-tokens 128 --warmup-steps 3 --visible-len 8192
```

INPUT_PROVENANCE must accurately be `native-ruler` or
`repeated-needle-body-extension`. The entry duplicates prompts and supports
only B=1/2/4; it does not prove large heterogeneous batch support. At this base,
`nfa_args()` explicitly sets tail/ratio4 and K1=128/K2=1024. Supplement with
large/mixed-batch native profiling and EOS/slot reuse/recapture correctness.
Compare whole-selector/model latency and allocator peaks; profile statistics
and normalization kernels for attribution. Kernel-count/byte arithmetic is
not an end-to-end speedup. GPU acceptance is required before adoption.
