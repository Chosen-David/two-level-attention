# Single-row output assembly — 2026-10-27

Main8f213e48926431ea2d56cf3e7bc1a667d1105c74; PR10 parentebacac00ea994795e8393ddb7ed5bf54206a961b.
Read applicable CLAUDE, TASK and previous gate/build-lifetime evidence. No ref/review
changes; concurrent opt/l2-fine-topk56d503a owns L2 work. This run analyzes a different
main path: try_hybrid_batch_decode -> row workspace/linear SDPA -> torch.cat(outputs)
-> qwen compact.reshape.contiguous -> o_proj. Main builder blob579475722e2324b2d22e8f39b2c825b952ab6578.
PR10 retained the same unconditional output cat. Both ring and linear row attention
return newly produced SDPA output, not a persistent scratch alias. Downstream does
not mutate that result. For B1, concatenating a singleton list adds a redundant copy.

Change: return the sole output at B1, retain one torch.cat for B>1. Branch depends
on immutable row count, safe for capture/recapture; does not inspect device alive.
No new kernel/library/state. Reject replacing B>1 cat with B Python copy_ calls:
more launches, no demonstrated advantage. A custom fused attention writer could
avoid all row outputs, but requires a new attention/numerical contract and GPU data.
Standard Torch output reuse is the minimal mechanism here.

EOS audit: current main and PR hybrid selector/attention still run dead rows;
only KV writes are frozen. Host branches would be baked into captured graphs.
Do not silently drop dead-row outputs or alter fixed batch order. Device guarded
full kernels/active-row graph variants require separate integration; no such claim.

Validation:16 CPU passed,8 CUDA skipped (cpu.txt). Four new paired CPU cases run
36 steps on actual tiny Qwen models with B1/2 and ring/linear modes, EOS, live-row
wraps and saved-output lifetime. Frozen previous function differs only in final
cat. Exact logits rtol=0/atol=0; output cat calls B1 1->0, B2 remains1. Existing
hybrid tests cover B1/2/8/32 and ragged requests. Two new GPU cases compare old/new
whole-model capture/replay, EOS and fresh requests at two lengths, not executed.
Syntax passed. CUDA benchmark explicitly refuses CPU fallback (gpu-unavailable.txt).

No GPU latency, allocator or model-quality result. Mechanism removes one tensor
assembly operation per hybrid layer/step at B1; no claim that this dominates TPOT.
Illustrative H32/D128/bf16 output is8192 bytes, not measured GPU traffic. Unchanged
B>1 route and saved outputs remain valid. Full model/GPU acceptance pending.

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_hybrid_single_output.py tests/test_hybrid_batch_cache.py
python -m exp.bench.hybrid_output_compare --steps 100 --repeats 10 --out /tmp/output-paired.json
python exp/bench/test_e127_gates.py indexer
python exp/bench/test_tli_batch_gates.py
```

The CUDA microbenchmark alternates arms and reports raw event/wall samples at
B1/2/8/32 for output assembly only. Direct B1 arm performs no tensor work, so tiny
event measurements mostly reflect timer/host-launch gaps; never report as model
speedup. Pair full-model eager/Graph TPOT on the target GPU before promotion.
