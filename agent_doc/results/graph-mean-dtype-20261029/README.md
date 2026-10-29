# GQA mean dtype guard — 2026-10-29
Baseline main a2e7b67b4e5c131cfa169283f7c66d6fb8892b00; indexer blob e4d76e0bb1e81b358bb99247a684ae8c7a579a26.
Main's new full-width Graph scatter/softmax path calls _ako_gqa_mean_kernel for every contiguous softmax dtype with power-of-two G. The older eager call has the same missing dtype guard. Kernel explicitly casts acc/G to tl.bfloat16 before storing to OUT. FP16/FP32 output allocation does not undo that rounding. Distinct probability values can become ties and change downstream selection.
Fix both dispatch sites: require input dtype torch.bfloat16; retain existing Torch nan_to_num/mean fallback otherwise. BF16 dispatch/kernel and main's full-token normalization, safe scatter, GLM integration remain unchanged. No replacement reduction or loosened tolerance.

New-path performance analysis: sf_full has [H,S_cap+1] elements, score_full is its [H,S_cap] strided view, softmax materializes [H,S_cap], GQA mean [Hkv,S_cap], followed by masked-fill/topk. Full-width work is the new correct reference; do not restore tile-local normalization or omit protected-token denominators to save memory. A stride-aware compact globally normalized implementation needs fresh selection/rounding/GPU evidence. Existing compact PR13 targets overlap but does not contain this newer main dispatch. This guard fixes correctness; it can make non-BF16 slower than the invalid BF16-rounding path. No performance claim.

192 evaluations of actual caller AST conditions cover dtype, G1/2/3/4, Triton availability, contiguity and per-query mode.18 non-BF16 routes corrected; normal/-O identical. IEEE conversion examples: FP32 .2501/.2502 and FP16 .25/.25048828125 collapse to equal BF16 .25. These are rounding/dispatch evidence, NOT native GPU numerical or model tests. Inverted-guard negative fails; candidate/native harness compile. Torch/Triton/GPU unavailable locally.

Reproduce after checkout:
```sh
git show a2e7b67b4e5c131cfa169283f7c66d6fb8892b00:sparse_attn/indexer/tli_indexer.py > /tmp/mean-baseline.py
python agent_doc/results/graph-mean-dtype-20261029/probe.py /tmp/mean-baseline.py sparse_attn/indexer/tli_indexer.py
python -O agent_doc/results/graph-mean-dtype-20261029/probe.py /tmp/mean-baseline.py sparse_attn/indexer/tli_indexer.py
PYTHONPATH=. python agent_doc/results/graph-mean-dtype-20261029/native_check.py
bash scripts/test_tli_semantics.sh --cuda
python exp/bench/test_tli_batch_gates.py
```
Native script (unexecuted here) checks G1 exact dtype preservation with actual kernel/fallback and captured replay. Full-model/eager/Graph G4/G8, top-k ties, multi-batch/ragged/EOS/wrap/new-request capture and quality remain separate acceptance. No measured latency/memory/quality claim.

All25 remote branches compared to latest main;13 PRs/reviews/comments checked. Unchanged-head development evidence reused. PR10 2449f56 and PR13 aae80aa lack new main path; create a narrow latest-main repair rather than overwrite their indexers or reintroduce removed code. No dependency on unmerged PR10. Its later integration must preserve these dtype guards and new main semantics.
