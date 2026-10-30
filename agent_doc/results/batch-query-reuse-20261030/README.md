# Exact batch query projection reuse

Base: cd22430d8d2509bb414886420a5dec6945332f7c. Producer blob: 7b5d5fe0748e63c69bddf855c186144a5aae1179.

Actual route: exact_linear_batch_decode -> ExactLinearBatch.produce -> BatchedCandidateProducer.__call__ -> _consume_batched -> fine score / candidate normalization / rank / ordered attention. Production tail32 uses the same Python feature-index tensor for coarse and fine selection. The old method evaluated full[..., coarse_dims] and full[..., fine_dims] independently after model-dtype Q*scale and FP32 conversion. Both produce [B,Hq,32] FP32 buffers. The coarse, average and fine Triton kernels only load Q; their stores target separate OUT/MOUT tensors.

The patch shares qc only when fine_dims is coarse_dims. It does not compare device values or add synchronization, cache query data across calls, move scale rounding, or change GQA mean/ranking. Different objects (even equal-valued indices) conservatively retain separate projection. No new kernel or ABI is introduced; reusing the existing tensor is preferable to a custom projection kernel for this duplicate operation. Generic compiler CSE is not assumed for this explicit eager/capture path.

Scope: avoids one advanced-index operation and one projected-query temporary in the shared-index case. During Graph capture this is GPU work recorded in the graph, not just Python dispatch; native launch/time reduction is still unmeasured. At B8/Hq32/Dsub32 the eliminated tensor payload is 32 KiB, not an allocator peak-memory result. This is a small cleanup, not evidence that the remaining selector or mixed-cache bottlenecks are solved.

## Completed checks

- probe.py executes the actual baseline/candidate __call__ AST with NumPy tensor/consumer facades: 96 configurations x 2 calls, normal and -O pass. B1/2/8/32, FP16/BF16 rounding models, source layout variants, shared/full/distinct/equal-but-distinct/qat-style index arrangements; exact returned projection and coarse/average outputs, no query mutation, no cross-call reuse. 32 of the 192 paired calls remove one gather; the rest keep their original count. The unsafe unconditional-sharing mutant fails the fine-full shape contract.
- Production and native-test source compile with Python. This is not a Torch or GPU test.
- tests/test_batch_query_projection.py adds native Torch CPU/CUDA checks with mocked read-only consumers and two Graph captures/replays. Those tests and existing full native integration tests were NOT run locally because Torch/Triton/CUDA are absent.
- All 35 pre-existing remote heads and 15 open PR heads/timelines checked. No PR changes this producer; PR10/13 do not contain the file. Exact unchanged branch ancestry/diff evidence was reused. Physical-cache development is independent; no other branch is overwritten.

## Reproduce host check

```sh
python agent_doc/results/batch-query-reuse-20261030/probe.py
python -O agent_doc/results/batch-query-reuse-20261030/probe.py
```

## Pending native acceptance

On the repository's supported Torch/Triton GPU environment:

```sh
python -m pytest -q tests/test_batch_query_projection.py tests/test_batch_candidate_producer.py tests/test_batch_candidate_score.py tests/test_batch_coarse_score.py tests/test_exact_linear_batch_gpu.py
```

Retain zero-tolerance projection comparisons and existing integration gates. The new mocked-consumer capture test is not a replacement for the native kernels/model gate. Test B2/B4/B8 ragged lengths, page crossing, EOS/dead rows and recapture with the existing exact tests; B32 is projection-only coverage, not ordered-attention admission. Ring/gated paths are unaffected and receive no new performance claim.

For matched full-model comparisons, create detached worktrees at the base SHA above and this PR head, use the same inputs/model/environment, and run before/after/after/before on an otherwise idle GPU, retaining every result. The existing harness hardcodes the Qwen3-32B model path: it must be present exactly as configured. Its inputs need distinct original JSONL rows with index/input and supported token lengths (>=9217, length+generation<65536). Run the following command in each worktree, with a distinct output path per arm:

```sh
PYTHONPATH=. python exp/bench/check_quality_batch_qwen.py --input /path/to/original-ruler-inputs.jsonl --output /path/to/arm-result.json --batch 2 --max-gen 128 --cycles 2 --measure --legacy-batch-reference
```

Repeat B4/B8; B8's accepted setup requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, identically in both arms. Compare exact-batch candidate records across commits (not only each run's legacy-reference ratio); separate first-pair setup/compile from warmed timing. Profile separately with --eager-batch-reference --trace-dir /path/to/arm-trace and without --measure. Confirm the missing query gather in captured work, unchanged generated IDs and no extra copies. No GPU speedup, whole-benchmark selector quality, or resident-KV saving has been established by this change.
