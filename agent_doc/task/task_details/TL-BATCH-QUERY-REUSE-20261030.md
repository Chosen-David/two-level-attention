# TL-BATCH-QUERY-REUSE-20261030

Progress: main cd22430d8d2509bb414886420a5dec6945332f7c's exact batch producer independently gathered L1 and L2 queries even when their feature-index object is identical. Reuse the current invocation's read-only coarse projection in that case; preserve independent/full/qat paths, model-dtype scale rounding and native GQA reduction.

All 15 open PR heads and changed-file inventories were checked. None includes batch_candidate_producer.py; PR10 2449f56983d7e97c5fe1c4df300c2b50b0c16b81 predates this implementation. This narrow main-based branch has no unmerged dependency and does not replace the PR10 integration work.

Validation: 96 actual-method AST/NumPy before/after cases, two invocations each, pass normally and with python -O; unsafe unconditional-sharing mutant rejected. No Torch/Triton/CUDA available locally. Added native Torch unit/capture regressions are unexecuted here; no latency, peak-memory, model-quality or GPU-launch measurement.

Evidence and remote acceptance commands: [README](../../results/batch-query-reuse-20261030/README.md).

Plan: run native producer/consumer and exact Graph integration gates, then matched before/after GPU ABBA profiling and timing. Keep B32 projection coverage distinct from the default ordered attention B<=8 support. Mixed ring/linear batching, gate/EOS/wrap behavior is unchanged by this patch.
