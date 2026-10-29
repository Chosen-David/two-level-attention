# Experimental compact decode and protected attention overlap

Default remains TLI_OPERATOR_MODE=off. Opt in with serial or overlap;
TLI_BATCH_INDEXER=legacy is required. Overlap starts a fused Sink/SWA partial
attention kernel before index preparation and joins after selected-token
attention. The two softmax states are merged exactly in FP32. Q scaling retains
model dtype rounding. Warm up before graph capture. Ring decode is unsupported
and fails explicitly. Mixed batches retain per-request scheduling.

Local H20 microbenchmark, FP16 B=1 Hq=32 Hkv=8 D=128, sink=4 SWA=128:
separate Sink/SWA plus merge vs fused protected plus merge measured 9.48→8.03us
SIMT and 8.44→7.01us TC, medians across 20 alternating rounds of 200 graph
replays. These narrow local measurements are not full-model or overlap speedups.
Full model quality, true direct-gather double buffering, streaming selection and
Hopper warp specialization remain outstanding. Pipeline stages=2 only requests
compiler software pipelining; default is 1.

Run PYTHON=/existing/gpu/python bash scripts/setup_test_env.sh --require-cuda,
then PYTHON=/same/python bash scripts/test_tli_operator.sh --cuda.

Integration validation: 124 regression tests passed on the isolated main-based
checkout; native compute-sanitizer memcheck: 63 passed, zero reported errors.

Selector follow-up: BF16-specific GQA mean fusion now requires BF16 at both
call sites. FP16/FP32 use the existing dtype-preserving reference chain. Removed
a discarded sentinel-score mask and SWA fill overwritten by the subsequent
middle-region mask. Dtype dispatch CPU checks and native G=1 eager/captured
checks pass for FP16/BF16/FP32; 22 targeted native regression cases pass.
These checks do not establish full-model quality or a selector speedup.

Experimental model entry routing: TLI_OPERATOR_BACKEND=simt|tc selects both
partial attention backends; TC requires fp16/bf16. TLI_OPERATOR_SELECTED_STAGES
accepts 1/2/3 for selected attention only. Defaults remain simt/1. Four native
SIMT/TC x stage1/3 fork/join graph replay tests pass; these options are not an
automatic tuning policy and do not establish model speedups.

Opt-in exact normalization: `TLI_COMPACT_NORMALIZATION=1` removes dense
query-head score/probability buffers for pinned Torch 2.8.0 BF16 generic CUDA
softmax geometry. Unsupported shapes retain the reference path. Warm up before
Graph capture; the extension requires a CUDA compiler. Per-KV-head owner and
ranking buffers remain full width; this is not yet fully candidate-domain Top-K.
Normalization follows the actual L1/bypass token domain; final Sink/SWA attention
protection is separate. Fine-scoring a protected superset does not imply those
tokens survive the eager reconstruction mask before softmax.

Local H20/Qwen3-8B BF16, exact 32768 input/64 generated, B=1, two ABBA cycles,
three full warmups and three measured requests per process: serial Graph TPOT
21.6943 -> 20.9361 ms/token (1.0362x). Both arms generated identical tokens on
this prompt. Request wall 9055.19 -> 8976.90 ms is only a descriptive difference;
prefill noise and two cycles do not establish a robust end-to-end gain. Capture
was excluded by warmup, but per-request cache fill is included in request wall.
This is not an FA2/overlap/quality claim. Evidence: project-local
`agent_doc/results/operator_compact_integrated_20261029/paired-confirmed-domain`.
