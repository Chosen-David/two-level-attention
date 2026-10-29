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
