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
Normalization now uses the unique L1/bypass union with complete Sink/SWA tokens
in both eager and Graph paths. Final protected attention remains a separate
operation. Measurements below used their stated earlier snapshots; see the
integration note for the normalization-domain change.

Local H20/Qwen3-8B BF16, exact 32768 input/64 generated, B=1, two ABBA cycles,
three full warmups and three measured requests per process: serial Graph TPOT
21.6943 -> 20.9361 ms/token (1.0362x). Both arms generated identical tokens on
this prompt. Request wall 9055.19 -> 8976.90 ms is only a descriptive difference;
prefill noise and two cycles do not establish a robust end-to-end gain. Capture
was excluded by warmup, but per-request cache fill is included in request wall.
This is not an FA2/overlap/quality claim. Evidence: project-local
`agent_doc/results/operator_compact_integrated_20261029/paired-confirmed-domain`.

Opt-in candidate-domain selection: set `TLI_OPERATOR_MODE=serial` and
`TLI_SPARSE_SELECTION=1`. On the supported Torch 2.8.0 BF16 CUDA geometry
(aligned capacity, native generic softmax, power-of-two GQA group, K2=1024),
this removes the capacity-wide normalization, owner, ranking and middle-mask
buffers after candidate scoring. Other geometries retain the existing selector.
The flag stays off by default; overlap remains experimental.

The implementation preserves each query head's native reduction order and
BF16 rounding over unique L1/bypass and complete protected tokens. Normalization returns a canonical
slot per token; ranking reuses it and includes implicit zero-score tokens from
the entire eligible interval. For capacities below 65536, exact packed 32-bit
keys use tiled sorting and a top-1024 merge tree; other supported capacities
use the generic ranking path. Empty quotas and enough-positive-token cases
have separate output paths. No approximate threshold or tile-local softmax is
used. Keep the owning indexer alive as long as its captured graph: constants
are retained per geometry/mode to preserve captured addresses.

Independent H20/Qwen3-8B BF16 measurement, exact 32768 input/64 generated, B=1,
serial whole-forward Graph, two ABBA cycles and three warmups/three measured
requests per process: published compact normalization vs candidate selection
TPOT **20.9412 -> 20.1718 ms/token (1.03814x, 3.6743% reduction)**. All generated
tokens matched on this prompt; all 36 patched layers used the intended path.
Request wall 8977.87 -> 8964.57 ms is a descriptive 0.1481% difference, smaller
than prefill variation, and does not establish a request-total speedup. Graphs
were reused after warmup; these numbers do not measure recapture per new request.
Evidence: project-local `agent_doc/results/operator_sparse_model32k_20261029/`
(`paired-compact-sparse` and `independent-compact-sparse-review.md`).

Separate paired 32K/64-token H20/Qwen3-8B run (two ABBA cycles, three warmups
and three measured requests per process): FlashAttention2-configured eager
DynamicCache vs this serial Graph/static-cache/selector implementation measured
TPOT **36.7259 -> 20.1745 ms/token (1.8204x)** and request wall
**9844.83 -> 8929.33 ms (1.1025x)**. This compares complete execution strategies,
not attention kernels under identical Graph/cache conditions, and does not show
an overlap benefit. Tokens were stable within each arm but differed across
arms; this prompt's timing comparison does not establish quality equivalence.
Do not multiply this ratio by the separate compact-to-sparse ratio above.
Evidence: `operator_sparse_model32k_20261029/paired-fa2-sparse` and its
`independent-fa2-sparse-review.md` in the project result store. Timing records
retain their configured-backend status; separate profiler traces are identity
audits and their instrumented timings must not be included in the paired data.

Release regressions: 120 targeted CPU/CUDA tests pass on the release snapshot.
Selection checks include 56-case producer memcheck and 20-case
independent tiled-ranking memcheck/racecheck, with zero reported errors/hazards.
These checks do not close the separate batch/ring, overlap-profiler, direct-gather
buffering or Hopper-specialization work.

A separate Nsight identity audit on the same model/code/config recorded 126
single-token forward ranges. Each contained 36 FlashAttention split-KV kernels
and 36 combine kernels (4536 of each total), confirming FA2 decode execution
in that audit. This does not retroactively turn the paired runs into profiled
runs or remove their Graph/cache differences. Audit artifacts are under
`operator_sparse_model32k_20261029/backend-audit`.

Graph-quality integration note (2026-10-29): the graph-quality branch now
normalizes the unique L1/bypass union with the **complete sink/SWA protected
domain**, consistently in eager and Graph. Candidate-only selection consumes
that completed domain as well. The numerical/timing records above were produced
on their named pre-integration snapshots; do not reuse their speed or quality
conclusions for this changed normalization domain. New integration regression
evidence lives under the main project's `graph_g1_main_integration_20261030_0070`;
performance and broader model-quality validation remain separate requirements.
