# TL-GRAPH-RUNNER-OWNERSHIP-20261101

Status: host regression passed; native CUDA and full GraphReady integration pending.
Base: main 27b06b67cb7286923ed6b8b80bd6ef33657a57c0.
Source: sparse_attn/engine/graph_runner.py, blob c5cd309874309378828e0b43c5d9422b87fe6797.

## Mechanism and change

GraphRunner captures fixed [bucket,1] input/position buffers and model/cache device
pointers, but previously selected a replay solely by bucket. step(model_B, cache_B)
could prepare/commit cache_B while executing model_A/cache_A's graph and returning
its static output. Scheduler.cache could independently remain cache_A.
Failed recapture also left the old bucket graph available.

Bind each successful bucket capture to model/cache object identities and keep those
owners alive with the graph. Reject mismatched scheduler.cache on capture and step,
and reject changed captured owners before scheduler.prepare_step, metadata mutation,
copies or replay. Recapturing one bucket replaces only that bucket; failed recapture
clears its graph/output/owners and leaves the existing eager route available.
Same-object request changes do not invalidate this ownership check.

This is a correctness prerequisite, not a GPU performance optimization. It adds
Python identity checks, no new CUDA launches or synchronization. References are
bounded by configured captured buckets and deliberately retain captured owners;
old owners can remain alive until all their buckets are recaptured or runner destroyed.
This is not a measured reduction in allocated/reserved GPU memory.

## Scope and alternatives

Current GraphReady metadata still loops by layer/request/page, while its attention
kernel batches query/head work; the B1 compact paged path is separate. This patch
does not make the GraphReady selector complete or repair the previously recorded
all-invalid/padded-query req_idx=-1 load. Full-model capture remains pending those
contracts. No change to kernels, selected IDs, gate policy, GQA or numeric thresholds.
The tests emulate changed request IDs through B=32, not native EOS/ring-wrap behavior.

Identity guards do not detect replacement of tensors inside the same model/cache,
mode/config changes or mutation of public runner dictionaries. Owners must retain
fixed storage/layout/execution configuration until explicit recapture. Direct use
of returned CUDAGraph.replay bypasses runner validation. Normal API is runner.step.

Compared with automatic recapture, explicit rejection avoids hidden capture latency
and cannot silently enter the still-unvalidated GraphReady capture route. A
per-owner unbounded graph cache would retain obsolete KV arenas; per-bucket binding
keeps the existing bounded design. No new kernel or library dependency is needed.
PyTorch documents fixed pointer arguments and long-lived input/output references:
https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs

## Validation

Full production module loaded with host Torch/CUDA doubles, not an AST replacement.
Baseline: 9 methods run, 7 failing assertions including two recapture subtests,
0 errors, 1 CUDA test skipped. Candidate: 8 host methods pass ordinary and -O;
1 native CUDA test skipped because Torch/Triton/CUDA are unavailable.
Source hashes and result counts: agent_doc/results/graph-runner-ownership-20261101/validation.json.
Cases: swapped model, swapped cache+scheduled cache, scheduler/cache disagreement,
same-owner replay with request replacement and 1/2/3/8/17/31/32 rows, per-bucket
recapture, warmup/capture failure, owner lifetime/release, unchanged eager fallback.

Reproduce host and baseline:
```sh
python tests/test_graph_runner_ownership.py -v
python -O tests/test_graph_runner_ownership.py -v
git show 27b06b67cb7286923ed6b8b80bd6ef33657a57c0:sparse_attn/engine/graph_runner.py > /tmp/graph-runner-baseline.py
GRAPH_RUNNER_SOURCE=/tmp/graph-runner-baseline.py python tests/test_graph_runner_ownership.py -v
```
Baseline is expected to fail the ownership regressions.

Native command on a CUDA checkout:
```sh
PYTHONPATH=. python tests/test_graph_runner_ownership.py -v
```
Require NativeOwnership.test_replay_and_recapture_distinct_storage to PASS, not skip.
It uses real CUDAGraph and distinct device scalars, exact outputs 5/18/20 before/
after owner recapture and in-place content update. It deliberately does not invoke
the known-invalid GraphReady attention kernel. Full paged-model recapture,
heterogeneous requests, EOS/repeated ring wrap, sanitizer, strict-selector quality
and paired whole-decode TPOT/memory remain unverified. No GPU benefit claimed.

## Remote overlap and continuation

All 57 heads match preceding coverage before publication; all 56 development
ancestry/date/unique-commit and actual diff/task/experiment evidence reused.
12 historical ancestors are not transplanted; opt/l2-fine-topk is historical.
opt/ako-pool-score a8eb0f08 and e175-skip-ring-kv 9b6e792e remain unchanged
candidates with prior ABI/quality/native-validation constraints. Remote running
status is unknown. All 23 existing PRs open; reviews/issue/inline comments empty.
PR10 2449f569, PR12, PR13 and PR17 trees do not contain this new runner.
PR23 c15be813 contains exactly the unchanged main runner blob, so its constant
lifetime fix does not cover this defect. Independent change, no unmerged dependency.
Next: native ownership test, then resolve packed/padded query ABI before tuning
MAX_VISIBLE loop bounds or metadata/input transfer costs.
