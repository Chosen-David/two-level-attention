# TL-JSSA-INSTALL-20261110

Date: 2026-11-10. Tested source: main 54a8e7166f963b3d254d10a73085b5d4cbbe26a9.

## Confirmed defects and scope

Qwen3 accepts a missing calibration layer, installs `jssa_M=None`, then crashes
at decode `None.to`. A late invalid basis shape leaves earlier Qwen3/Llama
layers patched even though enable raises; retry wraps the adapter again, losing
the original dense forward. Repeated successful enable also overwrites
`flash_forward`. Qwen3 maps bases by module traversal rather than native layer ID.
Malformed top-level checkpoint raises NameError inside its error formatter.

Preflight the complete layer-ID set, basis coverage/shapes and prior installation
before mutating any module. Bind basis by native `layer_idx`, reject repeated
installation without replacing its original forward, and fix the error message.
This adapter deliberately requires all layers calibrated. Missing-layer dense
fallback was claimed by a stale Qwen comment, but was not implemented; no silent
fallback is introduced. Valid standard-model selection math is unchanged.

## Evidence

Real CPU Torch 2.8.0+cpu, Transformers 4.56.2, Python 3.12. New suite on pristine
base: 11 failed, 5 passed. Fixed suite plus adjacent suites: 43 tests and 4
subtests pass normally and under -O. Covers missing/late-shape/duplicate-ID/count/
malformed inputs, exact dense-model output after rejected install, repeated
install rejection, native-ID basis binding under registration reordering,
Qwen3/Llama full-budget dense decode parity, and existing baseline cases.

```
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q \
 tests/test_jssa_install_transaction.py tests/test_llama_vendor_adapters.py \
 tests/test_baseline_integration.py tests/test_vendor_frozen_manifest.py
# Repeat with python -O.
```

Minimal baseline reproduction: two-layer Qwen3, M only for layer 0; enable returns
2 and decode crashes. M[0]=(2,4,8), M[1]=(2,3,8), rank=4; enable raises but layer 0
has already changed. New regression verifies no adapter attributes are installed
and the original model still produces exactly the same CPU logits.

## Boundaries and pending work

JSSA adaptation identity remains arXiv:2610.19026v1 as recorded in the adapter;
its original vLLM patch/calibrator lives under ignored third_party/JSSA and was
not available for independent source verification in this checkout. This change
is an installer safety repair, not a new official-method equivalence claim.
Adapter doc defines calibration as M[layer_idx]; tiny native model layer IDs and
binding are exercised directly. Valid scoring, budget, full-dimensional attention,
GQA aggregation and prefill/decode policy are unmodified. No historical score
is marked invalid solely from malformed-install failure cases.

Required E127 attempted: CPU-only Torch cannot initialize CUDA. Existing batch
gate attempted: imports removed sparse_attn.indexer.tli_indexer and fails.
These gates are not passed. No GPU, real corpus, Graph, performance or memory
measurement; no speedup claim. Remote CUDA/model tests remain required before
merging. PR41's AsyncTLS padding repair is separate and not imported.

## 14:10 continuation: merge current main safely

The prior section records the original HF4 run, not current verification.
PR42 had a TASK.md-only merge conflict with main 65ed618008cccc71e6e24b575d24cb6bb2e9d447.
Retain both task entries and all main additions/deletions. Production JSSA patch
and its regression suite are unchanged from the original PR; no other PR is mixed in.

Current environment: Python3.12, Torch2.8.0+cpu, Transformers5.22.2.
New installer suite on pristine current main reproduces 11 failed/5 passed.
Candidate installer+JSSA+frozen-manifest tests: 24 passed +4 subtests, 19 deselected,
normal and -O. Full four suites: 38 passed +4 subtests /5 failed. The five failures
are the existing AsyncTLS legacy global-channel fixture and Quest/SparQ HF5 API
mismatch, covered separately by PR43/44. Current main adjacent suite has these
same five failures; no claim that all repository gates pass.

Required E127: six failures (CPU-only CUDA and deleted indexer/API); retained TLI
batch gate imports removed sparse_attn.indexer.tli_indexer; README's named
NFA batch gate does not exist. No GPU, Graph, corpus quality, speed/memory, or
HF4 rerun in this continuation. No baseline semantics or result data changed.
Fresh fetch before publication verifies main and PR head unchanged. Independent
read-only review and publication/tree/CI receipt are recorded with PR metadata.
