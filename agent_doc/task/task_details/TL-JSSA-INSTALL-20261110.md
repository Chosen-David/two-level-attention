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
