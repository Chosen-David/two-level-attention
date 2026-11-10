# TL-ASYNCTLS-INSTALL-20261110

Date: 2026-11-10
Status: CPU-validated; draft PR, GPU acceptance pending.
Tested base: 6c24eb8d0d9bf38203e94c8763b842b50afec7f3.

## Confirmed problem and bounded repair

Qwen3 AsyncTLS installation mutates early layers before discovering a missing or extra channel layer. It accepts empty, repeated, negative or out-of-range channel indices; Llama rejects some of these only after modifying earlier layers. Reinstalling either adapter overwrites the original dense forward. Dictionary channels with missing layer IDs are silently compacted and relabeled.

Preflight the complete native layer-ID set, channel count, per-layer unique in-range indices and existing installation before writing any module. Retain legacy numeric-suffix dictionary keys, while rejecting missing or repeated IDs. Bind Qwen channels to native layer_idx, matching ordinary HF model order without relying on registration order. Reject boolean channel indices.

This changes installation safety only. Scoring, calibration values, INT4 arithmetic, GQA aggregation, budgets and prefill/decode routing are unchanged for valid calibrations. PR41's partial-block scoring repair is separate and is not included. The referenced calibration generator / upstream source is not present in this checkout, so this is not an official-method equivalence claim.

## Reproduction and verification

Real CPU Torch 2.8.0+cpu, Transformers 4.56.2, Python 3.12.

- New suite on pristine base: 21 failed, 11 passed.
- Candidate new and adjacent suites: 52 passed normally and under `python -O`.
- Covers unchanged model state and dense output after rejected install, malformed channel sets, repeated install, registration-order independence, native layer IDs, all supported channel formats, and full-budget Qwen3/Llama decode versus dense.
- Parent independent read-only review requested preservation of numeric-suffix keys; compatibility and duplicate-ID tests added and passed.
- `pip check`, `py_compile`, `git diff --check`: pass.

Commands:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/test_asynctls_install_preflight.py tests/test_llama_vendor_adapters.py tests/test_baseline_integration.py
OMP_NUM_THREADS=1 python -O -m pytest -q tests/test_asynctls_install_preflight.py tests/test_llama_vendor_adapters.py tests/test_baseline_integration.py
```

Required repository gates are not passed: E127 requires CUDA; the retained batch gate imports deleted `sparse_attn.indexer.tli_indexer` (README's renamed gate is absent). No GPU, model-quality corpus, CUDA Graph or performance run. No claim of changed historical valid-calibration scores or speedup. No history or guide edits.
