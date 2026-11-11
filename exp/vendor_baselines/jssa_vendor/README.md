# JSSA vendor — official vLLM patch + HF adapter

Source: jiwonsong-dev/JSSA commit
`ff65a6aec5c62534aef2a4d6f938c73546f74d92`; MIT license included.

`upstream/` is a read-only snapshot of the official repository
(`calibration/`, `benchmarks/`, `vllm_patch/`, README and LICENSE). The official
decode backend lives inside `upstream/vllm_patch/jssa_vllm.patch`; its newly
added source files were extracted, unmodified, to `upstream/patched_src/`
(`vllm/v1/attention/backends/jssa_attn.py` plus `triton_jssa_*.py` and the MLA
indexer).

The HF adapters (`qwen3_jssa_patch.py`, `llama3_jssa_patch.py`) cannot import
the official vLLM backend directly because it depends on vLLM internals. They
implement the official `_sparse_decode_forward` selection semantics in eager
torch (group-mean q, subspace projection, per-KV-head top-B, exact attention),
and are clearly labelled as a semantics-port, not the official vLLM/Triton
kernel. Running the real official kernel requires the L3 vLLM integration.
Calibration uses the official `upstream/calibration/jssa_calibrate.py`
(`compute_basis`, `M = TopEig_r(G_Qm/tr + G_K/tr)`).
