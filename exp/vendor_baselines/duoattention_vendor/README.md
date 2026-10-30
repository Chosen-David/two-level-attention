# DuoAttention — official source preparation

Official repository: https://github.com/mit-han-lab/duo-attention

Frozen commit: `fe93c314ae87306ef6629dc16713250b4718ffe7` (retrieved 2026-10-30).
`upstream/` contains unchanged source, scripts, model head patterns and configs;
MIT notice is preserved in `upstream/LICENSE`. `UPSTREAM.json` records every
included file's SHA-256 and every omitted path. Third-party Paul Graham essays
and upstream figures are omitted because their separate redistribution
rights were not verified. The user-supplied preprint PDF is archived separately
under `agent_doc/advice/other_paper_pdf/DuoAttention-2410.10819v1.pdf`, with
original rights retained; the code MIT license does not apply to the paper. Thus this is not a claim of a complete byte-for-byte
mirror. Needle evaluation needs separately licensed input prose.

This delivery does NOT add a benchmark adapter or change any baseline semantics.
Official dispatch supports Llama/Mistral/Mixtral, not Qwen3. The shipped head
patterns are checkpoint-specific: no Qwen3 compatibility, calibration, matched
quality, GPU speedup, or GPU-validated environment is claimed.

- Environment: `bootstrap_env.sh`, `requirements-cu121.lock`, `ENVIRONMENT.md`.
  Python3.10/Torch/legacy FlashInfer/FlashAttention were actually installed and
  CPU-import checked; BlockSparse and GPU execution remain unverified.
- Paper retrieval and rights: `PAPER.json`, `fetch_paper.py`. The user-supplied PDF is arXiv v1 (2024), not the ICLR camera-ready.
  The optional retrieval script still fetches to ignored local `.cache/`.
- Source verification: `python verify_source.py` (standard library only).

Original upstream README still contains commands for installing packages,
downloading models and running training/evaluation. They are preserved as
source documentation, not automatically executed or newly authorized jobs.
