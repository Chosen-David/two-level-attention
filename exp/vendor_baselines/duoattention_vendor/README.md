# DuoAttention — official source preparation

Official repository: https://github.com/mit-han-lab/duo-attention

Frozen commit: `fe93c314ae87306ef6629dc16713250b4718ffe7` (retrieved 2026-10-30).
`upstream/` contains unchanged source, scripts, model head patterns and configs;
MIT notice is preserved in `upstream/LICENSE`. `UPSTREAM.json` records every
included file's SHA-256 and every omitted path. Third-party Paul Graham essays
and the upstream figures/paper are omitted because their separate redistribution
rights were not verified. Thus this is not a claim of a complete byte-for-byte
mirror. Needle evaluation needs separately licensed input prose.

This delivery does NOT add a benchmark adapter or change any baseline semantics.
Official dispatch supports Llama/Mistral/Mixtral, not Qwen3. The shipped head
patterns are checkpoint-specific: no Qwen3 compatibility, calibration, matched
quality, GPU speedup, or runnable CUDA environment is claimed.

- Environment: `create_venv.sh`, `requirements-upstream.txt`, `ENVIRONMENT.md`.
- Paper retrieval and rights: `PAPER.json`, `fetch_paper.py`. The PDF is fetched
  to ignored local `.cache/`, not redistributed in this repository.
- Source verification: `python verify_source.py` (standard library only).

Original upstream README still contains commands for installing packages,
downloading models and running training/evaluation. They are preserved as
source documentation, not automatically executed or newly authorized jobs.
