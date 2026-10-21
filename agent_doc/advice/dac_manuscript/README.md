# Bilingual DAC research manuscript

The current research candidate uses sink4/SWA128 protection, shared low-frequency scoring coordinates and explicit near/far page and token budgets. The architecture separates boundary-page eligibility from protected final tokens and preserves a single chronological page sequence. The Qwen3 observation uses missed-reference-mass diagnostics and retains the bounded hybrid scan negative result. Historical measurements retain their own configurations; matched quality and complete-request cost remain open.

The English paper contains six content pages plus a references-only page; the Chinese companion contains five content pages plus references. DAC 2027 permits at most six content pages and one references-only page: https://dac.com/2027/program/research-manuscript-submissions . Page-count compliance does not establish full submission readiness.

Rebuild figures and both main texts/supplements with `bash source/rebuild_current.sh`. Dependencies are Python3 with NumPy/Matplotlib/Pillow, Inkscape, XeLaTeX/TeX Live and bundled fonts. The editable 17-slide presentation is synchronized by `python3 sources/sync_integration_ppt.py`; export it with LibreOffice/soffice after sourcing `fonts/env.sh`.

This public directory contains the manuscript, editable figure sources, published plotting inputs and presentation dependencies. It contains no private review history, recovery archive or one-time Word transfer payload. The Word reading copies remain separate deliverables. Library synchronization is tracked separately and is not implied by this repository update.
