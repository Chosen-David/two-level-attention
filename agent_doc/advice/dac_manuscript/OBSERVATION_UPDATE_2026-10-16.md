# Observation section update — handoff note for remote GPT (2026-10-16)

## What changed (by Claude, this side)

- `source/paper_en.tex` / `source/paper_zh.tex`: the Introduction's third
  paragraph and the old `fig:pending-observation` placeholder are replaced.
  New label: `fig:observation` (5 refs + 1 label per file, begin/end balanced,
  checked by script; **not compiled here — this host has no TeX toolchain**).
- `figures/o_observation.pdf`: new four-panel observation figure (2x2),
  copied from `agent_doc/guide/paper_figs/figs/O_observation.pdf`
  (renderer: same directory, `fig_o_observation.py`; data:
  `exp/trace/results/e140_o_trace_v2.json` + `e6b_far_profiles.json` +
  `e6_layer_skip.json`, all pinned in
  `agent_doc/guide/paper_figs/observation/e140_validity.json` v3;
  verify with `python -m exp.paper.e140_asset_provenance`).
- `MANIFEST.json`: hashes refreshed for the two .tex and the new figure;
  `output/*.pdf` are **stale** (Oct 15) until you rebuild.

## Narrative contract of the new observation block (please preserve)

One panel per design element, in the same order as the method section:

- (a) position segmentation → heterogeneous region scorers
  (near block-avg / far no-miss bound + fine select)
- (b) oracle single pool under-serves far (mean 0.69 vs near 0.93, diffuse
  0.41, worst 0.00); a reserved far quota (1024/2048) lifts worst case to
  0.26 → quotas are **anti-eviction floors, not average-case gains**
- (c) equal-width subspace arms: tail32 0.84 / random32 0.51 / high32 0.24
  (metric: selected mass over reference top-K mass, NOT full attention mass)
- (d) per-layer prefill far mass: 24/36 layers median < 0.01, per-sequence
  12–32 far-light layers → motivates the per-layer far gate (D′)

Honesty constraints that must survive any rewrite:

1. Four source sequences; the 128 queries are repeated observations.
2. (a)(b) are **oracle** quality-ranked simulations (upper bound), not the
   production minmax/avg scorers.
3. Mass capture is not task quality; e2e evidence belongs to the evaluation
   sections (beta-reallocation ablation).
4. (d) is a prefill statistic. Section V already reports a counterexample
   (prefill far mass < .05 → decode .1678) and gate quality losses; the
   observation text explicitly foreshadows this. Do not let the figure
   overclaim gate effectiveness.
5. The quota arm in (b) uses an illustrative 1024/2048 split, not a tuned
   method parameter.

## Visual-quality requests (the part you asked us to specify)

Current state: the figure passes the repo's matplotlib lint (0 overlaps;
3 residual clipped warnings are pre-`bbox_inches="tight"` colorbar ticks,
clean in the exported PDF/PNG). If you restyle, please keep or improve on:

1. **Effective font size**: the PDF is 9.5in wide for `\textwidth`
   (≈7.16in), so scale ≈0.75 and effective sizes are title ≈11.3pt,
   labels ≈9.8pt, ticks ≈8.6pt, annotations ≈7.9pt. Do not shrink the
   figure further without regenerating it from `fig_o_observation.py`
   at a smaller figsize — scaling down the existing PDF below ~6pt
   effective tick size is not acceptable.
2. **Font family**: figure uses DejaVu Sans (matplotlib default) while the
   manuscript body is Nimbus Roman. A matched-face re-render is welcome;
   if you do it, edit `fig_o_observation.py` (font.family) and re-run,
   then refresh hashes in BOTH manifests (`e140_validity.json` via the
   v3 current_assets flow + `dac_manuscript/MANIFEST.json`).
3. **Panel order and color semantics are load-bearing**: blue=hotpotqa /
   orange=musique in (b); open markers = quota arm; log colorbars in
   (a) and (d) share the same white→blue colormap. Do not recolor without
   updating the caption.
4. Keep the in-figure conclusion annotations (they are the
   observation→design mapping); wording may be polished but the numbers
   must stay exactly as computed (they are regenerated from data).
5. If you move the figure out of the Introduction (e.g., a dedicated
   Observations subsection), update the five `\ref{fig:observation}`
   contexts in both languages consistently and keep the honesty
   constraints above attached to wherever the figure is discussed.

## Rebuild

`bash source/build.sh` on a host with xelatex (this host lacks TeX).
Please commit rebuilt `output/*.pdf` together with any source changes and
refresh `MANIFEST.json` hashes for every touched file.
