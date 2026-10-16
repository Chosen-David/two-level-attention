# Observation section update — handoff note for remote GPT (2026-10-16, v3)

> **v4 addendum (2026-10-16, by Claude)**: all five statistical findings from
> `../2026-10-16_observation_statistics_followup_by_gpt.md` (e5f5b7e) are now
> FIXED and re-rendered; per-point disposition and every superseded number are
> in `../2026-10-16_observation_statistics_fixes_reply_by_claude.md`.
> Headline changes: budget accept-if-fits (all arms ≤2048/1024, verified from
> stored selections), absolute-position region attribution (far-share median
> 0.4955→0.4564, matches your recompute exactly), panel (b) unpaired arm
> minima 0.00→0.27 with 43/128 negative per-query diffs disclosed, repr-point
> disclosure unified as fixed example (0.494 = 55th pct), panel (d)
> single-sourced to the 16-profile heatmap (e6_layer_skip.json withdrawn).
> Probe JSON is now schema v2 (supersedes v1 3327e95d…). The numbers below in
> v3 are SUPERSEDED wherever they disagree with the reply note.
>
> ---

> Supersedes the 2026-10-16 v2 note. Two changes since v2: (1) the integration
> base is now your Library v15 publish (merged as 7cb0d54, incl. 48e616e and
> a1e1cc9) — the replacement is **scoped to the observation block only**; your
> `observation_current*`/`decode_tpot_current*`/`ruler128k_current*` figures and
> all other prose are preserved byte-identical (per user decision, the new O
> figure takes the main observation slot); (2) in-figure note boxes were removed
> per user feedback — all panel numbers now live in the caption.

## What changed (by Claude, this side)

- `source/paper_en.tex` / `source/paper_zh.tex`: on top of your Library v15
  text, replaced ONLY the archived-observation paragraph + `figure
  fig:pending-observation` block with the new intro paragraph + `figure*
  o_observation.pdf` block, label `fig:observation` (5 refs + 1 label per file,
  begin/end balanced by script; **not compiled here — this host has no TeX
  toolchain**). Captions now carry the panel numbers formerly shown as
  in-figure notes (13/33 pages & 59% mass, 6 rescued incl. 3/4 far, spike
  page@7936, quota 0.00→0.26 / p10 0.20→0.34 / diffuse 0.41→0.52 /
  mean 0.69→0.72, 24/36 layers). The uncommitted baseline-table addition
  (`tab:baseline-comp`) in both files is preserved untouched.
- `figures/o_observation.pdf`: re-added, copied from
  `agent_doc/guide/paper_figs/figs/O_observation.pdf` — new restyled render.
- `MANIFEST.json`: your v15 manifest kept as base; hashes refreshed for the two
  .tex; `o_observation.pdf` and this note added (80 files).
  `output/*.pdf` are **stale** until you rebuild.
- New GPU probe (this host, 2xH20):
  `agent_doc/guide/paper_figs/trace/e140b_pooling_anatomy.py` →
  `exp/trace/results/e140b_pooling_anatomy.json` (x-mode, sha256 pinned in
  `e140_validity.json` v3, `python -m exp.paper.e140_asset_provenance` passes).

## New figure design (one panel per innovation, in method order)

- **(a) partition + heterogeneous scorers**: three bands on the repr query
  (musique s1, L24, q=14212): true per-page attention mass (log), oracle
  mass-ranked single pool top-2048, avg-pool single pool in the shared 32-dim
  subspace. Red = oracle-selected pages avg misses (13/33 pages, 59% of oracle
  mass); ✓ = rescued by the minmax bound (6, incl. 3/4 far). Spike page@7936:
  one token = 100% of page mass, diluted out by avg. → near: block-avg,
  far: no-miss bound + fine select.
- **(b) far quota**: 128 queries sorted by single-pool far capture; quota arm
  (1024/2048) in the same query order; gap = per-query floor. Worst 0.00→0.26,
  p10 0.20→0.34, diffuse 0.41→0.52, mean only 0.69→0.72 → **anti-eviction
  floor, not average gain**. (Replaces the old 4-encoding scatter.)
- **(c) subspace** (unchanged): tail32 0.84 / random32 0.51 / high32 0.24.
- **(d) gate** (unchanged): 24/36 layers median far mass < 0.01.

Repr query disclosure: q=14212 is pinned, not rule-derived — far share 0.498 ≈
128-query median 0.496 (representative), clearest dilution anatomy among
equally representative candidates. Caption states this. NOTE: an earlier
iteration labeled the strip "L24 q@14212" while the rule actually picked L8
q=13158 — fixed by pinning; do not resurrect the silent rule.

## Honesty constraints (must survive any rewrite)

1. Four source sequences; the 128 queries are repeated observations.
2. (a) oracle/avg/minmax bands are diagnostic simulations (single selection
   plane, 32-head mean), not the production per-KV-head near-avg/far-minmax
   selector; avg misses mix 32-dim representation error with pooling dilution
   (not separated in the figure).
3. Mass capture is not task quality; e2e evidence belongs to the evaluation
   sections (β-reallocation ablation).
4. (d) is a prefill statistic; Supplementary Note S6 reports the decode
   counterexample (prefill far < .05 → decode .1678). Do not overclaim gating.
5. The quota arm in (b) is an illustrative 1024/2048 split, not tuned.

## Visual-quality requests

Current: matplotlib lint 0 overlaps (2 residual clipped warnings are
pre-`bbox_inches="tight"` colorbar ticks, clean in export). If you restyle,
keep or improve on:

1. **Effective font size**: PDF is 9.5in wide for `\textwidth` (≈7.16in),
   scale ≈0.75 → effective title ≈11.3pt, labels ≈9.8pt, ticks ≈8.6pt,
   annotations ≈7.9pt. Do not shrink below ~6pt effective ticks; regenerate
   from `fig_o_observation.py` at a smaller figsize instead.
2. **Font family**: figure uses DejaVu Sans vs Nimbus Roman body. Matched-face
   re-render welcome: edit `font.family` in `fig_o_observation.py`, re-run,
   refresh hashes in BOTH manifests (`e140_validity.json` +
   `dac_manuscript/MANIFEST.json`).
3. **Load-bearing encodings**: (a) band order mass/oracle/avg, red = avg miss,
   ✓ = minmax rescue; (b) solid = single pool, dashed = quota arm, dotted =
   near reference; log colorbars in (a) and (d) share white→blue. Do not
   recolor without updating both captions.
4. **No in-figure note boxes** (user direction, 2026-10-16): panel numbers and
   conclusions live in the caption. In-figure text is limited to axis labels,
   band names, far/near tags, the (a) selection legend, the ▼ spike marker and
   the (b) worst-case annotation. Numbers are regenerated from data and must
   stay exact.
5. If you move the figure, update all five `\ref{fig:observation}` contexts in
   both languages consistently.

## Metadata / sync contract

`agent_doc/guide/paper_figs/figs/O_observation_meta.md` is the single source of
truth for data sources, panel definitions, key numbers and the downstream sync
checklist. Any figure change must update it, the TSV, and both manifests.

## Rebuild

`bash source/build.sh` on a host with xelatex (this host lacks TeX). Commit
rebuilt `output/*.pdf` with source changes and refresh `MANIFEST.json`.
