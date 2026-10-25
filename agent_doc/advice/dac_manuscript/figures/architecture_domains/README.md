# Selected-configuration architecture

`build_figure.py` is the only generator for `fig2_domains_{en,zh}`. It writes editable SVG, a JSON scene, PDF, PNG and `architecture_manifest.json` from the same scene. The physical canvas is 180 × 92 mm; declared text is 7.5–9.5 pt. Both languages describe the same selected configuration. Source/data identity and output hashes are recorded in the manifest.

Build with the manuscript's bundled font environment and Python with CairoSVG:

```sh
source ../../fonts/env.sh
python build_figure.py
```

The figure distinguishes numeric tensors (solid) from page masks/token IDs (dashed). Boundary pages enter page eligibility outside the ranked quotas; protected token IDs bypass selection and enter the final union. Original-dimensional Q/K/V enters final attention without implying a mandatory physical gather. The 892 nominal token slots assume disjoint four-token prefix and 128-token recent protection; arrays are schematic.

`build_boundary_architecture.py`, `boundary_scene_*.json`, old gate insets and any native PowerPoint objects are historical artifacts, not synchronized exports of this figure. The historical generator now writes into `boundary_archive/`, so it cannot overwrite the manuscript architecture.
