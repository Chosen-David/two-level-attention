# Current pooled configuration quality figure

`current_pooled_quality.pdf` is the manuscript figure (7 × 3.15 inches); the SVG retains editable text, and the PNG provides a preview. `current_pooled_quality_pdf_render.png` is an actual PDF rasterization inspected after the final generation.

Rebuild with `XDG_CACHE_HOME=/tmp/tla-current-cache python3 build_current_quality.py`. Requires Python and Matplotlib. All six source JSON snapshots are SHA256-checked before rendering. `provenance.json` records their tracked repository paths, Git blobs and source commit. No testdata input is used.

The CSV retains all 33 task comparisons and exact configuration keys. Decimal arithmetic computes differences before display rounding. Fifteen task differences are negative; the range is −18.00 to +24.33. Three RULER macros each contain 11 reported tasks with 100 examples/task/arm. Unreported QA tasks are excluded, not imputed as zero. No cross-length aggregate is shown.

Suggested caption: Current pooled Qwen3-8B RULER quality records at 32K, 64K and 128K. (a) Equal-task macro scores over 11 reported tasks per length. (b) All 33 task-score differences, pooled minus dense FullKV; dots denote tasks and diamonds denote equal-task macro differences. Vertical offsets separate nearby values only. Each reported task has 100 examples per arm. SQuAD and HotpotQA are unreported and excluded from means. Differences are computed before rounding. These descriptive records lack fully matched execution identities, prediction replay and paired confidence intervals; they do not establish non-inferiority or a quality–cost benefit.

Validation: verified six tracked inputs, SHA256 validation, exact task-set joins, source configuration/length assertions, 33 CSV rows and 128K delta +0.38, one-page PDF text extraction and actual rendered-PDF inspection. Final visual inspection found no clipped text or omitted plotted range. This is a new figure build, not a byte-identical reproduction of any earlier Overleaf figure. No repository files were modified.
