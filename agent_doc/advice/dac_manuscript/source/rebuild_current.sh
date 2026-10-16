#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${PAPER_MPLCONFIGDIR:-$(pwd)/texmf-var/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$(pwd)/texmf-var/cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
"${PAPER_PYTHON:-python3}" figures/method_details/build_details.py
"${PAPER_PYTHON:-python3}" sources/build_e134_quality.py
"${PAPER_PYTHON:-python3}" sources/build_evidence_figures.py
DAC_FIG_LANG=zh "${PAPER_PYTHON:-python3}" sources/build_evidence_figures.py
bash source/build.sh paper_en paper_zh supplement_en supplement_zh
