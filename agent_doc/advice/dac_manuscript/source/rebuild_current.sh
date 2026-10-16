#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 figures/method_details/build_details.py
python3 sources/build_e134_quality.py
bash source/build.sh paper_en paper_zh supplement_en supplement_zh
