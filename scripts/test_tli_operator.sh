#!/usr/bin/env bash
# Stage 1 semantics; stage 2 IDs/partial kernels; stage 3 model + graph replay.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:---auto}"
case "$mode" in --auto|--cpu|--cuda|--compile|--interpret) ;; *) echo "Usage: $0 [--auto|--cpu|--cuda|--compile|--interpret]" >&2; exit 2;; esac
python_bin="${PYTHON:-python}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TLI_OPERATOR_MODE=off  # Each test explicitly controls the candidate route.
if [[ "$mode" == --compile || "$mode" == --interpret ]]; then
    exec "$python_bin" exp/bench/operator_kernel_checks.py "${mode#--}"
fi
if [[ "$mode" == --cpu ]]; then export CUDA_VISIBLE_DEVICES=""; fi
bash scripts/test_tli_semantics.sh "$mode"
"$python_bin" -m pytest -q tests/test_operator_pipeline.py
echo 'Correctness checks completed; skipped CUDA tests are NOT GPU acceptance.'
echo 'No speedup claim. See docs/operator_overlap.md for paired real-model timing.'
