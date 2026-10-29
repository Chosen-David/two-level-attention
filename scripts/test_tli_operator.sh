#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:---cpu}"
case "$mode" in --cpu) export CUDA_VISIBLE_DEVICES="";; --cuda) "${PYTHON:-python}" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"';; *) exit 2;; esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" -m pytest -q tests/test_partial_attention.py tests/test_operator_overlap_cuda.py tests/test_operator_overlap_schedule.py tests/test_compact_selection.py tests/test_phase4_decode_indexed.py tests/test_phase4_decode_indexed_e2e.py tests/test_serial_graph_align.py
