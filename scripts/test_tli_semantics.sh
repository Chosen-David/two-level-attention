#!/usr/bin/env bash
# Run from any directory; use the project's existing environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:---auto}"
case "$mode" in --auto|--cpu|--cuda) ;; *) echo "Usage: $0 [--auto|--cpu|--cuda]" >&2; exit 2;; esac
python_bin="${PYTHON:-python}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$mode" == --cpu ]]; then export CUDA_VISIBLE_DEVICES=""; fi
"$python_bin" - "$mode" <<'PY'
import sys, torch, pytest
print('Python:',sys.version.split()[0], 'Torch:',torch.__version__, 'CUDA:',torch.cuda.is_available())
if sys.argv[1]=='--cuda' and not torch.cuda.is_available():
    raise SystemExit('ERROR: --cuda requires a usable GPU; CPU skips are not GPU acceptance.')
if torch.cuda.is_available():
    import triton
    print('GPU:',torch.cuda.get_device_name(), 'Triton:',triton.__version__)
else:
    print('CPU validation only: CUDA memory safety and capture/replay remain UNVERIFIED.')
PY
"$python_bin" -m pytest -q tests/test_paged_region_cache.py tests/test_frozen_region_runtime.py tests/test_paged_region_tli.py
