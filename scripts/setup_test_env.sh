#!/usr/bin/env bash
# Run on the GPU host. Never copy a virtualenv between machines.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
check_only=0
require_cuda=0
for arg in "$@"; do
    case "$arg" in
        --check) check_only=1 ;;
        --require-cuda) require_cuda=1 ;;
        *) echo "Usage: PYTHON=/path/to/python $0 [--check] [--require-cuda]" >&2; exit 2 ;;
    esac
done
python_bin="${PYTHON:-$PWD/.venv/bin/python}"
if [[ -z "${PYTHON:-}" && ! -x "$python_bin" ]]; then
    if (( check_only )); then
        echo "Missing .venv; run setup without --check, or set PYTHON to the existing GPU interpreter." >&2
        exit 1
    fi
    command -v uv >/dev/null || { echo "uv is required to create .venv; alternatively set PYTHON to an existing environment." >&2; exit 1; }
    # Existing project configuration includes a host-specific flash-attn wheel.
    # Do not sync an existing GPU environment: that could remove local packages.
    echo "Creating .venv from pyproject.toml; its flash-attn wheel must be available on this host."
    if [[ -f uv.lock ]]; then
        uv sync --frozen --python "${TLI_BASE_PYTHON:-3.11}"
    else
        echo "No uv.lock in checkout: dependencies will be resolved locally (not a fully locked reproduction)." >&2
        uv sync --python "${TLI_BASE_PYTHON:-3.11}"
    fi
fi
command -v "$python_bin" >/dev/null || { echo "Python not found: $python_bin" >&2; exit 1; }
if ! "$python_bin" -c 'import pytest' >/dev/null 2>&1; then
    if (( check_only )); then
        echo "pytest is missing in $python_bin; run without --check to install it." >&2
        exit 1
    fi
    if command -v uv >/dev/null; then
        uv pip install --python "$python_bin" 'pytest==8.4.2'
    else
        "$python_bin" -m pip install 'pytest==8.4.2'
    fi
fi
"$python_bin" -m pytest --version
"$python_bin" - "$require_cuda" <<'PY'
import importlib.util
import json
import sys
report = {"python": sys.executable}
try:
    import torch
    report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
except ImportError:
    report.update(torch=None, cuda_available=False)
report["triton_installed"] = importlib.util.find_spec("triton") is not None
print(json.dumps(report, indent=2))
if int(sys.argv[1]) and not (report["cuda_available"] and report["triton_installed"]):
    raise SystemExit("Native GPU tests require CUDA-enabled Torch, a visible GPU and Triton. No GPU stack was replaced.")
PY
echo "Use this exact interpreter for tests: PYTHON=$python_bin bash scripts/test_tli_operator.sh --cuda"
echo "The operator script is present on the operator branch; setup alone is not GPU test acceptance."
