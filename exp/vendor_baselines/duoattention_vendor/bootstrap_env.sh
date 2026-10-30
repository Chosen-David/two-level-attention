#!/usr/bin/env bash
# Linux x86_64 only. Installs packages; never executes a GPU/model workload.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install trusted uv first (validated with uv 0.12.23)' >&2; exit 1; }
export UV_PYTHON_INSTALL_DIR="$here/.cache/python"
export UV_CACHE_DIR="$here/.cache/uv"
export HF_HOME="$here/.cache/huggingface"
export MPLCONFIGDIR="$here/.cache/matplotlib"
export XDG_CACHE_HOME="$here/.cache/xdg"
export CUDA_VISIBLE_DEVICES=''
marker='duoattention-fe93c314-py310-cu121-v1'
if [[ -e .venv ]]; then
  [[ -f .venv/.duo-environment && "$(cat .venv/.duo-environment)" == "$marker" ]] || { echo 'Unowned .venv: refusing modification' >&2; exit 1; }
else
  uv python install 3.10.22 --no-bin
  uv venv --python 3.10.22 .venv
  printf '%s\n' "$marker" > .venv/.duo-environment
fi
.venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,10) else "Python 3.10 required")'
uv pip sync --python .venv/bin/python --require-hashes requirements-cu121.lock
.venv/bin/python verify_source.py
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e upstream
uv pip check --python .venv/bin/python
.venv/bin/python check_environment.py
