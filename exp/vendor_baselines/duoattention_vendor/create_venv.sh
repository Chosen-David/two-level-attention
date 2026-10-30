#!/usr/bin/env bash
# Create an isolated environment only. Never install packages or touch a GPU.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON310:-python3.10}"
"$python_bin" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,10) else "DuoAttention upstream requires Python 3.10")'
venv="$here/.venv"
if [[ -e "$venv" ]]; then
  echo "Refusing to overwrite existing $venv" >&2
  exit 1
fi
"$python_bin" -m venv "$venv"
"$venv/bin/python" -c 'import sys; print(sys.executable); print(sys.version)'
printf '%s\n' 'Environment created, dependencies NOT installed. See ENVIRONMENT.md before installing CUDA packages.'
