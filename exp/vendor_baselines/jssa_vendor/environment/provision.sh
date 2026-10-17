#!/usr/bin/env bash
# Generates a reviewable recipe only. It NEVER executes installation commands.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run && "$1" != --check ) ]]; then
  echo 'Only --dry-run (default) and --check are supported; no automatic installation.' >&2
  exit 2
fi
python3 "$HERE/manage.py" --check
if [[ "${1:---dry-run}" == --check ]]; then exit 0; fi
cat <<'RECIPE'
# REVIEW ONLY, not executed. On a separately authorized idle host:
# 1. Choose a NEW writable directory and supported CUDA wheel variant.
# 2. Use Python3.12 to create an isolated .venv; do not reuse project CPU env.
python3.12 -m venv "$JSSA_ENV_DIR"
source "$JSSA_ENV_DIR/bin/activate"
# JSSA_VENDOR_DIR must reference this archived jssa_vendor directory.
# Clone only the pinned vLLM source to a new directory, then check/apply core patch.
git clone https://github.com/vllm-project/vllm.git "$JSSA_VLLM_DIR"
git -C "$JSSA_VLLM_DIR" checkout --detach 39910f2b25aacc09f5e7f166cdf0030b19f8b9e8
git -C "$JSSA_VLLM_DIR" apply --check "$JSSA_VENDOR_DIR/upstream/vllm_patch/jssa_vllm.patch"
git -C "$JSSA_VLLM_DIR" apply "$JSSA_VENDOR_DIR/upstream/vllm_patch/jssa_vllm.patch"
# STOP before installing: resolve/hash the exact wheel + full dependency set,
# record driver/toolchain compatibility, then review the resulting complete lock.
# Upstream installation mechanism (not a validated or hash-locked installation):
# VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=39910f2b25aacc09f5e7f166cdf0030b19f8b9e8 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 python -m pip install -e "$JSSA_VLLM_DIR"
# Calibration requires transformers==5.8.0 plus datasets/accelerate; resolve/pin
# them separately. Do not download data/weights or launch calibration/benchmarks.
# No JSSA_MLA=1, DeepGEMM cache cleanup, FlashMLA build or shared env mutation.
RECIPE
