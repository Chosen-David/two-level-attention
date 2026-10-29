# TL-TEST-ENV-20261029
User requests help unblocking remote pytest execution without disturbing ongoing operator work.

Added scripts/setup_test_env.sh. Default target is repository .venv; PYTHON selects an existing GPU interpreter. Existing environments are never uv-synced. If pytest imports, reuse it; otherwise install pytest==8.4.2 via uv pip or the selected interpreter's pip. --check makes no installs. --require-cuda checks Torch CUDA availability and Triton presence; it does not certify kernels, sanitizer availability or performance.

For a missing default .venv, bootstrap from existing pyproject using uv and TLI_BASE_PYTHON (default3.11). Use --frozen only when uv.lock exists. Current main has no tracked uv.lock; its host-specific flash-attn wheel must be accessible. Fresh full GPU environment creation was NOT run here. No virtualenv files, model weights or GPU stack binaries committed.

Actual local validation: bash -n passed; uv created isolated /tmp/tli-pytest-bootstrap-check with Python3.12.14, then script installed pytest8.4.2 successfully. Four exit checks passed: existing pytest check=0, absent CUDA with --require-cuda=1, nonexistent Python=1, unknown option=2. No Torch/Triton/CUDA available locally; no native tests or speed claim. The project's Python3.11 full environment still needs remote validation.

Remote usage on the operator working tree, after obtaining this script:
```sh
# Existing repository .venv:
bash scripts/setup_test_env.sh --require-cuda
PYTHON="$PWD/.venv/bin/python" bash scripts/test_tli_operator.sh --cuda

# Or explicitly use the interpreter that already passed native H20 smoke:
PYTHON=/absolute/path/to/gpu-env/bin/python bash scripts/setup_test_env.sh --require-cuda
PYTHON=/absolute/path/to/gpu-env/bin/python bash scripts/test_tli_operator.sh --cuda
```

scripts/test_tli_operator.sh is supplied by the operator branch/PR13, not current main. Do not switch away from or overwrite uncommitted operator changes to install this helper. The screenshot reports six native SIMT smoke cases passed; pytest, TC, sanitizer, Graph replay and performance acceptance remain separate. No remote shell access was exercised. Branch/PR delivery only, no main push or merge.
