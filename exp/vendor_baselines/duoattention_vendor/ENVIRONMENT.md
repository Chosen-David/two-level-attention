# Environment preparation (not a validated runtime)

The frozen upstream README is the authoritative original recipe. It uses
Python 3.10, a Conda CUDA 12.4 toolchain and unpinned PyTorch, but a legacy
FlashInfer wheel index labelled CUDA 12.1 / Torch 2.4. These are not a complete
lock and must not be silently replaced by today's `flashinfer-python` API.

Run `bash create_venv.sh` to create a **new local `.venv`** with Python 3.10.
Set `PYTHON310=/absolute/path/to/python3.10` if necessary. The script refuses
existing environments. `.venv` is intentionally ignored, not portable Git data.
It does not install anything, query/use GPUs, download models, or run benchmarks.

Before claiming a runnable environment, on an authorized build host:

1. Resolve a compatible Python 3.10 / Torch 2.4 / CUDA toolkit and NVIDIA driver
   combination. Record exact wheels, versions, hashes, OS and compiler; do not
   modify the main project's environment. The current host has Python 3.12,
   not the required Python 3.10, so no `.venv` was claimed as created here.
2. Install the packages in `requirements-upstream.txt` into this `.venv` after
   resolving the unpinned dependencies against transformers 4.45.2 and
   huggingface_hub 0.25.2. This file is an upstream inventory, **not a lockfile**.
3. Build/install `flash-attn==2.6.3 --no-build-isolation` with the selected Torch
   and toolkit. Resolve the original `flashinfer` package from the upstream
   index `https://flashinfer.ai/whl/cu121/torch2.4/`; record its precise wheel and
   hash. Availability and ABI compatibility have NOT been verified here.
4. Run `.venv/bin/python -m pip install --no-deps -e upstream` from this directory.
5. Inspect/license-check and install Block-Sparse-Attention from
   https://github.com/mit-han-lab/Block-Sparse-Attention . HEAD observed on
   2026-10-30 was `49d6c39e4dc0303442cda3bb758b3925d4399c49`; it is a candidate
   dependency pin, NOT an upstream DuoAttention-tested compatibility promise.
   Do not build it until the toolchain and its own dependencies are checked.
6. Run pip check, import-only checks, then save `pip freeze --all`, wheel hashes,
   compiler/toolkit versions and the source commits in a local build receipt.
   CUDA execution and model quality/performance require separately authorized
   hardware testing; import success alone proves neither.

QServe/W8A8KV4 is a separate optional upstream demo environment, not this
baseline's default dependency set. No QServe or model download was performed.

Remaining blockers: compatible Python 3.10 build host, resolved complete
transitive lock, historical FlashInfer wheel/API compatibility, CUDA extension
builds and imports. A setup recipe is not an already-working `.venv`.
