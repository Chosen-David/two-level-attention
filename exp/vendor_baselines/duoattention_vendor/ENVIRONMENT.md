# Isolated environment: actual CPU validation, GPU validation pending

## Reproduce

On **Linux x86_64**, with trusted `uv` available (tested 0.12.23):

```bash
cd exp/vendor_baselines/duoattention_vendor
bash bootstrap_env.sh
```

This installs managed CPython **3.10.22** into local `.cache/python`, creates an
isolated `.venv`, and installs **requirements-cu121.lock** with SHA-256 checks.
It never installs into the project's main environment, runs a model, or invokes
a GPU workload. It downloads several GB of wheels. It refuses an existing
unmarked `.venv`; repeat execution is permitted only for its own marked env.
The legacy `create_venv.sh` remains a create-only alternative for an existing
Python3.10 interpreter. No environment binaries are committed to Git.

The lock was actually resolved and installed here, not copied from the paper.
The current local interpreter is in the delivery workspace, **not on the user's
remote GPU machine**. Source package is installed editable with no dependency
resolution/build isolation. `check_environment.py` uses CPU tensors, imports
and the shipped head-pattern file only; it does not load model weights.

## Verified version / ABI choice

- Python 3.10.22, Torch **2.4.0+cu121**, Torch CUDA build 12.1, C++11 ABI **false**.
- transformers **4.45.2**, huggingface_hub **0.25.2**, tensor_parallel **2.0.0**.
- Legacy `flashinfer` **0.1.6+cu121torch2.4**, official cp310 wheel linked by
  `https://flashinfer.ai/whl/cu121/torch2.4/flashinfer-python/`; downloaded hash
  matches the official index (`d7605fbe...`). Not replaced with modern API.
- FlashAttention **2.6.3**, official `cu123torch2.4cxx11abiFALSE` cp310 wheel.
  It imports successfully with the selected Torch build; CUDA execution remains
  untested. CUDA12.3 wheel naming is recorded, not relabelled CUDA12.1.
- Setuptools **75.1.0** is necessary for this tested configuration: the first
  resolution selected a release without `pkg_resources`, causing the pinned
  tensor_parallel import to fail. Pinning 75.1.0 repaired this actual failure.
- NumPy is constrained `<2` for this Torch generation; the exact resolved
  dependencies and artifact hashes are in the lock. These are **new local
  resolution choices**, not the authors' historical full environment.

`evidence/environment-check.json` records actual successful imports, ABI, CPU
arithmetic, Llama3.1 pattern shape `(32,8)`, source-defined sink128/recent256,
and 0.5 head sparsification. These pattern values are not new main-method
settings or evidence of quality. `pip-check.txt` records package consistency.
No full repository/model/GPU correctness or performance validation is claimed.

## Remaining Block-Sparse-Attention backend

The official DuoAttention source optionally imports `block_sparse_attn` for
its blocksparse streaming path. That backend has **not** been installed or run;
no fake/stub replacement is provided. Full training/that path is not ready.
The evaluation module imports successfully without it, which is not proof that
every forward path can run.

Use historical Block-Sparse-Attention commit
`453e1bfdc9a6bdaec19fd0f5466deb1ddeb2bda0` (2024-10-15), the latest commit
before the frozen DuoAttention release date. This is a provenance-based
candidate, not an authors-locked dependency. License: BSD-3-Clause.
Its CUTLASS gitlink is `a75b4ac483166189a45290783cb0a18af5ff0ea5`.
The inspected setup requires NVCC (>=11.6), Torch headers, compiler and CUTLASS;
this host has **no NVCC/CUDA development toolkit**. Runtime CUDA wheels are
not the development toolkit. The setup's prebuilt fallback URL points to the
FlashAttention repository, so it must not be assumed to supply this backend.

On an authorized compatible CUDA build host, inspect the fixed source and
submodule licenses, clone/check out that commit with its pinned submodule,
activate this `.venv`, then build using its official setup. Record toolkit,
compiler, build log and installed wheel hash before any GPU acceptance run.
Do not use `SKIP_CUDA_BUILD` as a fake successful backend installation.

QServe/quantized demo and xformers alternatives are separate optional paths,
not silently installed. No Qwen3 support or checkpoint-specific calibration is
added by this environment. Model/GPU execution requires a separately accepted
experiment protocol.
