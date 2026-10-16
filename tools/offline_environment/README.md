# Paper and registered-baseline offline preparation

Task `TL-OFFLINE-ENV-20261016`, based on main `88dcb39e3fd2e6d85126bd0610d36d0043783e9c`.
Integrated newer main `08ccb54627e7c456fb020d75a68a3ca040f46d2d` before publication;
repeated the full isolated rebuild against its updated introductions.
No model weights, GPU jobs, CUDA builds, paid APIs, Claude or SGLang were used.
Paper/PPT/figure/data outputs in the repository are unchanged. Build validation
used a separate copy, not the introduction writer's working tree.

## What is actually ready

| Payload | Prepared here | Scope and delivery |
|---|---|---|
| Paper Python | 15 wheels, 45,977,253 bytes (see manifest for authoritative size), hashes and dependency closure | `offline/paper/`, in Git; installed without an index into `.venv-paper` |
| Paper system tools/fonts | 286 Debian packages, exact versions and SHA256 checked against APT's authenticated indexes | large local tar, **not in Git or uploaded**; Debian 13 amd64 only, target install untested |
| Baseline CPU dependencies | torch 2.8.0+cpu, transformers 4.57.3, triton 3.4.0 and locked transitive wheels | large local tar, **not in Git or uploaded**; not an official CUDA environment |
| Official source | fixed revisions reacquired; ClusterKV/Quest/TileLang recursive gitlinks materialized | ignored `third_party/offline-sources/`; existing vendor code reused; MoBA source and HISA kernel subset in `offline/sources/` |

Authoritative byte sizes and SHA256 of large tars are in `large-artifacts.json`.
Do not transfer `.venv-*`: Python paths, ABI and native dependencies are host-specific.
Do not call this whole project “offline-ready”: the target host OS/Python/ABI has
not been supplied, native baseline dependencies remain missing, and the large
payloads have not reached the remote host.

## Small host probe, paper bootstrap, real entrypoints

Run on the **target host**, return/save the JSON before choosing a bundle:

```bash
python3 tools/offline_environment/paper_env.py probe
```

The checked-in bundle requires CPython 3.12, Linux x86_64 and glibc >=2.28.
It was built on Debian 13 using Python 3.12.14, not proven on other hosts.
Python must already have `venv`/`ensurepip`; it is not included in the wheelhouse.
For ARM, musl, another Python ABI or old glibc, prepare and separately test a new
bundle; do not rename wheels or reuse this manifest.

```bash
python3.12 tools/offline_environment/paper_env.py verify
python3.12 tools/offline_environment/paper_env.py install
PAPER_PYTHON="$PWD/.venv-paper/bin/python" \
  bash agent_doc/advice/dac_manuscript/source/rebuild_current.sh
```

Install refuses existing environments, verifies every wheel and the requirements
hash, uses `--no-index --only-binary=:all: --require-hashes`, runs `pip check`, and
writes a receipt. The manifest trust root is this reviewed Git revision. Embedded
license/notice texts remain inside wheels; `paper-pypi-provenance.json` independently
matches each wheel to the official PyPI release digest.

`rebuild_current.sh` writes manuscript figure/PDF outputs when intentionally run;
for engineering validation, copy `dac_manuscript` outside the repository first.
`build.sh` now defaults to the four existing paper/supplement sources (the old
`evidence_register.tex` default no longer exists), rejects missing sources, and
supports `PAPER_XELATEX`, `PAPER_XETEX`, `PAPER_TEXMF_DIST`, `PAPER_TEXMF_EXTRA`,
`PAPER_FONT_ROOT`. Python plotting accepts the **file** `PAPER_CJK_FONT`.
`PAPER_MPLCONFIGDIR` controls the writable Matplotlib cache. `inkscape`, `pdftoppm`
and fontconfig must be available in PATH; named fonts need fontconfig discovery,
not just `PAPER_CJK_FONT`.

## TeX/font offline route and transport

The Debian bundle covers XeTeX/TeX Live, Inkscape, Poppler, fontconfig,
Noto CJK, Nimbus/URW Base35 and Latin Modern plus the dependency closure resolved
from an **empty dpkg status**, so already-installed packages on the packaging
host are not silently omitted. No package maintainer scripts were executed and
no system package was installed in this task. The successful paper rebuild used
preinstalled system tools, not an installation of this tar.

- `debian13-packages.json`: exact package version, architecture, SHA256, official
  archive path, source package identity, embedded copyright location.
- `download_locked_system.py`: fetch **exact recorded bytes**, no re-resolution
  and no installation. Fails if upstream removes a version; never substitutes it.
- `download_debian13.sh OUTPUT_DIR`: **new resolution** against moving official
  Debian repositories. This is a preparation recipe, not a reproducer of the
  old lock. It downloads only; do not confuse its output with the locked bundle.

Transport the complete tar as one file by an authorized artifact/object store,
GitHub Release or direct file transfer, then verify its SHA256 and every member
against the manifest. No Git LFS configuration or paid storage is assumed.
The current environment's `gh auth status` failed, and no working release-upload
path was established; ordinary Git push works. Thus large tars remain local.
They have not been split or committed as oversized Git blobs.

On a matching Debian 13 amd64 host, the system administrator can use the verified
local `.deb` closure with APT offline after reviewing the target package plan.
Do not blindly install this closure on Ubuntu, a different Debian release or an
unknown production host. Installation can update libc/system libraries; it is
not automated by this bootstrap. The packages retain their Debian copyright
files and source identities; Debian source archives provide corresponding source.
An alternate OS requires its own official package snapshot/TeX Live platform
bundle, checksum and target test. No universal TeX tar is claimed here.

## Baseline status (this run only)

Full fixed SHAs, license hashes, model/stage boundaries, dependency pins and
recursive gitlinks are in `baselines.json`; actual installed packages in
`evidence/baseline-venvs.json`. Historical remote GPU claims were not reused as
validation of this environment.

| Method | Source | Environment / CPU import | Still missing |
|---|---|---|---|
| FFD | existing Apache-2.0 vendor + full pinned source | independent CPU env, official `ffd_core` import passes | native cu128/FlashAttention GQA route, weights, actual kernel/model run |
| ClusterKV | full pinned source and recursive gitlinks | Python 3.10.15 **scaffold only** | pylibraft-cu11 24.10 wheel (NVIDIA index 403), CUDA/build, faithful local adapter, project redistribution license |
| Quest | existing Qwen3 quality port + full upstream/gitlinks | independent CPU env, Qwen3 adapter import passes | official efficient CUDA kernel; native upstream HF4.45.2 differs from this CPU port environment |
| FASA | existing port + full upstream | independent CPU dependencies installed; import fails on official import-time CUDA allocation | CUDA, calibration; native upstream torch2.5.1/HF4.43 differs from CPU port candidate |
| SparQ | existing vendor/core + full upstream | independent CPU env, core import passes | real model/GPU route; current integration tests expose old/new adapter contract mismatch |
| MInference | existing vendor + full upstream | independent CPU deps; import fails at `minference.cuda` | compiled CUDA extension and pattern/real model verification |
| MoBA | full MIT source, also transported archive | independent CPU deps; import fails at `flash_attn` | flash-attn2.6.3 and continued-training/model conditions |
| HiP | fixed source, FSL-1.1-MIT | separate **scaffold only**, import lacks torch | method-specific CUDA12 stack; no verified current HF Qwen3-8B route; not newly redistributed |
| HISA | fixed author entry + TileLang full gitlinks, MIT kernel subset transported | separate **scaffold only**, source import lacks built TileLang libraries | compiled runtime; DSA/Sparse-MLA interface not current Qwen3/DeepSeek-V2-Lite; SGLang route excluded |

All real-checkpoint executions and GPU-performance cells are **not run**.
The CPU envs deliberately do not mix the incompatible official native CUDA stacks.
A common CPU dependency wheel cache is shared as immutable input; environments
are separate. CPU3.12 is not a proposed replacement for ClusterKV's recorded3.10.15.
FFD's upstream `requirements.txt` pins NumPy1.26.4 and cu128; this CPU candidate
uses NumPy2.3.5 and CPU torch, and is explicitly not the native reference env.
Its pyproject's `dependencies` table is misplaced under `[project.urls]`; no
blind `pip install .` or third-party build script was executed to paper over it.

The MInference recorded `d542c489…` object is an annotated tag; its actual commit
is `d76b76e89cb59817c89e1777c4c51b1c7f233335`. Both are retained. ClusterKV's Python
RAFT24.10 and C++ RAFT24.02 gitlink remain distinct. No sklearn replacement.

```bash
# Safe default prints identity/licensing only; explicit fetch never installs.
python tools/offline_environment/fetch_baselines.py
python tools/offline_environment/fetch_baselines.py --fetch ClusterKV Quest --recursive
# Only after receiving/verifying/extracting the complete large CPU tar:
python3.12 tools/offline_environment/bootstrap_baseline_cpu.py \
  --name ffd --bundle /path/to/baseline-cpu
```

## Validation and limits

- Offline wheel install + `pip check` + synthetic PDF/PPTX roundtrip passed.
- Four real documents rebuilt in an isolated copy: 6/6/8/8 pages; alternate
  TeX/font root lookup also compiled. This is build verification, not scientific
  acceptance or visual review. No rebuilt deliverables were committed.
- 8 offline-safety tests pass normally and under `-O`; tests declare their target
  platform and reject tampering, traversal, musl, incompatible ABI and overwrite.
- Existing `tests/test_baseline_integration.py`: **12 pass / 3 SparQ failures**,
  both normal and `-O`. Unchanged main source/test expectations differ in
  `enable_sparq` return value and `_baseline_stats`; retained logs, no algorithm fix.
- Independent read-only review checked wheel hashes, pip check, shell syntax,
  and isolation/platform guards. It did not certify target-host installation.
- `governance.json` preserves guide/task input hashes without copying credentials
  or rewriting human guides. Remote highest-priority mapped files were unavailable;
  current explicit environment-only authorization governs this bounded work.
