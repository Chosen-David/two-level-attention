# TL-DUO-ASSETS-20261030

User request: place DuoAttention official source, environment and PDF on main.
Delivery directory: `exp/vendor_baselines/duoattention_vendor/`.

Fixed official source `fe93c314ae87306ef6629dc16713250b4718ffe7`, MIT;
62 unchanged source/config/script/pattern files, verified against official Git
blob SHA and SHA-256 inventory. No Qwen3 adapter or benchmark registration.

Partial delivery: PDF v1 downloaded and inspected locally, hash and lawful
local retrieval script retained; redistribution license not established, so
PDF not pushed. Python3.10 unavailable; isolated `.venv` creation script and
upstream dependency inventory supplied, not a tested runtime or complete lock.
See ENVIRONMENT.md for legacy FlashInfer/CUDA ABI and resolution prerequisites.
No GPU/model/paid operation performed; no author guides or existing kernels changed.

Validation: 4 mocked PDF contract tests normal and -O each; bash syntax;
Python AST parsing; 62 official Git blob hashes and inventory verified.
Whole repo regressions, dependency installation, actual Python3.10 venv,
CUDA/model imports/quality/performance not run. Details in VALIDATION.json.

Next: resolve historical dependency wheels and complete a locked, validated
build on an authorized Python3.10/CUDA host; establish paper redistribution
permission before committing PDF. Task remains open, not a runnable baseline.

## Continuation: actual isolated environment and user-supplied PDF

Base main `6a68fbfdde21f81473303ff814f880723a6717de`. Installed managed
CPython3.10.22, Torch2.4.0+cu121 and pinned legacy wheels into delivery-local
`.venv` (not the user remote machine), resolved a SHA256 lock, passed CPU
imports/ABI/pattern check and pip check. Bootstrap does not run GPU/models.
Setuptools75.1.0 fixes observed pkg_resources failure with TP2.0.0.
BlockSparse remains absent: requires CUDA development toolkit/NVCC and CUTLASS;
source historical commit pinned in ENVIRONMENT.md. This does not claim all
forward paths runnable or performance/quality validated. Initial preparation
record above is historical, superseded for Python/dependency status only.

User supplied original DuoAttention.pdf and requested archival. It is arXiv
2410.10819v1 (20 pages), not ICLR camera-ready. Archived unchanged as
`agent_doc/advice/other_paper_pdf/DuoAttention-2410.10819v1.pdf`; SHA matches
PAPER.json. Original paper rights retained; no claim that code MIT covers PDF.
