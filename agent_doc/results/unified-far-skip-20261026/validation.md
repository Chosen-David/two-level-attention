# Unified far-skip validation — 2026-10-26

Inputs: main `4846b04a1b822be2e19bfb57e9f876158c17291b`,
PR #7 `fc242e45070b71afe56c4d9521ef978f28020bfb`,
PR #9 `7e10bafd2eb5a4b91b7a5e10536084161e025158`.
The new integration commit records all three parents; main is not merged by
the author. `docs/unified_far_skip.md` is the current support/acceptance matrix.

## Executed results

- CPU related suites: **323 passed, 251 skipped, 395 subtests passed**, 98.93 s.
  This is the downloaded related `tests/` subset listed in `manifest.json`,
  not every test in the repository. See `cpu.txt`.
- Triton interpreter: **99 passed, 38 skipped**, 67.36 s. Skips include BF16
  interpreter limitations and real-CUDA-only cases. The 144 warnings are the
  interpreter's NumPy scalar-conversion deprecation. See `interpreter.txt`.
- Python syntax compilation and whitespace checks of integrated changed source
  passed. GPU kernel bodies are unchanged from PR #7; historical offline build
  receipts are retained, not represented as new CUDA executions.

CPU environment: Python 3.12.14, Torch 2.8.0+cpu, Transformers 4.56.2,
pytest 9.1.1, NumPy 2.5.4, Triton 3.4.0 installed. Project Python 3.11/CUDA
acceptance remains outstanding. Source hashes and exact suite paths are in
`manifest.json`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/tli-venv/bin/python -m pytest -q tests --tb=short

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/tli-numpy-2.2.6:/workspace/scratch/7d83fead945d/triton-runtime \
TRITON_INTERPRET=1 /tmp/tli-venv/bin/python -m pytest -q \
  tests/test_tli_gpu_pipeline.py tests/test_tli_batch_fusion_kernels.py --tb=short
```

The `/tmp` paths describe the actual validation environment, not repository
dependencies or deployment paths. A temporary NumPy 2.2.6 installation was
used only for the interpreter; project requirements and kernel code were not
changed to accommodate this tool limitation.

## Failures found and resolved

Naive combination repeated `cache_position` in GraphRunner.forward, causing a
syntax error. PR #9's hybrid reader also assumed an eagerly created workspace,
while PR #7 makes it lazy. The shared setup helper now seeds hybrid rows during
construction and keeps B=1 staged eager allocation lazy; a new regression
checks setup idempotence and independent workspace ownership.

Initial combined run after syntax/setup fixes: 35 failed, 285 passed,
251 skipped. It included missing files in the partial checkout, old tests
omitting the required `near_tokens=off`, a test passing no model to the new
model-dtype GraphRunner, obsolete RULER B=1 assertions, and a CPU oracle trying
to launch installed Triton. Missing dependencies were fetched unchanged from
main. Test harnesses now use the actual entry contracts and real Torch CPU
reference operations. Assertions and numerical tolerances were not relaxed.
The complete initial log is `cpu_integration_initial.txt`.

The initial interpreter attempt with NumPy 2.5.4 produced 14 failures because
Triton 3.4 tried `int(array([value]))` in runtime loop bounds. The failure log is
`interpreter_numpy25_failure.txt`. Repeating with NumPy 2.2.6 passed all 99
executable cases; this is an interpreter compatibility result, not evidence
that CUDA execution succeeded.

## Scope

Real random Qwen3 models exercise B=1/2/8/32, ragged lengths, request-local
frozen gates, ring/full layers, repeated wrap, EOS storage freeze, old prefill
storage release, new batches, static refill and output comparisons. RULER
compact entry additionally compares B=1/2/8 generated tokens with an independent
DynamicCache reference. Full-batch pipeline tests cover both skip/non-skip.

No GPU was available. CUDA skips, interpreter runs, CPU storage bounds and
historical compiler receipts do not establish real GPU capture safety, speed,
allocator/peak savings or real-model quality. In particular, no new 13% memory
claim is made. Hybrid index/attention still loops over rows. Remote CUDA,
sanitizer, paired performance and model-quality commands are in the support doc.

Before publishing, main advanced from 2289900 to 4846b04 with only the E175 script register_patch API fix. Its exact upstream blob is preserved. The script was syntax-checked; this independent benchmark-entry fix changes none of the tested attention/cache/kernel modules.
