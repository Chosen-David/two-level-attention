# Paged CUDA benchmark import repair

Base: main 00c2a1d3566746dfbc0b0c7942dd1d18eee3f28a. Both stale imports were already present in 308b38946feae176b654ce80de423e090b0a6548; the subsequent multi-request cache commit changed neither benchmark.

The tree has sparse_attn/cache/paged_region_cache.py exporting PagedRegionLayer and no sparse_attn/ops/paged_region_cache.py. Both CUDA check scripts imported the absent path. Only those two import statements change.

## Executed checks

- check_imports.py against baseline: exit 1, two missing-module failures.
- Against repaired files: exit 0, two module/export checks pass.
- Both scripts compile; normalized ASTs identical except the cache module import.
- No torch, triton, pytest or GPU locally. These are static checks, not CUDA execution or speed measurements.

Run the dependency-free check from the checkout root:

```sh
python agent_doc/results/paged-bench-imports-20261031/check_imports.py
```

Run native checks in the existing GPU environment (retain all current numerical thresholds):

```sh
mkdir -p /tmp/tli-paged-import-check
PYTHONPATH=. .venv/bin/python exp/bench/check_paged_region_kernels.py --out /tmp/tli-paged-import-check/kernels.json
PYTHONPATH=. .venv/bin/python exp/bench/check_paged_region_arithmetic_boundary.py --out /tmp/tli-paged-import-check/arithmetic.json
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_paged_region_cache.py tests/test_paged_region_batch_cache.py tests/test_paged_region_tli.py
```

GPU correctness, selected-ID/model quality and end-to-end performance on the new paged default require separate acceptance. No historical ring/Graph speedup transfers to this migration merely because the scripts import successfully.
