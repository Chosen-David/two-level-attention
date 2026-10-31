# TL-PAGED-BENCH-IMPORTS-20261031

Source: user's recurring main implementation/performance audit. Base main 00c2a1d3566746dfbc0b0c7942dd1d18eee3f28a; migration introduced in 60614e21d50c99982cd7d62a508df0ed6be88390.

Problem: the cache moved from sparse_attn.ops.paged_region_cache to sparse_attn.cache.paged_region_cache. Two retained CUDA checks still import the deleted module, preventing entry into correctness validation even with GPU dependencies installed.

Change: update only the cache import in check_paged_region_kernels.py and check_paged_region_arithmetic_boundary.py. All numerical comparisons, tolerances, cases, and kernel code are unchanged. No dependence on PR10; its current head 2449f56983d7e97c5fe1c4df300c2b50b0c16b81 does not cover this post-migration repair.

Validation: dependency-free source/module export check fails twice on the pinned baseline and passes twice after the repair. AST comparison after normalizing the import target proves the remaining script bodies are identical; both compile. Torch, Triton, pytest and CUDA are absent locally, so native execution and GPU correctness remain pending. This is an engineering import repair, not a scientific verdict or performance improvement. Reproduction and GPU commands: ../../results/paged-bench-imports-20261031/README.md.

Plan: retain upstream multi-request cache work, publish isolated PR, leave merge to remote team.
Progress: two import fixes and local checks complete; GPU checks pending.
