# Quest vendor — unmodified official source + thin adapter

Source: mit-han-lab/Quest commit
`01c1623bf9395009520874e989e29f683203b357`; original license is included.

`upstream/` is a read-only snapshot of the official repository (source, kernels,
evaluation and scripts; large binary assets and unpopulated submodules omitted,
see `UPSTREAM.json`). Adapters load the official
`upstream/evaluation/quest_attention.py` via `_upstream_loader.py` and never copy
or edit the selection algorithm.

Native Qwen3 prefill and the first two layers remain dense. Other decode layers
use Quest page selection with full QK + mask; this is not the efficient sparse
KV-read runtime. B1 unpadded dynamic-cache decode requires explicit positions;
nontrivial masks are rejected before selection. Positive token/chunk budgets are
required. CPU tiny-HF full-budget parity is a necessary check, not scientific
quality or GPU performance acceptance. See ../README.md for current scope.
