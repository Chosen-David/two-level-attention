# Quest Qwen3 quality adapter

Source: mit-han-lab/Quest commit
`01c1623bf9395009520874e989e29f683203b357`; original license is included.
This adapter imports `quest_attention_official.py`, which already contains
Qwen3 Q/K normalization and scale changes. Despite its historical filename,
it is not byte-for-byte upstream, nor a clean-room implementation.

Native Qwen3 prefill and the first two layers remain dense. Other decode layers
use Quest page selection with full QK + mask; this is not the efficient sparse
KV-read runtime. B1 unpadded dynamic-cache decode requires explicit positions;
nontrivial masks are rejected before selection. Positive token/chunk budgets are
required. CPU tiny-HF full-budget parity is a necessary check, not scientific
quality or GPU performance acceptance. See ../README.md for current scope.
