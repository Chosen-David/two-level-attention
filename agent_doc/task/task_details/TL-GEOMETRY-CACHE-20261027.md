# TL-GEOMETRY-CACHE-20261027

Source: user requests latest-main performance analysis and validated PR updates.
Main8f213e4; PR10 parent19e69bc. Isolated candidate-geometry-cache-20261027 snapshot.

Done: traced hybrid row geometry cache; actual-main B>=10 cache-thrashing probe;
request-owned, batch-sized bounded cache; frozen-old-policy full-model pairs.
25 CPU passed/9 CUDA skipped;840 exact-logit comparisons, B32 two-layer eager
geometry builds64->32. Additional batch-indexer regression recorded in evidence.

Pending: CUDA correctness, paired GPU event/wall/allocator measurements, production
TPOT and quality gates. Capture still duplicates geometry; no Graph speed claim.
No-context bounded fallback and inference-mode bypass remain. More cached live
geometry at large B is an explicit memory/runtime tradeoff, not KV savings.

Evidence: agent_doc/results/geometry-cache-20261027/validation.md.
Submit as PR10 update after latest-ref check; team reviews/merges. Separate
23:30 integration task will reconcile all open PRs against then-current main.
