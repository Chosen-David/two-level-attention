# TL-SUBSPACE-PAD-20261027

Main8f213e4 / PR10 parentc96752e; isolated candidate-subspace-pad-20261027.
Trace full preparation -> full-D zero/pad -> subspace gather -> fused statistics.
Adopt: gather selected dimensions before padding, keeping packed boundaries and
MoBA full-D mean. Existing statistic kernel reused; no reduction/kernel change.
Qwen cold/Graph-init scope; MLA eager full rebuild repeats per step. No steady
Qwen Graph speed claim and no KV-residency change.

Done:201 CPU passed,24 CUDA skipped;30 Triton interpreter pairs exact. Actual CPU
padding backing smaller; formula-only128K tail256->64MiB, not GPU peak savings.
Pending: GPU tests/required gates, alternating timing/allocator benchmark,
independent validation and production e2e. Frozen main reference retained.
PR11 and opt branches checked: no overlapping padding fix. Do not duplicate
PR11 fixes; all-PR integration remains separate. Capture geometry duplication
requires capture identity/lifetime handling; direct-source stat needs packed
page mapping. Neither is claimed fixed here.

Evidence: [validation](../../results/subspace-pad-20261027/validation.md).
Publish on existing PR10, leave review/merge to remote team.
