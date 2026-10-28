# TL-UNIFIED-FAR-SKIP-20261026

User requested one latest PR integrating existing work for the remote team to
merge. Do not merge main on their behalf.

Integrates main 4846b04, PR #7 fc242e4 and PR #9 7e10baf. Preserves main's
unrelated work, request-local hybrid cache lifecycle and optional GPU batch
fusion. Fixes lazy-workspace integration, duplicate configuration/position
arguments and superseded RULER routing. Existing kernel bodies are unchanged.

CPU related suites: 323 passed, 251 skipped, 395 subtests passed. Triton
interpreter: 99 passed, 38 skipped. Logs, failures, environment and hashes:
`agent_doc/results/unified-far-skip-20261026/`. Current route matrix and remote
acceptance: `docs/unified_far_skip.md`.

GPU capture/replay, performance, allocator memory and real-model quality remain
unverified. Hybrid index/attention is still per-row, not a fused ragged-batch
kernel. Leave this task pending remote acceptance; merging only this integration
PR is sufficient to bring in #7/#9's recorded heads.
