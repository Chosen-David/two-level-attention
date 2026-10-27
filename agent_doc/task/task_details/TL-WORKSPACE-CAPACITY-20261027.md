# TL-WORKSPACE-CAPACITY-20261027

Plan: audit main cache/workspace ownership and hidden allocation capacity.
Progress: fixed Graph cat and PR linear gather overallocate via geometric growth.
Use exact capacity for fixed-shape workspaces, keep variable eager growth.64 CPU
regressions pass,9 CUDA skipped. Paired CPU backing storage8MiB->4.515625MiB for
standard linear pair, not GPU allocator/latency. Prior4.52MiB claim corrected.
Update PR10, no main merge/push. GPU acceptance pending.
[Evidence](../../results/workspace-capacity-20261027/validation.md)
