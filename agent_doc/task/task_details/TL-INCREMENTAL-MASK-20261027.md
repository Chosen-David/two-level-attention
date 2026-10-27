# TL-INCREMENTAL-MASK-20261027

Plan: audit main hybrid runner input work outside Graph replay and EOS lifecycle.
Progress: main17e56e8 still rebuilds full causal mask each step. Use validated
append-only recurrence: initialize once, scatter one valid position per row.
Preserve full dense mask shape/address and all gate/cache semantics. 19 CPU tests
passed,12 CUDA cases skipped; GPU timing pending. Update PR10, no main push/merge.
Evidence/remote commands: [validation](../../results/incremental-mask-20261027/validation.md).
