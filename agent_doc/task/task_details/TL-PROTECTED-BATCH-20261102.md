# Mixed-length protected selection

Base: d4d35be8b1807a79f41b08979cffba172c0a55b9. Status: host verified; native CPU/CUDA pending.

`compute_selected_ids_by_layer` uses `_protected_selection_batch` for an all-both-skip layer. The new standalone layer-batched selector also calls it for protected IDs. Its branch is chosen by maximum batch length. With sink4/window128 and lengths[1024,100], the short row previously emitted sink[0:4] then [-28:68], rather than [0:100]. This loses recent tokens and introduces negative/duplicate IDs inside the reported valid count. Same-length long rows and all-short batches conceal it.

Use the already computed row-specific `swa_count = min(length,sink+window)-min(length,sink)` as the tail extent. Then `length-swa_count` starts at sink for overlapping short rows and length-window for long rows. No new kernel, allocation, sync or numerical tolerance; output shape/count and invalid suffix contract unchanged. Reuses the vectorized Torch implementation rather than adding a custom kernel. No measured speed/VRAM benefit claimed.

600 source-bound host cases execute the actual function against an independent set-union oracle: sink0/1/4/16/64, window0/1/7/32/128, lengths around sink and combined limits, long/short order, high request IDs, invalid/removed slots, and pending lengths differing from committed length. Baseline fails; candidate normal/-O passes. Two native tests skipped here because Torch is absent. Native CPU runs the same matrix; CUDA adds the matrix and fixed-input capture/replay. This does not certify dynamic GraphRunner lifecycle, whole-model generation, other selector bugs, or native GPU numerical/performance behavior.

Run from repository root on a configured machine:

```sh
python tests/test_protected_batch_lengths.py
python -O tests/test_protected_batch_lengths.py
compute-sanitizer --tool memcheck python tests/test_protected_batch_lengths.py
```

Ensure native tests are not skipped on the GPU host. Baseline pairing: export the base version of `sparse_attn/ops/paged_region_kernels_nfa.py` to a temporary file and set `NFA_PROTECTED_SOURCE` to it. Do not expect the baseline to pass mixed-length cases. Follow with full both-skip eager/Graph generation using unequal lengths, EOS compaction, replacement/recapture and repeated page reuse after independent request-row/residency fixes.

Overlap: PR10 has another cache ABI. PR26 repairs request-to-page mapping, PR27 residency, PR28 D2H ordering; none changes this helper's logical protection arithmetic. No unmerged dependency for this isolated repair; full-model validation still depends on resolving those separate issues. No guide/manuscript edits or merge.
