# TL-GQA-DTYPE-20261029

Main a2e7b67 reviewed. Both eager and Graph callers allowed non-BF16 into a kernel with explicit BF16 output rounding. Add dtype guards, preserve existing fallback and latest main semantics.192 actual AST caller cases pass normal/-O; rounding counterexamples and inverted-guard negative provided. Native GPU/Graph/full model acceptance pending; no speed claim.

Evidence: ../../results/graph-mean-dtype-20261029/README.md. Narrow latest-main repair independent of unmerged PR10; preserve when integrating. All25 branch heads and13 PRs checked; no main push/merge.
