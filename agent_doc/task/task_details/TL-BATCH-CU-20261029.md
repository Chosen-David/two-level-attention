# TL-BATCH-CU-20261029

Implemented: invocation-local read-only [0,T] tensor reused across fallback rows. Main 0fd4051 analyzed; depends on PR10 cf8eb15, latest-main integration remains pending.100 actual wrapper AST pairs normal/-O pass, wrong-bound negative rejected; no native Torch/CUDA. GPU timing, Graph recapture and full integration pending; no speedup/memory claim.

Evidence and reproducible commands: ../../results/batch-cu-reuse-20261029/README.md. All25 branches/13 PRs checked; unchanged branch conclusions reused. No main push or merge.
