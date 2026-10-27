# TL-CAND-LOAD-20261028

## Plan
Preserve main and PR10, predicate invalid candidate loads in the existing shared fine scorer; no change to selection or arithmetic order.

## Progress
48 frozen-kernel interpreter pairs exact;49 CPU tests passed,12 CUDA skipped. GPU gates unavailable. Evidence and remote commands: ../../results/candidate-load-20261028/validation.md. Target GPU correctness/performance remains pending.

Production-tile follow-up: initial 48 pairs used BT64/4 warps, not production D32 BT16/1. Added 216 exact actual-wrapper interpreter pairs and 72 offline SM80/90 compilations; 9/36 generic codegen pairs use more registers, no spills. Added alternating-order eager/Graph CUDA microbenchmark with empty/mixed/full occupancy. See ../../results/candidate-load-20261028/production-validation.md. No additional production change, GPU measurement or acceptance. Whole-inactive-tile branching deferred pending GPU evidence; downstream mask fusion overlaps the unsafe concurrent softmax branch.
