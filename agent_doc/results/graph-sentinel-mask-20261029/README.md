# Graph sentinel score-mask removal — 2026-10-29

Main a2e7b67b4e5c131cfa169283f7c66d6fb8892b00; baseline PR14 c147f3c2707263135dd16181b85bfc7b29e299db. Only production delta: remove the sfc torch.where/full_like immediately before sf_full scatter in compute_mask_graph. Preserve dtype guards and dead-SWA-fill fix.

## Mechanism and scope
Actual main lines4473–4483: keep_h and cand_h expand KV candidates across GQA heads. safe_idx maps every invalid candidate to S_cap in an allocated [H,S_cap+1] buffer. score_full slices away this extra column before softmax. Thus invalid sfc values, including NaN/Inf, cannot enter retained columns. Valid entries and duplicate valid-index scores are unchanged. Concurrent invalid writes only contend on the discarded column, already present in baseline. No downstream consumer reads the masked sfc; the next iteration creates new scores.

Removes one full_like allocation/fill and one elementwise where/output allocation, each [H,R_cap]. At H32,R8448,BF16 each is 540672 bytes of logical tensor payload; not a measured peak-memory or HBM reduction (allocator/liveness/fusion matter). Full-width scatter, softmax, GQA reduction, and protected-token denominators remain. Applies when compute_mask_graph runs, including graph replay; not generic eager selector, ring-only route or bypass early-return. Mixed-cache row dispatch remains serial on unchanged main. No new GPU kernel required: reuse existing sentinel scatter, lower maintenance cost than another fusion kernel.

PR10/13 have older compact-normalization code and do not cover this latest-main path. Update PR14, do not transplant their old indexers. All28 branch heads and15 PR timelines refreshed; unchanged ancestry/date/diff evidence reused from page-maintenance audit chain. No new GPU experiment or remote branch delta.

## Validation
81 pairs executing actual AST scatter fragments with NumPy facade, normal and -O: H1/4/32, cap1/17/129, width1/8/132, all/none/mixed validity, duplicate indices and invalid NaN/±Inf. Byte-identical retained outputs. Negative real-token sentinel control detects corruption. Candidate and test compile. This is host fragment evidence, NOT Torch/GPU, full selector quality, native racecheck, peak memory or speed evidence.

## Remote acceptance (not executed here)
```sh
git show c147f3c2707263135dd16181b85bfc7b29e299db:sparse_attn/indexer/tli_indexer.py > /tmp/sentinel-baseline.py
python agent_doc/results/graph-sentinel-mask-20261029/probe.py /tmp/sentinel-baseline.py sparse_attn/indexer/tli_indexer.py cuda --bench
bash scripts/test_tli_semantics.sh --cuda
python exp/bench/test_tli_batch_gates.py
```
Native mode checks actual tensor fragment and captured replay (81 pairs), then optional alternating paired fragment timing, H32/R8448/S131072/BF16, six trials per arm, 100 repetitions. Benchmark is NOT end-to-end TPOT. Before adoption also validate full outputs/selection and eager/Graph paired TPOT under B1/2/10/32, ragged lengths, frozen gates/EOS, wrap/recapture; existing dtype regressions must pass. All remain pending.
