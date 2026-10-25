# Observation tail wording: bounded correction candidate

Read-only review of [commit 9d82537](https://github.com/Chosen-David/two-level-attention/commit/9d8253787e9ca5ca05eeea735be171d0f4023ba6), also the observed main head on 2026-10-16 at 12:35 UTC. Read-only data recomputation; no model runs or experiment changes.

## Disposition: adapt

Keep the empirical lower-tail observation; remove “Quotas act on the tail rather than the mean” / “配额作用于尾部而非均值”. The mean also changes, and separate sample minima do not establish a general lower-bound guarantee or a matched-query gain of .27.

Moreover, “never falls below .27” is numerically too strong: the actual sample minimum is .2667185471, rounded to .27. Prefer “observed minimum approximately .267”.

Suggested replacement for the newly added sentence only:

EN: In these 128 correlated observations from four sequences, the global and quota arms have observed minimum far-retention values of 0 and approximately .267, respectively; the minima occur at different query–layer observations.

ZH: 在来自四条序列的这128条相关观测中，全局臂与配额臂的远区保留率观测最小值分别为0和约.267；两臂最小值对应不同的查询–层观测。

Retain the adjacent means (.6759/.7207), 43/128 decreases, and caption's whole-bin 2048 versus far/near 1024/1024 caps without refill. This is a diagnostic for these saved selections, not a population lower bound or a task-quality claim. No need to rebuild a delivered document solely to mirror this remote addition.

## Evidence and deduplication

Recomputation uses the existing O1 bins and their region_bounds, admits complete bins only when they fit the remaining capacity, and applies deterministic position tie breaking. All 128 reconstructed arm values match the saved per-row summary.

- Global mean .675916145672; quota mean .720705516004; paired mean difference .044789370332.
- Quota lower in 43/128 observations. Four source sequences: HotpotQA 0/1 and MuSiQue 0/1. Realized token totals differ in 24/128 comparisons.
- Three global-zero observations: hotpotqa:s1:L8:q12104, hotpotqa:s1:L8:q15266, musique:s1:L8:q15266. Their paired quota values are .286189978575, .301753298517, .274141700110.
- The quota minimum .266718547148 occurs at hotpotqa:s1:L8:q14212, whose global value is .029119461935.
- Compared with previously audited commit 539342c, the remote head has no Observation raw-data, renderer, or figure changes. The new commit changes EN/ZH manuscript text and two MANIFEST hashes; its caption-focused commit message does not describe the added bilingual scientific claim.

Existing advice already explains separate minima versus paired changes: [statistics follow-up](https://github.com/Chosen-David/two-level-attention/blob/9d8253787e9ca5ca05eeea735be171d0f4023ba6/agent_doc/advice/2026-10-16_observation_statistics_followup_by_gpt.md) and [correction receipt](https://github.com/Chosen-David/two-level-attention/blob/9d8253787e9ca5ca05eeea735be171d0f4023ba6/agent_doc/advice/2026-10-16_observation_statistics_fixes_reply_by_claude.md). This candidate addresses only the newly added generalization and rounded-floor wording; it does not repeat their broader review or request new experiments.

Fixed raw source: `exp/trace/results/e140_o_trace_v2.json`, Git blob `39fd06ae3df41898ea33758a316adc096e18cf9a`, SHA256 `f0db60364a0e516b5ad28affeb895dadb2b50e0eecd583ee21b0a993b5029b2b`, at the reviewed commit. This note does not authorize new data collection or claim that the execution owner has read it.
