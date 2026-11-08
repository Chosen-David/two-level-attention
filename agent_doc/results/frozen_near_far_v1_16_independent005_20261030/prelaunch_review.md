# V1 16-task prelaunch extension review

Reviewer: /root/verify_pool_initial. Worktree: /home/wangyuanshuo02/two-level-attention-frozen-main.

No blocking defect identified in the four benchmark-file extension as reviewed. This is a source/regression review, not GPU smoke or final result acceptance.

- 16 unique English/code V1 tasks include the original five and multi-hop 2wikimqa. No-chat list correctly includes trec, triviaqa, samsum, lcc and repobench-p; lsht is outside this English/code scope.
- SAMSum EOS plus newline and first generated token suppression match existing pred.py min_length=input_length+1. Independent check compares greedy_token against Transformers MinLengthLogitsProcessor at generated counts0/1/2. Other tasks retain model generation-config EOS behavior; no claim of universal pred.py pipeline equality (one-shot frozen prefill is explicitly different).
- Validator independently reconstructs the v4 no-chat prompt and SAMSum stopping policy, preserves historical v2/v3 reconstruction, rejects early/interior stop tokens and empty/overlength sequences. Empty SAMSum sequence currently raises IndexError before length assertion; it still fails closed.
- AVG16 uses equal weight per task, not per-request weighting; paired bootstrap resamples within each task before averaging. Full export consistently uses AVG16 keys, tables and labels. Original AVG5 compatibility regressions pass.
- Independent synthetic18-request/16-task fixture with unequal task counts confirms macro37.5, paired constant drop20pp, and export16 output with no AVG5 label. Fixture is a regression only, not empirical quality evidence.
- Existing analysis/full-dataset/request-guard suite:29 passed. Independent extension script: PASS. Script retained at /tmp/review_frozen_v1_extension.py; no repo source edited by reviewer.

Full-run acceptance still requires frozen source/plan, model and input identities, all3750 distinct requests/all11250 arms, per-task scores and errors, and independent result validation. New v4 results must not inherit AVG5 v3 data. GPU smoke and deployment queue changes are outside this four-file review.

Reviewed file hashes:

```json
{
  "benchmark/LongBench/prefill_frozen_skip.py": "602d431951f37341f2855c938fb0c182dec423a284e070e53cab1e10e3837e34",
  "benchmark/LongBench/analyze_frozen_skip.py": "ca61ba7181249417ed1b188dfb17c6c9e7c645cfeca88a3d21bfa248cbac1eae",
  "benchmark/LongBench/validate_frozen_skip.py": "b27a5ce14e83629d5f36d10e9dfe160a504e794d6723734066d49c65d0702f14",
  "benchmark/LongBench/export_frozen_skip.py": "5ad5af2879bd1253f9b665c326bcf99447e5640476f436a6afe7628273ea731c"
}
```
