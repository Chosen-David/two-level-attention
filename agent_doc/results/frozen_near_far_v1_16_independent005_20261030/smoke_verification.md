# Independent V1 smoke acceptance

Verdict: **usable-with-scope — ready for full V1 launch**. This approves the smoke evidence and launch readiness, not full-run results or accuracy equivalence.

Verifier: `/root/verify_pool_initial`, 2026-10-30; separate from producer `/root`.
Source: `08ce44d6027a235080e7ff4127754af6b1ae91e0`.
Smoke identity: `943ca7ce1dddd338a6e054bd7fdf7b894027ab32f5f6b28d1d3b28ab1e0e051e`.

- All3 planned requests completed: SAMSum152, TREC152, LCC473; each has full/far_0.05/dual_0.05, total9 arms. Worker completion, manifests, mechanical validation and summary share identity.
- Independently checked81 bound source files against frozen checkout and verified checkout HEAD. Three-host preflight records have matching16 dataset plus model/config hashes; recorded shared-lock counter is90. This review inspected receipts, not a second cross-host lock run.
- Reconstructed all three formatted prompts directly without chat templates and matched input-token hashes against local bound datasets/tokenizer. All are declared no-chat tasks.
- Independently rechecked SAMSum EOS+newline/minimum1, all tasks’ max-length and interior-stop rules, token decoding, and scores with pinned difflib backend. No nonfinite scores or malformed outputs found.
- Independently recomputed every smoke layer’s own mean-mass<0.05 Near/Far decision and checked Far remains unchanged when adding Near gating. Both gates off in full; native full-generation parity is recorded for one request.
- Producer mechanical validation also passes3/3 requests,9 arms,324 request-layer checks, identical prefill profiles across arms; analyze/export succeeded with dynamic AVG3 labels for this subset. Prior independent16-task regression separately established AVG16 equal-task weighting and export behavior.

Independent CPU recheck script retained as `independent_smoke_check.py`; executed successfully against completed raw requests. No GPU rerun or production modifications made by this reviewer for this smoke. Source review, independent recomputation and existing native parity provide the scoped reproducibility cross-check; no latency claim.

Limits: one sample per smoke task, only three of16 tasks. Near-only staging activated in zero smoke layers, so this smoke does not newly prove its migration behavior (covered separately by prior cache audit/regressions). No result-quality inference from9 outputs. Full launch remains the frozen16-task/3750-request/11250-arm/128-shard plan; final data must receive a new independent acceptance. Do not merge historical AVG5/v3 records into this v4 run.

Evidence bindings:

```json
{
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/manifest_rank0.json": "e9e6c2b9e2381eff917bedcae55bff814f4afd3558528e50d2963521cabef579",
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/validation.json": "2ac25321d37ac7722ffea87593cb6771cb000e2e2e504a2cf8c9d4bf4db4d4eb",
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/summary.json": "83ab626f5c8d47424c18890ae68aa78370bf623def0a3f9842d8f3f3a94aa2bb",
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/worker_done_rank0.json": "f333be44abc386e50a13bf8a998cec2101cfe39d77fe91682c8a1b98b88c76f7",
  "/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/preflight/receipts.json": "5eeca92b28783ef4b4d65865dc5ee225f81ec52701910f9125bb992805cdbc6a",
  "/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/independent_smoke_check.py": "a39baf246e7f2fc306c921a5d3a6c4568228ce76302ad8d704963aae4c29152a",
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/requests/lcc/0473.json": "d744226afb3bdfec469f2c78bb4672b04c1975cb2536c0da5f245e27d0c90939",
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/requests/samsum/0152.json": "8a01b5badcb482c938fd98e2b3918281bb54903376228b2e7c0d3d4662861c5f",
  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030/requests/trec/0152.json": "02e6d6b06daa170c860e09028fd5342b0ce07bd6641d67beb8fa1bad4f64bbc9"
}
```
