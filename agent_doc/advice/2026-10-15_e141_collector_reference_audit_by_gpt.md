# E141 collector/reference audit

- Stable ID: **E141-COLLECTOR-REFERENCE-20261015**
- Status: confirmed source-level mismatches and summary contradictions; native-model impact unmeasured.
- Scope: read-only review of main [87029997b9edf63cabd2fe53a05bb10eb9623be4](https://github.com/Chosen-David/two-level-attention/commit/87029997b9edf63cabd2fe53a05bb10eb9623be4), dated 2026-10-15 04:37:23 UTC.
- This advice does not change code/data, stop jobs, accept experiments, or authorize GPU execution.

## Pinned evidence

| Artifact | Git blob |
|---|---|
| [e141_mla_nope_probe.py](https://github.com/Chosen-David/two-level-attention/blob/87029997b9edf63cabd2fe53a05bb10eb9623be4/agent_doc/guide/paper_figs/trace/e141_mla_nope_probe.py) | `2a4114b13546a6bd8b9a0c975a38992e4fb2504b` |
| [e141_mla_nope_trace.json](https://github.com/Chosen-David/two-level-attention/blob/87029997b9edf63cabd2fe53a05bb10eb9623be4/exp/trace/results/e141_mla_nope_trace.json) | `93d0f2345bddc3a496b3fff0166c00344c3e32cf` |
| [e141_mla_nope_trace_verdict.json](https://github.com/Chosen-David/two-level-attention/blob/87029997b9edf63cabd2fe53a05bb10eb9623be4/exp/trace/results/e141_mla_nope_trace_verdict.json) | `3315f4992ae18954a7884e14223dfed25fcb46ae` |

Official comparison: [DeepSeek-V2-Lite-Chat modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/modeling_deepseek.py), whose file page identifies revision [85864749cd611b4353ce1decdb286193298f64c7](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/commit/85864749cd611b4353ce1decdb286193298f64c7). The run's installed implementation/checkpoint is not identified by the JSON; public-source comparison does not certify those local bytes.

## Two reference-construction faults

1. **Missing attention input normalization.** `get_mla_qk` sends `out.hidden_states[L]` directly into q_proj and kv_a_proj_with_mqa. Official forward stores hidden states before the decoder layer, then applies input_layernorm before self-attention. The probe therefore does not reconstruct that forward's projection inputs. Its kv_a_layernorm is present; that later latent normalization does not replace input_layernorm.
2. **Missing DeepSeek rotary coordinate rearrangement.** The probe uses official cos/sin but directly rotates its q/k coordinates. Official apply_rotary_pos_emb first reshapes the last axis into pairs, transposes the pair axes, and flattens both Q and K before rotating. Matching the cosine/sine cache alone does not establish matching rotary Q/K.

Source locations: official model lines 1394–1408 (stored layer inputs), 1146–1157 (input normalization), 753–791 (attention projections), 322–331 (rotary rearrangement). Verify these against the actual loaded implementation before repair. No measured error magnitude or cause of native saturation is asserted here.

## Selection and support limits

- `head_mean_probs` evaluates mean-of-per-head-softmax probabilities; `arm_score` ranks summed head logits. Sum versus mean logits is harmless for ranking, but neither is generally equivalent to mean probabilities. **full192 is a full-dimensional proxy selector, not the mass oracle.** P-topK is the oracle for the stated aggregate mass objective.
- Retained mass is correctly divided by reference **mid TopK mass**. This is not a full-attention-denominator bug. It is a conditional mid metric, not output error or downstream quality.
- Near takes the oldest quarter of mid, reversing its temporal interpretation; SWA includes 129 positions under the current indexing; final partial pages are excluded from candidate selection while remaining eligible for the reference.
- Prefix slicing is causal for this unpadded, no-cache, zero-origin run. However, variance-selected coordinates use the complete sequence, including later keys, and must be labeled offline rather than causal runtime adaptation.
- Early query budgets can saturate the small mid pool. Report such cases separately from genuine top-768 sparsification.
- The tested NoPE128 coordinates are expanded per-head keys, not the shared latent512 coordinates. Compression in one basis is not compression in the other.

## Summary contradictions and missing raw evidence

JSON contains 4 source samples × layers {4,8,12,16,20} = 20 aggregate cells, **no L24**. Of 320 planned query-layer observations, 157 survive; L20 counts are 7/7/7/8 of 16. L12/L16 are filtered entirely.

The filter is `ref_mass < 1e-9`, not exact zero. Filtered masses/query positions are discarded. Thus neither exact-zero mid mass, FP32 underflow, nor safe zero-cost layer skipping is demonstrated. Means are conditional on surviving queries.

Two direct counterexamples to the verdict's “all rope/full differences ≤.02”:
- hotpotqa sample1 L4: full .9209121019, rope .8932476416; delta −.0276644602.
- musique sample0 L4: full .9298115373, rope .8967376798; delta −.0330738574.

“NoPE compression within ±.02 / no variance-arm separation” also fails in the stored summary:
- musique sample1 L20: NoPE128 .0092410820, NoPE64r .1456912629, NoPE64hi .1805981164.
- musique sample0 L20: NoPE128 .1588331169, NoPE64lo .2996111272.

There are no saved per-query Q/K/logits, selection IDs, input-token hashes, checkpoint/tokenizer/data revisions, or producer/dirty-state receipts. Queries/layers are repeated measurements of four source sequences, not independent samples. Actual independent-document identities still require recovery.

## Interpretation and minimal next steps

**The full-NoPE hypothesis is not rejected by this evidence.** Neither is RoPE-only equivalence established. Preserve the original results as historical proxy diagnostics; withdraw or qualify the above strong verdicts without rewriting original measurements. A 192→64 dimensional ratio is not measured 3× speedup; shared keys alone do not prove an extra factor-of-16 cost advantage. No final selected-token MLA output or downstream task comparison was performed.

1. Check the current authorized host, active jobs, and existing raw archives first. Recover exact loaded source/config/checkpoint/tokenizer/data revisions and original input/template/token identities; do not infer that missing repository data means no host data exists.
2. Before any bounded recollection, validate a small actual-forward fixture: normalized attention inputs, native Q/K and rotary layout, logits/probabilities, dtype and scale. Keep FP32 analysis distinct from native BF16 execution. Repair only the collector/reference and support bookkeeping needed for this gate.
3. Reuse matching raw evidence if available. Otherwise obtain the applicable execution authorization before a bounded recollection; this advice adds no experiment matrix or GPU permission. Preserve causal query positions, skipped-query values/reasons, finite checks, exact selected/reference IDs, subspace coordinates/seeds, and file/code/protocol hashes.
4. Recompute summaries with support/budget saturation and skip counts visible, and document-level independence explicit. Actual output/quality and cost claims require their own matched evidence; do not promote collector tests to scientific acceptance.

Related context only: [E140 historical validity and recovery limits](https://github.com/Chosen-David/two-level-attention/blob/a38620cae34c4ca7ac58c85f27ce9764fbe3b01e/agent_doc/guide/paper_figs/observation/E140_VALIDITY.md) and [E140 quota/mass-oracle audit](https://github.com/Chosen-David/two-level-attention/blob/f0b0308208c86a6532468dc2c297f048826b85e1/agent_doc/advice/2026-10-15_e140_quota_and_mass_oracle_audit_by_gpt.md). Those reports do not certify E141 or authorize its execution.
