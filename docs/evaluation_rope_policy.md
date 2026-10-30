# Near-Far Attention: corrected Qwen YaRN policy

This is the intended evaluation configuration. It is not a completed benchmark result.
Policy ID: `near-far-rope-32768-v1`. Implementation: `benchmark/rope_policy.py`.

| Model / nominal input tier | Additional YaRN | YaRN original length | Runtime position capacity |
|---|---:|---:|---:|
| Qwen3-8B / Qwen3-32B, 32K | off | not applicable | unchanged snapshot (local snapshots: 40960) |
| Qwen3-8B / Qwen3-32B, 64K | factor 2 | 32768 | 65536 + maximum output tokens |
| Qwen3-8B / Qwen3-32B, 128K | factor 4 | 32768 | 131072 + maximum output tokens |
| GLM-4.7-Flash | no Qwen override | preserve published configuration | preserve snapshot (local snapshot: 202752) |

K means 1024. The model snapshot's `max_position_embeddings=40960` does not define
Qwen's YaRN original length. Explicit factor, original length and runtime capacity
are separate. The extra output reservation does not change the explicit factor.
Both single-request and batch RULER entrypoints reject actual input+output lengths
that exceed the configured capacity. Input truncation is not performed by this policy.

Before weights load, the installed Qwen rotary class is instantiated on CPU and its
`inv_freq`, attention scaling and cosine/sine output are compared with an independent
static-YaRN formula. The loaded model's actual rotary module is checked again.
A framework that overrides the factor fails before generating predictions. Verified
locally: Transformers 4.56.2 and 5.12.1. The latter requires its compatible Hub package.
GLM must use a compatible Transformers build and retains its published configuration.

New receipts separate `native_max_position_embeddings` (snapshot capacity),
`rope_original_max_position_embeddings`, `rope_policy.runtime_max_position_embeddings`
and the actual `runtime_rope_verification` tensors/hashes/scaling/source versions.
The main-table scorer checks the new evidence and fingerprints it. Historical
receipts remain readable but long-context Qwen receipts without corrected runtime
proof are rejected by default; `--rope-policy legacy` explicitly scores their original
setting. This switch does not make old results eligible for the corrected main table.

## Reproducible commands

Run from the fixed source checkout with the project's Torch2.8/Transformers4.56.2
Python. Use new output directories; never overwrite old generation pointers.
GPU0 below is an example; verify it is available first.

```bash
CUDA_VISIBLE_DEVICES=0 /home/wangyuanshuo02/two-level-attention/.venv/bin/python \
  -m benchmark.RULER.pred_ruler \
  --model Qwen3-8B --model_path /home/wangyuanshuo02/Qwen3-8B \
  --task niah_multikey_2 --context_length 65536 --method none \
  --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
  --output-dir /NEW/OUTPUT/DIRECTORY --pred_postfix _rope32768 \
  --t corrected_yarn --max-num 100 --yarn --yarn_factor 2
```

For 128K use `--context_length 131072 --yarn_factor 4`; for 32K omit both YaRN flags.
`pred_ruler_batch` shares the policy and accepts `--batch-size N`; only methods
admitted by that existing entrypoint may use larger batches. Do not change a
baseline's selection semantics to make it fit batching. All Qwen baseline methods
loaded through these entrypoints receive the same verified RoPE configuration.
This does not certify their attention-token budgets or complete Near-Far method
coverage; those are separate blueprint-audit items.

For historical scoring explicitly add `--rope-policy legacy` to existing
`benchmark.RULER.score_ruler_formal` commands. New main scoring defaults to
`--rope-policy main`; missing long-context provenance cannot be repaired by
supplying `--yarn` at scoring time.

## Evidence and reuse boundaries

Canonical evidence: `PROJECT_ROOT/agent_doc/results/evaluation_blueprint_audit_20261030/`.
The independent CPU probe covers Qwen8B/32B and actual GLM snapshot configuration.
The 8B GPU diagnosis covers one MultiKey2 input at each 64K/128K, comparing only
original length under identical runtime capacities, weights and inputs. It stores
first-step logits and 8 greedy tokens, not benchmark accuracy or latency evidence.
A failed initial diagnostic attempt is retained separately and excluded.

- Qwen 64K/128K predictions actually generated with original 40960 require regeneration
  under the target policy, including Full Attention and each compared baseline.
- The completed `frozen_qasper_mk2_64k_20261030` 100-row run explicitly used 40960.
  Its raw predictions remain historical; they are not relabeled 32768.
- Unscaled 32K/LongBench and native GLM do not require regeneration because of this
  YaRN correction. Other input/freeze/budget mismatches may independently require it.
- Archives with only declared factor or missing actual producer configuration remain
  unknown until records are inspected. A summary title or code comment is insufficient.
- Corrected full quality tables, complete-method coverage and 32B GPU quality remain
  unexecuted until separately dispatched and independently validated.
