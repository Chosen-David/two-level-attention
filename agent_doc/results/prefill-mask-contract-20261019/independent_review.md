# Independent prefill mask contract review

Reviewer: independent context `twolevel_prefill_mask_review36`.
Baseline: `3b0cfea1dac5598b44bfa28818b020c5a7e61227`.
Scope: read-only production inspection; no GPU, model, performance, remote or publishing actions.

## Plan review — approved for implementation

The frozen plan is a narrowly scoped fail-closed correction. Both `reference_prefill.sparse_prefill_attn` and candidate `_prepare_sel` currently discard returned block size and slice `mask[0, 0]` without validating the selector protocol. Checking metadata before this slice is the right consumption boundary. Do not repeat/expand block selections: that would introduce a separate prefill algorithm and protection semantics outside this task.

Required contract: a strided bool Tensor on the Q/K/V device, exactly rank 4 and batch/query dimensions `[1, 1]`, selection heads equal to Hkv or H, token extent at least the current group prefix; block size must be `numbers.Integral`, not bool, and equal to 1. The helper must return the original storage slice, not copy or normalize strides. In particular scalar Tensor block sizes must be rejected without conversion or device synchronization. Padding may contain arbitrary true bits beyond the prefix because it is sliced away. Non-contiguous and expanded/zero-stride masks are legal read-only input; contiguity must not be required.

### Reachability and preserved meaning

- `qwen3_attn_patch.py` dispatches through this prefill entry when `TLI_SPARSE_PREFILL=1`, B=1 and attention mask is absent. There is no indexer-class restriction on that branch, so incompatible Quest/block-level selector output is reachable, not merely a fabricated unit-test type. The change makes unsupported input explicit; it does **not** add Quest prefill support.
- `Indexer.prepare_mask` returns `compute_mask` output and `get_block_size()`. `TLIIndexer.get_block_size()` returns literal 1; TLI's page size is independent from its returned token granularity. The old `ProtectionIndexer` fixture returns token-wide masks but reports page size: correcting only that return declaration to 1 is justified, while retaining its page configuration and numeric assertions.
- Hkv selections share support within each GQA group; H selections preserve distinct query-head support. Prefix extent must use each actual group end, including an unaligned partial final group, not total prompt length or the outer chunk end.
- Candidate regular groups, sequential batch fallback, and unaligned batch tails all reach `_prepare_sel`. Internal successful `batched_group_sels` outputs are a distinct trusted path, explicitly unchanged by this task.
- Empty/all-dense inputs do not consume a selector and need not reject nonexistent outputs. The helper is after selector execution: it does not promise rollback of indexer caches/metrics. Existing quota `finally` cleanup must execute even when the returned mask is invalid.

### Independent baseline experiment

Bounded CPU command (`timeout 120s`, torch threads=1) used Q=K=0, V[t]=t, L=8, H=2, Hkv=1, chunk=4, dense_below=4, sink=SWA=0. Each malformed output sets only its first slot true.

| Malformation | Reference baseline | Candidate baseline |
|---|---|---|
| `[1,1,1,1]`, block=1 | Accepts; last output=3.5 (silent all-prefix broadcast) | Late RuntimeError from in-place broadcast |
| `[2,1,1,8]`, block=1 | Accepts, ignores batch 1 | Accepts, ignores batch 1 |
| `[1,2,1,8]`, block=1 | Accepts, ignores query 1 | Accepts, ignores query 1 |
| `[1,1,1,8]`, block=64 | Accepts, ignores granularity | Accepts, ignores granularity |

Outcome: real silent defects reproduced on baseline; candidate late failure for width=1 is not a valid fail-closed metadata contract. Plan approved with no blocking design issue. Result review remains pending implementation and test evidence.

## Result review — approved within the frozen scope

Inspected the completed `prefill_contract.token_selection` and both consumption-point diffs. The implementation matches the planned contract, uses explicit `ValueError` rather than assertions, and only accesses metadata before returning `mask[0, 0, :, :length]`. There are no reductions, scalar Tensor conversions, copies, new kernels, automatic block expansion, or unrelated production edits. Existing selector `finally` cleanup precedes helper validation. The fixture change is limited to declaring its already-token-level mask granularity; numerical protection assertions are unchanged.

Independent CPU verification performed after implementation, each command bounded by 120 seconds:

- **15 helper checks passed**: nonzero storage offset with token stride 2, zero head stride, zero token stride, Hkv and H masks; returned data pointer, storage offset and strides match the original sliced storage. `numpy.int64(1)` is accepted. Bool, zero/negative/block granularity, float 1.0, CPU scalar Tensor, meta-device scalar Tensor and numpy bool granularity are rejected.
- **54 independent numerical cases passed**: random nonzero Q/K logits and V, L in {5,7,9}, H=4/Hkv=2, shared and per-query-head masks, strided/expanded-head/expanded-token masks, both runtimes and candidate group=2. The independent oracle explicitly enumerates each absolute-position causal support, chooses the correct group-end selection, applies per-row/head diagonal fallback, and computes attention directly. Comparison tolerance: rtol=2e-5, atol=2e-6. Selector budgets cover their possible selected counts.
- **2 candidate routing checks passed**: forced batched-selector `None` fallback and successful aligned groups with an invalid unaligned last-group selector both raise the new contract error.
- Independently reran `python -O -m unittest sparse_attn.ops.tests.test_prefill_mask_contract -v`: all 3 methods passed, including invalid metadata, silent singleton regression, and valid padding/stride/heads/short inputs. Also inspected the main agent's normal-mode passing `after.log`.

Diagnostic caveat: an initial random-selector experiment configured K2=2 but allowed arbitrarily many selected tokens; it exceeded the candidate's existing `w_sel_cap` assumption and failed the independent numerical oracle. This capacity assumption and the corresponding code are unchanged by this patch. The experiment was corrected to configure a covering selector budget. This review does not claim the metadata helper validates selected-token counts or makes arbitrary budget-inconsistent selectors safe.

Compatibility is intentionally narrowed for previously ignored malformed metadata, scalar-Tensor/float granularity, and all block_size!=1 selectors. Legitimate token masks retain padding, stride, GQA, per-head, empty/all-dense, chunk/group-end and oracle-cleanup behavior. No blocking finding remains for this change. Internal successful batched selector output validation and candidate capacity policy are outside this patch. Verification environment: torch 2.8.0+cpu, CUDA unavailable; no GPU, real-model, fused-kernel, performance, or serving acceptance is inferred from these results.

## Reproducible evidence and actual-input fingerprint

At the main agent's request, the already executed helper/oracle/routing checks were organized into `verify.py` in this directory and **actually rerun** successfully. This is not a reconstruction of baseline evidence. The script keeps the successful oracle's torch seed 107 and explicit covering budget. It adds an ordered aggregate SHA256 of actual input Tensor events: Q/K/V, selector-call inputs, masks consumed by the helper, scalar Tensor granularity arguments, and aligned mocked batch masks. Each event encodes shape/dtype/device/stride/storage offset and logical C-order bytes with length framing; the meta scalar is explicitly encoded as unmaterialized metadata, since it has no physical value payload. The hashing instrumentation is CPU test-only and is not part of production or its no-synchronization claim.

Executed from repository root:

```sh
timeout 120s env PYTHONPATH=. python agent_doc/results/prefill-mask-contract-20261019/verify.py
```

Raw short output (exit status 0):

```text
helper_metadata_storage_checks 15
independent_random_logit_oracle_cases 54
batch_fallback_unaligned_tail_rejections 2
input_tensor_events 614
input_sha256 fb7440b0d854e1716e80e50693a20d2bd60b918ae778ea9a37ae950400f98565
torch 2.8.0+cpu numpy 2.3.5 cuda_available False
PASS
```

The fingerprint is an ordered input-event aggregate, not an output/model/GPU fingerprint. Earlier normal/optimized unittest evidence remains separate; this rerun did not rerun the original unfixed baseline.
