# TL-VALID-LENGTH-20261023: reject malformed fixed-capacity lengths

## Scope and contract

Baseline: `fea02f0a9a49dad40f8c59494269832f50632b90`. The relevant runtime files are unchanged from `a7bcdddf727aa99196d925b9f106d785ce9273d4`.

`Indexer.prepare_mask` accepts `None` or a non-boolean host integer in `[1, k.shape[1]]`. Integral scalar types are normalized to Python int. Reject invalid metadata before this method slices K or calls `prepare_index`; retain the existing `None`, legacy indexer signature, tail clearing, ranking, budgets and protection behavior.

This is not a transaction guard for the whole decode caller: cache updates and caller `indexer.clear()` can occur earlier. The direct Graph mask path does not call this entry. No Graph, model quality, latency or GPU improvement is claimed.

## Reproduction and adjacent test repairs

The original runtime accepts oversized lengths and bool/scalar-tensor metadata or fails after entering the indexer on negative/empty lengths. The new real-TLI test fails on the original `base.py` (four failing subcases and three errors) and checks rejection before the preparation call, preservation of prior incremental state, exact valid-prefix/tail masks, partial pages, noncontiguous storage, incremental growth, `None`, and legacy-signature compatibility.

Existing adjacent tests also failed identically with original and candidate base implementations. The test fixes preserve their original negative cases and numerical oracles:

- `test_tli_async_candidate_gate.py`: `_gather_fine_cand` now returns four items (including L1 selection) and forwards `sc_mean`/`sc_avg_mean`; the probe accepts those keywords and still verifies async/unsupported routes never select current candidates.
- `test_tli_runtime_shape_contract.py`: bind the non-Graph harness dependency explicitly; dispatch probes return the actual dictionary contract and accept `valid_length`. Add explicit-length propagation checks for both full and incremental routes.
- `test_eager_decode_input_contract.py`: the public B>1 dispatcher rejects mismatched batch geometry through the batch validator, while private B=1 consumers retain their single-query rejection. Keep every malformed input and pre-allocation check. Add heterogeneous B=2 CPU Q/K/V and mask comparison against per-request outputs with zero tolerance.
- Three CPU prefill fixtures isolate only `torch.cuda.is_current_stream_capturing`, which is unavailable in CPU-only Torch; real tensor indexing, quantization, attention and original numerical assertions still execute. The patch is scoped and cleaned up.
- `test_qwen_static_mask.py`: bind the actual `batch_decode_ctx` imported by the production forward function into its AST namespace. Retain actual HF cache/SDPA and all original static-mask cases.

These are test-harness maintenance changes, not production route or threshold changes. Early failed attempts (including a missing test import) were retained in the recovery evidence.

## CPU validation commands

Use a CPU environment with Torch, NumPy, einops and Transformers. Validation environment here: Python 3.12, Torch 2.14.1+cpu, NumPy 2.4.4, einops 0.8.2, Transformers 4.56.2. This differs from the repository's declared Python 3.11/Torch 2.8 stack.

```sh
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -p test_indexer_valid_length.py
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -O -m unittest discover -s tests -p test_indexer_valid_length.py
PYTHONPATH=. python -m unittest discover -s sparse_attn/ops/tests -p 'test_*.py'
PYTHONPATH=. python -O -m unittest discover -s sparse_attn/ops/tests -p 'test_*.py'
```

The additional host regression set consists of these modules (run via unittest with `PYTHONPATH=.:tests`, in normal and optimized Python):
`test_batch_decode_context`, `test_e139_graph_contract`, `test_explicit_near_page_budget`, `test_indexer_valid_length`, `test_qwen_static_mask`, `test_tli_async_candidate_gate`, `test_tli_debug_contract`, `test_tli_enum_contract`, `test_tli_runtime_shape_contract`.
This selected set is not the full repository suite. No model weights were downloaded.

Independent review additionally compared 64 real CPU configurations: representation, quantization, scorer, GQA grouping and partial lengths; exact valid masks and false padding matched physical truncation. Those completed comparisons were reused rather than rerun after test-only edits.

## GPU gate applicability and remaining work

`E127_REPORT.md` lines 53–59 describes optimization acceptance (including CUDA E127 gates); `TL-E155D-TG-20261021.md` lines 47 and 80–81 require Graph-vs-eager and large-batch-vs-single-request checks before changing T3/T4/T6 consumers. Neither states that every host-only input guard draft PR requires GPU execution. The current change does not authorize or validate those consumer switches.

The following existing CUDA gates remain unexecuted and relevant before GPU deployment; list only, do not infer results:

```sh
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
```

The batch suite explicitly checks fixed-width prefixes, tail clearing and incremental equivalence, and requires CUDA. No thresholds were relaxed and no tests were skipped to claim a green GPU gate.

## Progress

CPU final-run results and independent recheck are recorded in the PR. Publication is a draft branch only; main merging and GPU acceptance are separate.

### CPU result (2026-10-23)

With the repaired test harnesses held fixed, the original runtime fails only the guard regression (four failing subcases and three errors); all other selected host cases and 94 ops tests pass. Candidate: 48 selected host methods plus 94 ops methods pass in both normal and `-O` modes (142 per mode). Independent reviewer separately reran 33 affected methods and 12 Qwen static-mask methods in both modes, and checked the final baseline/candidate logs. No full-repository or CUDA pass is asserted.
