# Paper runtime and evidence entry

This branch implements CPU-verifiable contracts and opt-in runtime candidates. It
contains no new GPU measurements, speedup result, quality acceptance or host
allocation. `python exp/paper/workflow.py --out /tmp/paper-plan` defaults to an
UNFROZEN plan with no launch commands. Unknown revisions, confirmation documents,
thresholds, repeats and host snapshots remain null until independently recovered.

## Runtime

`TLI_PREFILL_RUNTIME=reference` is the default. It retains main's full-history
score → mask → softmax → nan_to_num → GQA probability mean, shape guards and
K2-derived dense-prefix policy. `TLI_PREFILL_RUNTIME=candidate` explicitly selects
the e138-derived batch selector and its separate fixed/default dense-prefix policy.
It requires B=1, matching dtype/device and valid chunk/group geometry; asynchronous
prefetch is disabled. This route is not a claim of reference equivalence.

The compact Triton traversal is reused from the existing performance branch.
The shared GQA index candidate is opt-in with `TLI_PREFILL_SHARED_GQA_INDICES=1`.
Graph signatures bind configuration, backend and input schema; values are copied
into independently schema-checked static buffers. Graph parity and replay-counter
failures raise exceptions, including under `python -O`. Counters record routing
or launches, not completed GPU execution or verified numerical equivalence.

Near average and far minmax remain unchanged. Shared low-D coordinates precede
near/far separation, L1 selects pages, L2 tokens, final attention uses original D,
and D′ is far-only. Group-end-Q selection with row-causal reads is not prefix
invariant. Support receipts count each actual row's unique union with sink/SWA;
protection may exceed the nominal cap, but additional unprotected excess fails.
They do not silently change a selector's budget policy.

## Three thin layers

- `contract.py`, `selection_contract.py`, `receipt.py`: frozen identities,
  effective per-arm routes, synchronized wall boundaries, lifecycle attribution,
  actual support receipts and paired timing diagnostics. T0 kernel, T1 attention,
  T2 model decode TPOT and T3 request wall are separate. T2 divides the measured
  decode-after-first wall by actual generated tokens minus one. Component medians
  are never summed into wall time. Failed/missing/resumed rows cannot create a
  successful curve. Negative speedups remain visible. Caller-provided identities,
  timestamps and routes are declarations, not independently proven provenance.
- `workflow.py inventory`: hashes the four specifically reviewed historical
  families (dense, FS, SS, SF), without exporting predictions. FF is absent only
  in this ledger. Missing historical identities block scientific reuse; today's
  defaults do not repair old identity. `protocol.unfrozen.json` plus executable
  validators define the public schema. Native per-arm configurations can differ;
  controlled comparisons constrain the declared factor and actual per-unit K.
- `workflow.py observations|quality|timing|architecture`: delegates to existing
  CPU analysis and produces immutable diagnostics. Fig1 maps O1/O3, Fig2 the
  architecture, Fig3 E2 actual costs, Fig4 E1/E3 paired quality. Historical Step
  curves are development-only. O3 optionally exports protected/nonprotected mass
  and O1 distance bins in raw JSONL; the existing O3 figure still plots total
  retained mass. Paired quality adds absolute scores and frozen −delta markers;
  intervals are not automatic noninferiority acceptance.

The timing helper is a collector building block, **not an integrated model run**.
No runnable GPU job is emitted. Before GPU use, a host-specific runner must bind
true request boundaries, all observed per-layer phase routes, completed CUDA work,
per-row support coverage, frozen workload, run-cap/scorer and producer identities.
Graph checks gate only the Graph path. Full-width/candidate tie/padding, Triton
numerics, real memory and performance still need independent GPU validation.

## CPU commands

Install CPU-only torch, numpy, einops, matplotlib, jieba and rouge in an isolated
environment. Run from repository root; choose a fresh output directory each time.

```sh
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -v
python -m unittest discover -s sparse_attn/ops/tests -v
python -m unittest discover -s exp/paper/tests -v
python -m unittest discover -s exp/figures/tests -v
python exp/paper/workflow.py --out /tmp/paper-plan
python exp/paper/workflow.py inventory --out /tmp/paper-inventory
python exp/paper/workflow.py architecture --out /tmp/paper-architecture
python exp/paper/workflow.py observations --protocol PATH --out NEW_DIR --plot
python exp/paper/workflow.py quality --protocol PATH --out NEW_DIR --plot
python exp/paper/workflow.py timing --protocol FROZEN_JSON --rows RAW_JSONL --out NEW_DIR --plot
```

Synthetic test fixtures are not measurements. Plan/inventory/architecture success
means only that the requested metadata/diagram was exported. Timing diagnostics
never claim quality-matched scientific acceptance. Existing output directories
are not overwritten.

## Integrated main update

Main `ceb8fa8cdef24941ca09085be4b19b6f70c06b88` is merged into this branch.
Its gather/Graph/candidate-topk/E140 chain is retained in `eager_prefill.py` with
our incremental guards/counters and explicit runtime routing. The original
full-width reference is retained in `reference_prefill.py`; `candidate_prefill.py`
is only a compatibility import, not a second implementation. Other incoming
main files are preserved. Shared-GQA and compact traversal remain opt-in.

E140 tie-probe v2's reported ALL PASS excludes the all-zero boundary. That
boundary had mismatches in every reported case; this branch does not claim
full-domain candidate/full-width equivalence. No existing historical GPU result
is reclassified as validation of this branch.

The requested guide sentence is provided separately in `guide_append.txt`;
`guide_append.json` gives byte-preserving manual instructions. GUIDE is unchanged.

## Observation and collection repair (2026-10-15 v2)

Main `9449c8e9d827f4c079f9ecff5ce2cb8243273d58` is integrated. Direct Graph-cache
callers now also qualify their keys by input schema; failed replays evict only that
qualified entry. The stricter backend/geometry/stride checks are preserved.

See `agent_doc/guide/paper_figs/observation/E140_VALIDITY.md` for the immutable
historical evidence and unexecuted minimal recollection command. The corrected
collector defaults to plan-only. It uses actual normalized projections and actual
RoPE inputs, right-side near, exact SWA, partial pages and explicit metric names;
no historical result was relabeled as corrected data.

Frozen timing protocols now require `support_units[sample_id][arm][unit]`, each
containing query_position/length/sink/swa. Actual receipt units and geometry must
match exactly. This prevents partial declared coverage from silently passing; the
completeness of the frozen list itself still needs independent host validation.
`collect_timing_call` measures the real supplied callback between synchronization
boundaries, freezes runtime counters before outside-wall observation, and validates
actual routes, support and lifecycle. It rejects T0 (host wall is not device kernel
time), counter resets/disappearance, fallback and inconsistent receipts. It is a
host adapter building block, not a completed model-level benchmark or GPU runner.
