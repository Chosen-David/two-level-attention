# Eager selector plan admission

Status: pending native GPU / Graph / performance validation.
Base: 40756b6c925ad0caa5741ccbc305299fb8189567.
Scope: user's continuing main correctness/performance audit; only this repository.

## Mechanism and change

Whole-call eager replay only rebinds the fused page-stat/Q producer. With
`l2_subspace="full"` or disabled coarse subspace, standalone prepare-Q and
`paged_write` launches are recorded too, but replay retains their original
Q, K, V and slot arguments. A later token can therefore read the first Q and
write the first K/V to the old physical slot. This is a host argument-lifetime
defect, independently of any native kernel numerical issue.

Retain whole-call plans only when Q preparation is fused and a decode write is
present. Nonfused routes keep normal dispatch and the existing per-kernel
compiled-launch cache. Calls without a write cannot take a prior write-plan
shortcut. Add write presence to geometry so HAS_WRITE specializations cannot
be confused when switching call modes. Default fused decode remains eligible.
No math, tensor geometry, thresholds or GPU kernel changes.

PR31 at c48a50ab repairs Q/K coordinate pairing in an older selector without
this replay implementation. This fix is separate and does not resolve those
pending coordinate issues. Do not claim full-subspace model correctness from
this admission fix. PR10 remains an older cache ABI, not a direct dependency.

## Evidence

The host test executes actual production AST for plan construction, public
dispatch and replay with distinct input objects, compiled/direct and hook
paths. Before fix: five failed assertions across admission, mode switch and
specialization cases. After fix: five tests pass in Python and Python -O.
Static buffer references remain unchanged; current fused inputs are rebound.
Baseline retained locally under /tmp/tli-replay-audit/baseline.py.

Both required gate scripts were attempted and blocked by missing `torch`.
No native Torch/Triton/GPU execution, model-quality result or timing claim.
Normal dispatch may cost more host time for nonfused configurations; default
fused replay is retained, but performance preservation is not measured.

## Reproduce / remote acceptance

```sh
python tests/test_nfa_eager_replay_admission_host.py
python -O tests/test_nfa_eager_replay_admission_host.py
git show 40756b6c925ad0caa5741ccbc305299fb8189567:sparse_attn/indexer/nfa_layer_batch_selector.py > /tmp/nfa-replay-before.py
NFA_SELECTOR_SOURCE=/tmp/nfa-replay-before.py python tests/test_nfa_eager_replay_admission_host.py
# The baseline invocation must fail; the two fixed invocations must pass.
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
python -m pytest -q tests/test_exact_batch_selection.py tests/test_frozen_region_runtime.py
```

Native acceptance still needs paired repeated-token selector/model execution
against forced normal dispatch: changing Q/K/V and physical slots, B1/4/32,
tail/full subspace, independent request gates, EOS/slot reuse, repeated page
reuse, and fresh CUDA capture. Check written KV locations, selected IDs and
attention outputs with existing tolerances. Compare the default fused route's
whole-step latency and nonfused dispatch cost; the listed existing native tests
alone do not provide this full regression coverage.
