# TL-PAGED-COMPACT-BOUNDS-20261101

Status: host guards validated; real Torch CPU/CUDA tests pending.
Base main: 27b06b67cb7286923ed6b8b80bd6ef33657a57c0.
File: sparse_attn/ops/paged_region_kernels.py; baseline blob d8aad9694bd17765277add634f53a2f8e07511de.

## Defect and scope

paged_attention_selected clamps IDs to [0,L-1] only for its visible-mask check.
It then sends original IDs to _attend. An active ID >= L can pass when L-1 is
visible (SWA normally is); an ID beyond the pointer-table allocation can cause
an out-of-bounds pointer-table read. Active negative IDs can pass when token 0
is visible and are subsequently treated as padding despite being included in
counts. Negative counts and counts above selection width also lacked rejection.
The normal canonical selector should not produce these malformed inputs; no
claim that current normal generations or historical quality measurements are
corrupted. This fixes the exported consumer boundary.

Reject active original IDs outside [0,L), retaining existing hidden/reclaimed
visibility checks; require each count in [1,width]. Inactive padding is ignored,
including 0/-1 or arbitrary unused values. Fold conditions into the two existing
torch.any gates: no additional host synchronization gate or altered attention
arithmetic, but additional elementwise tensor work still needs GPU measurement.
Do not remove validation to make profiling numbers look faster.

## Validation

Actual wrapper AST prefix through all guards is executed using NumPy tensor
doubles. The kernel/body after validation is not executed by host tests.
Baseline: six failing assertions including three out-of-range ID subcases.
Candidate: eight host test methods pass ordinary and -O, including 189 valid
geometry/width cases (378 head rows), masked inactive garbage, hidden IDs,
zero/negative/excess counts. Two full-wrapper tests (Torch CPU and CUDA/Triton)
are skipped because this environment lacks Torch/Triton. No numerical threshold
changed; native valid-output oracle is per-head Torch softmax over the selected KV.
No GPU sanitizer, real memory, latency, model-quality or strict-selector result.

Commands:
```sh
python tests/test_paged_compact_bounds.py -v
python -O tests/test_paged_compact_bounds.py -v
git show 27b06b67cb7286923ed6b8b80bd6ef33657a57c0:sparse_attn/ops/paged_region_kernels.py > /tmp/paged-baseline.py
COMPACT_GUARD_SOURCE=/tmp/paged-baseline.py python tests/test_paged_compact_bounds.py HostGuards -v
```
The baseline host command should fail six assertions. Do not run malformed
baseline inputs on CUDA: they may dereference an invalid pointer table.
On the fixed branch with CUDA/Triton:
```sh
PYTHONPATH=. python tests/test_paged_compact_bounds.py -v
compute-sanitizer --tool memcheck python tests/test_paged_compact_bounds.py NativeWrapper.test_cuda_wrapper -v
```
Require both native tests to PASS rather than skip. Full model, ragged batches,
EOS and repeated page reuse remain outside this narrow input-guard test.

## Remote overlap and continuation

Independent of PR10 (file absent), PR23 and PR24 (both retain exact main kernel
blob). Other unchanged PR/branch code evidence reused; no new dependency.
All 58 heads/57 development branches were unchanged before publication; all
24 existing PRs open, fresh reviews/issue/inline comments empty. Runtime jobs
cannot be inferred from branch timestamps. Intermediate baseline stays /tmp.
Known GraphReady padding req_idx=-1 and selector integration defects are separate.

Performance audit leading here: _protected_ids builds int64 arange(L), a fresh
visible[L], and boolean selection even for bounded sink+SWA. Both-skip also
builds visible[L]/arange(L). Attention recreates another visible[L] and uses two
torch.any Python gates plus int(count.max()). Thus compact IDs alone do not
establish a graph-safe/O(K)-only wrapper. canonical_selection then does two sorts
on [1,Hkv,Kmid+Kprotected]. Direct interval IDs and analytic O(K) visibility are
next candidates; compare existing bounded PR10 helpers and selection_pack before
adding kernels. No performance optimization is claimed by this guard patch.
