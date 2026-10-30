# Device revision through row views — 2026-10-30

Base main: 8ff7e78574a0dd8246ac1a5b677bd625939d4dc8. Prior main8718c90 and branch0516c3e assessments superseded where integration changes code. Local old evidence files were removed by workspace maintenance; prior conclusions are reused from exact-SHA conversation records, not represented as locally rerun tests.

## Mechanism and fix
DeviceDecodeInputs.inject uses two ordered kernels (_advance then _write_mask), fixing the previously reported cross-CTA race. It also increments lengths._tli_device_revision because Triton writes do not advance Torch _version. The Qwen legacy row path passes vl[b:b+1] to compute_mask_graph. Tensor view Python attributes do not inherit the parent's custom revision, so cache entries keyed by pointer/version/direct-view revision can remain stale across eager device-input steps for ordinary versioned tensors.

Lookup and store now include both direct-view and storage-base revisions. Same-step reuse remains; changed revision invalidates. Existing mutation version, capture namespace and stream keys are preserved. Inference tensors still refuse caching. This does NOT fix the separately recorded inference-mode sharing overhead or make mixed ring indexing batch-parallel. No extra GPU kernel, tensor allocation or transfer is introduced by the fix.

## Validation
Actual cache functions extracted from source: 4 host-facade tests pass normal and -O; 88 row/step subcases include fresh and nested views, B1/2/8 over4steps. Baseline main fails66 cross-step subcases; direct revisions, version changes, geometry/capture/stream keys and inference refusal are controls. Native CPU tensor view tests added to tests/test_geometry_capture_scope.py, not run locally (no Torch/pytest/Triton/GPU). Host facade validation is not native Torch or GPU validation.

Remote commands:
- python tests/test_geometry_device_revision_host.py
- python -O tests/test_geometry_device_revision_host.py
- .venv/bin/python -m pytest -q tests/test_geometry_capture_scope.py tests/test_quality_decode_inputs.py tests/test_quality_graph.py tests/test_quality_graph_cuda.py

Before acceptance, run legacy eager/device-input multi-step parity under both no_grad (versioned buffers) and inference_mode, ragged B1/2/8, EOS/reset; compare selected IDs/masks and outputs at unchanged thresholds, then Graph recapture. Correctness fix; no measured speedup or memory/quality claim.

## Coverage
Fresh43 branches and16 open PRs, all16 discussion/review timelines empty; all42 non-main ancestor relations/unique commit lists/head dates refreshed against new main. PR10 remains2449f56983d7e97c5fe1c4df300c2b50b0c16b81 (31ahead203behind), PR16 remains1bef6dec9d9dc2f6d5e646d87b1ebb38a81541f4 (1ahead27behind). This revision mechanism was newly introduced by main8ff7; older PRs do not cover its sliced-view use. No unmerged dependency.

Incremental0516c3e, paged1b7dbd6 and GLM e5d0fd9 are now main ancestors, historical snapshots; no transplant. Main adds GLM NoPE frozen-consumer support, pre-commit paged validation, cache-owned both-skip geometry and B>1 exact long-capacity rejection. Paged implementation remains opt-in eagerB1, not Graph/batched latent MLA; main retains logical lowdim scratch and selector overhead. Physicale55349f remains12ahead27behind, unmerged; its aggregate verdict and raw-evidence limitations unchanged. opt/l2-fine-topk historical ancestor, opt/ako-pool-score and e175-skip-ring-kv unchanged independent work (2ahead213behind and3ahead230behind). Other unchanged actual-code assessments reused by exact SHA. No inference of live developer processes.

Decision: isolated main-based correctness PR; no main push/merge or other-branch overwrite. Prior unsafe tiled branch should no longer be reported as a current-main issue. GPU acceptance still pending.
