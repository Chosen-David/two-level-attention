# Figure-by-figure engineering inventory (review candidate)

Pinned and rechecked remote main: `831fabfb0458d3c86b2703af9d6b63ca8bca8e23`.
Read all three `agent_doc/guide/paper_figs/*/*.md`, TASK, S-T022 and active advice.
No guide edits, raw private materials, GPU runs or experiment acceptance. The
local checkout contains historical artifacts, not remote current job state.
`asset_inventory.json` binds selected existing paths/hashes and explicitly missing
paths. Presence/hash is not independent scientific validation.

| Figure | Reuse: script / data / manifest | Real gap and existing owner/task |
|---|---|---|
| E0 (all) | `benchmark/LongBench/run_cap_manifest.py`; `analyze_e119_ruler128k_formal.py` + `test_e119_crossarm_identity.py`; existing `.runcap.json`, treatment/scorer/receipt chains | No new figure columns/point-to-source bridge found in existing figure scripts. This candidate adds a thin adapter, not replacement scorer/manifest. Upstream E0 evidence and independent verification still required. |
| E1 | `benchmark/LongBench/{pred,eval}.py`, `benchmark/RULER/pred_ruler.py`; E116b LB and RULER generations, E119 summary and raw/manifest/receipts | Check exact static dual32 configuration, full 16-task LBv1 / 503 LBv2 / RULER coverage and independent confirmation; model revisions and 32B/Llama coverage not established here. S-T019/S-T007/S-T022 Opus; external rows S-T023 DeepSeek. |
| E2 | `exp/bench/e131_arm_bench.py`, E126 timing, E128/E131/E134 cost JSON + SHA sidecars | Single-layer or summed-stage medians not request wall; matched quality/timing identity, raw TTFT/TPOT/request traces remain open. E139 wall referenced in planning absent from checked main. Reuse S-T025/E127 and phase advice; Opus. |
| E3 | baseline producer adapters and S-T023; Quest/FASA/MInference artifacts and existing cap/scorer gates | Verify faithful-port, native vs unified backend, actual budgets and full request boundary separately. HISA identity unresolved; FFD/ClusterKV support not inferred from names. DeepSeek baseline only. |
| A1 | `analyze_e125_screen_verdict.py`, `analyze_e128_tradeoff.py`; E123 raw, E125 screen verdict, E128 tradeoff | Historical finite-candidate ranking reusable after identities; no proof of independent confirmation or uniform paired request cost. Old E98 plots not current parameter evidence. S-T022. |
| A2a | production `tli_l2_subspace=full`; E133 verdict; E134 l1sub/l2full predictions with treatment/runcap; E134 performance reverdict | FS/SF/SS already have author artifacts. FF is missing experiment acceptance, not necessarily missing implementation. Actual coordinate/dtype/identity and matched wall must be closed. S-T022/E134. |
| A2b | `_subspace_indices`, `analyze_e85d_selection_methods.py`, `analyze_e85e_static_pair.py`, E90 subspace JSON | Existing random is unpaired fixed32. Offline complete-pair construction in this candidate is O3-only; production paired arm and natural-generation acceptance NOT delivered. Tail widths/seed list and confirmation frozen first. S-T022. |
| A3 | `analyze_e88_tight_partition.py`, `analyze_e130_gamma_off_grid.py`, E130 gamma-off/e2e verdict | Old gamma-off/grid not same-candidate pure-quota control. Fixed L1 support/normalization, same-scorer global/regional L1 and IDs/cost still needed. S-T009/E120 and existing L2-cost advice. |
| A4 | E129 ablation, E137 calibration/alignment/dynamic verdict and current gate implementation | G2 no-refill and frozen matched IDs/trajectory not inferred from switches. Calibration/request wall and independent quality incomplete. Dynamic work paused; do not dispatch or extend. S-T021 paused + existing E137 advice. |
| A5 | `analyze_e135_sinkswa_grid.py`, E135 grid/only/audit JSON | 2048 trace and 256-only do not supply 1024 equal-budget M2/M3 task comparison. Need frozen naive-middle policy and actual supports; reuse S-T009/E120, Opus. |
| A6 | `sparse_attn/ops/eager_prefill.py`, E130/E131 phase scripts and existing phase advice | main dense QK/PV masked reference is not sparse speedup. Prefix invariance, actual route and four-phase matched request evidence pending; E139 raw missing. Candidate-L2 and shared-cache engineering only after same-QK semantics. S-T025/E127/phase advice. |
| O0 | `collect_trace_lb_v.py` post-RoPE QKV capture, archive index for external trace locations | Old meta lacks document/input/model/producer identity; external trace bytes unavailable locally. Must recover actual receipts or retain blocked state. No new collector. |
| O1 | Existing QKV collector, E135 harness | Frozen bins, protected overlap accounting, actual rounded boundaries, per-token density and independent-document uncertainty not in current plot-ready contract. Reuse A3/A5 traces only with matching identity. S-T009/E120. |
| O2 | E137 author calibration JSON; QKV collector prompt observations | No verified raw-head/actual-ID/decode-drift closure; prompt-only trace cannot prove drift. Pause gate expansion, keep historical observations. S-T021 paused / existing E137 advice. |
| O3 | E85d/e85e complete pair construction; `analyze_p0p_perlayer_potential.py` per-Q softmax reference; old E133/E90 data | E85d sum(Q)-softmax and mask agreement are not new O3 reference/Jaccard. Candidate offline scoring diagnostic fixes this limited comparison and emits point lineage; full production L1/L2 replay, real trace identity and review still pending. S-T022/E134. |

No new TASK ID or competing task list. Historical guide/planning rhetoric is not
acceptance: CI includes zero != equivalence; bitwise mask agreement != Jaccard;
4x width reduction != 4x selector speed; author wall claims without raw identity
cannot enter measured main figures. Frozen JSONs remain untouched.

Candidate CPU evidence: 20 tests pass in ordinary and `-O` modes; independent
scalar GQA oracle, full-pair/seed, tie/clamp/mass, NaN/qpos, document-weighting,
hash and source-binding failures, missing CI/arm, duplicate/mixed/floating labels,
repeatable NPZ-to-TSV integration. Synthetic SVG/PNG preview was visually checked;
it is a software fixture, not research data. Exact commands and limitations in
`exp/figures/o3_retrieval.md`. Independent review and real PT/GPU checks pending.

Independent review revisions: reject duplicate trace/input relabeling that inflates N,
placeholder identity and non-real Q/K; check float32 conversion; snapshot protocol
once and export allowlisted path-free metadata; atomically publish whole output
without replacement on Linux. Initial failed probes and final reviewer receipts
are delivered separately as review evidence, not scientific results.
