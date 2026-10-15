# E140 historical observation limits and minimal recollection

The original JSON (`e670af6eda21d988d4eedcf458064bb72982bf33`, first commit
`096414368d2f7c102cb526a62c222e181b97620c`) remains unchanged. The v2 manifest
retains every legacy v1 artifact hash and pins that historical snapshot to this
commit. It separately records the later unannotated figure snapshot and current
SVG/PNG/PDF with a visible historical-diagnostic banner. Original bytes remain
recoverable from the recorded Git commits; annotation changes presentation only.
Neither the old nor annotated figures are repaired or accepted paper evidence.

`TL-E140-ASSET-PROVENANCE-20261015`: run
`python -m exp.paper.e140_asset_provenance` to verify current assets and unchanged
inputs against the manifest. Renderer default rejection remains mandatory;
explicit diagnostic rendering keeps a visible warning. Re-rendering changes
artifact identity and requires a reviewed manifest update, never promotion to
valid paper evidence by replacing hashes alone.

Confirmed collector defects:

- Old near selected the oldest left part of mid; near must be the right part
  nearest the query, excluding SWA. Unequal lengths prevent swapping labels.
- Old `[q-128,q]` SWA has 129 positions. v2 uses `[q+1-128,q+1)` and separately
  accounts for the sink/SWA union without double-counting short-prefix overlap.
- `far_worst_head` was a maximum of KV-group averaged probabilities, not a worst
  individual Q head. v2 names it `far_max_kv_group_mass`.
- O3 dropped its partial final page. v2 averages the actual tokens in every page.
  `reference_token_page_coverage` is reference-token coverage by chosen pages.
  `selected_mass_over_reference_topk_mass` is divided by reference mid TopK mass,
  not by all attention mass; v2 separately records `selected_full_attention_mass`.
- The Qwen3 path applies input_layernorm before self-attention. The old collector
  projected `hidden_states[L]` directly. This is a source-level mismatch with
  [HF Qwen3 v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py),
  not a measured claim about the unidentified old installed model build. v2 hooks
  actual q_norm/k_norm outputs and actual supplied RoPE values during forward.

V2 region alignment uses S=q+1, requested near length=max(bs,int(alpha*mid)),
floor-aligned near-left and clipping to nonprotected mid. This is explicit token
mass accounting, not an assertion that production block candidate pools contain
exactly those tokens for every nonaligned/sink-overlap case.

O3 remains a uniform page-average diagnostic, not production near-average/far-minmax.
The random arm uses one seed and unpaired RoPE coordinates. Query × KV-group rows
are repeated observations of four source sequences, not 1024 independent samples.
Original O3 aggregates do not depend on the erroneous O1 near slice, but share Q/K
and support-boundary risks. Existing numerical/causal conclusions are not certified.

No matching historical Q/K or per-token probability archive was found in the
accessible repository scan. Existing 64-token cross-group bins cannot reconstruct
arbitrary new cut positions; the folded SWA summary loses the boundary token.
No exact corrected numbers, interpolated estimates or relabeled figures were made.
Remote hosts were not searched. Reuse requires matching raw archives or recollection.

## Prepared command; not executed

First recover exact model/tokenizer/dataset revisions, original input/template and
document identities. Validate the installed HF path against a small real-model
reference before interpreting results. Use a fresh private output path; snapshots
can be large and must not be blindly committed or uploaded.

```sh
python agent_doc/guide/paper_figs/trace/e140_o_trace.py  # plan only, no GPU access
# On an independently approved host, after replacing every placeholder:
E140_DEVICE=cuda:0 E140_OUT=/FRESH_PRIVATE_PATH/e140_v2.json \
E140_MODEL_REVISION=PINNED_MODEL E140_TOKENIZER_REVISION=PINNED_TOKENIZER \
E140_DATASET_REVISION=PINNED_DATA QWEN_MODEL=/LOCAL_MODEL LB_DATA=/LOCAL_DATA \
python agent_doc/guide/paper_figs/trace/e140_o_trace.py --execute --save-qk
```

Minimal scope retains two tasks × two source sequences × two layers × sixteen
queries, with actual per-query/group rows and raw sampled Q/full K plus hashes.
Revision strings are declarations; local model/data provenance is still unverified.
CPU hook fixtures verify plumbing only. No GPU, model loading or recollection was
executed in this repair, and no service recovery is claimed.

### Follow-up: offline HF CPU parity (2026-10-15)

A subsequent check instantiated a real HF Qwen3 class with random weights on CPU
fp32 (transformers 4.57.3, torch 2.14.1+cpu), without downloading pretrained
weights. Native post-RoPE Q/K matched exactly; causal KV-group probabilities had
maximum absolute error 1.1920928955078125e-7. Thirteen affected tests passed both
normally and under `-O`; an injected RoPE error failed with exit 1 in both modes.
The small O3 fixture covers full selection/partial pages, not budget-constrained
ranking. This does not validate pretrained identity, bf16/GPU, old observations,
or scientific claims. Historical artifacts and their validity status remain
unchanged. Details and remaining conditions:
`agent_doc/advice/2026-10-15_e140_collector_cpu_validation_by_gpt.md`;
evidence: `exp/paper/evidence/observation-cpu-20261015/`.

### Follow-up: E140v2 recollection executed (2026-10-16, wys-4xh20 GPU1/GPU3)

Both collectors ran with `--execute` on the 4xH20 node (transformers 4.56.2,
bf16, declared local revisions with sha256 prefixes; see
`exp/trace/results/e140v2_observation_verdict.json` meta). Outputs:
`exp/trace/results/e140_o_trace_v2.json` (o3_raw stripped to a sidecar recorded
by sha256; npz Q/K snapshots kept private on the node per the snapshot rule) and
`exp/trace/results/e140_o4_select_trace_v2.json`. The O4 collector was rewritten
to the v2 contract (hooked q_norm/k_norm outputs + actual position_embeddings,
revision gating, `--execute`, `x`-mode output) before execution.

Panel-level re-verification (see verdict JSON for full numbers): (c) tail32 >
random32 > high32 survives with stronger margins (0.84/0.51/0.24); (d) far
starvation and K1=64 recovery survives (oracle 305.5 vs 128 vs 432 per query,
overlap 0.518→0.554); (a) the d>9K density rebound does not reproduce (monotone
decay); (b) far holds ~49% of nonprotected mid mass, not 83%. The paper O
section and `figs/fig_o_observation.py` must be rewritten against the v2 JSONs
before any figure is used as paper evidence; the historical-render gate remains
in force for the old JSONs.

### Merge follow-up: manifest v3 reviewed update (2026-10-16)

The annotation manifest (authored 2026-10-15 before the recollection) pinned
the annotated historical figures at the live `O_observation.*` paths. Merging
that state with the completed E140v2 recollection required the reviewed
manifest update this repository's own rule prescribes for re-renders: current
assets are now the v2-recollection renders (commit a8d9735, sources hashed in
`v2_recollection`), the annotated historical figures are pinned by git blob in
`historical_annotated_assets` (21b595d), and every legacy lineage section is
preserved verbatim. `fig_o_observation.py` renders v2 data fail-closed and no
longer calls `add_render_warning`; that banner remains valid for the legacy
assets only. No legacy bytes were modified. Run
`python -m exp.paper.e140_asset_provenance` to verify.
