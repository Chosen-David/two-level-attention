# E140 historical observation limits and minimal recollection

The original JSON (`e670af6eda21d988d4eedcf458064bb72982bf33`, first commit
`096414368d2f7c102cb526a62c222e181b97620c`) and existing O_observation figures
are preserved byte-for-byte. `e140_validity.json` records their hashes. They are
historical diagnostics, not repaired or accepted paper evidence.

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
