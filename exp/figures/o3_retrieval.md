# O3 CPU scoring diagnostic and E0 figure bridge

This is a review candidate, not a completed O3 experiment. It uses the existing
`collect_trace_lb_v.py` layout (post-RoPE `k[S,Hkv,128]`, sampled
`q[Nq,Hq,128]`, integer `qpos[Nq]`), with lazy CPU-only PT loading. NPZ input
needs NumPy; PT additionally needs PyTorch; `--plot` needs Matplotlib. No model,
CUDA, remote job, or production indexer is invoked. Keep traces and protocols
outside the public repository if their contents or locations are private.

The bounded diagnostic compares tail/high/paired-random **32 coordinates / 16
complete split-half pairs**. The pair relation reuses E85d/E85e's construction.
Unlike E85d's sum(Q)-then-softmax reference, this implements the per-Q-head
softmax then KV-group probability mean already used in
`analyze_p0p_perlayer_potential.py`. Full-reference probabilities use original
128-dimensional Q/K with scale `1/sqrt(128)`. Only candidate K is min/max QDQ
per token in float32; ties-to-even applies to the computed float32 quotient,
not ideal real-number half ties. All three arms use the same rule. This is
not bitexact production BF16/Triton validation.

**Scope restriction:** all causal keys, global TopK, no forced protection,
no L1 pages and no regional quota. It measures scoring fidelity, not complete
TLI selection, A2 natural generation, task quality, or speed. The source
collector is prompt-only, not a decode trajectory; no O2 drift conclusion is
available. No dynamic alpha/beta/gamma or dynamic gate is added. The production
`subspace=random` remains unchanged and unpaired.

Freeze an explicit `protocol.json` BEFORE confirmation analysis. Required keys:

- `run_id`, exact `model_revision`, `dataset_revision`, `producer_commit`, `split`.
- `rope_layout: "split_half_128_descending_frequency"`, `rope_receipt_path`,
  `rope_receipt_sha256`. The receipt must record checked checkpoint RoPE geometry
  and frequency ordering. The tool checks its bytes, **not the truth of the
  reviewer's declaration**. This is not a portable assumption for other models.
- `data_kind: "synthetic"` or `"observational"` (always unverified identity).
- `query_phase: "prefill"`, `support: "all_causal_global_topk"`,
  `quantization: "minmax4_k_float32"`.
- Positive integer `budget`, `bootstrap_repeats`, nonnegative `bootstrap_seed`;
  `pair_seeds` with at least two distinct nonnegative integers, frozen in advance.
- `traces`: each entry has `path` (relative to protocol or absolute), exact
  `sha256`, anonymous `sample_id`, `document_id`, `input_ids_sha256`, zero-based
  integer `layer`, and explicit nonempty `query_indices` into sampled Q rows.
  Same sample/layer cannot repeat; cross-layer sample identity must agree.
  A trace hash cannot be relabeled as another sample/document/layer; identical
  input hashes cannot be assigned to different documents to inflate N.
  Identity labels must be path-free public tokens; placeholder and floating
  labels such as UNKNOWN, main, pending and whitespace are rejected.

The old collector `meta.json` alone is rejected. Recover missing identity from
actual original input IDs, checkpoint receipt and producer logs; do NOT invent
hashes/revisions, rename old exploration as confirmation, or add guessed defaults.
If recovery is impossible, keep the trace as historical evidence and request only
the necessary already-authorized collection after E0 freezes. This script's
synthetic tests demonstrate mechanics, not that E0 provenance has been verified.

From repository root, with a reviewed private protocol and NEW output path:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tla-mpl XDG_CACHE_HOME=/tmp/tla-cache \
python exp/figures/o3_retrieval.py --protocol /approved/private/o3/protocol.json \
  --out /approved/private/o3/analysis-01 --plot
python exp/figures/figure_data_contract.py \
  --bindings /approved/private/o3/analysis-01/figure_bindings.json \
  --check /approved/private/o3/analysis-01/figure_data.tsv
```

Outputs: per-observation selection IDs, reference IDs, recall/Jaccard, retained
raw-reference mass and worst Q-head mass in `selection_metrics.jsonl`; copied
path-free protocol summary; `summary.json`; E0-column TSV and source bindings; SVG/PNG; pending
`validation.json` and an **analysis-only** receipt. Existing producer run-cap,
treatment, scorer and independent verification manifests remain authoritative.
No artifact here upgrades them or replaces the existing E119/RULER validators.
The final directory is published atomically only after all artifacts and the
receipt exist, using Linux `renameat2(RENAME_NOREPLACE)`. Existing destinations
are never replaced, including an empty directory created by another writer.
Failures retain a hidden `.o3-pending-*` staging directory for diagnosis, not a
published output. Unsupported platforms/filesystems fail rather than falling
back to replacing a destination. All outputs must be outside this repository;
synthetic runs are explicitly labeled and cannot enter repository results.
The protocol is read once; its byte hash is recorded. Only allowlisted metadata
is exported, without raw paths or arbitrary extra fields. Caller-supplied IDs
must be anonymous; their authenticity is still not certified by the tool.

Statistics: mean over observations within document, then equal-document mean;
percentile bootstrap resamples independent document IDs, never heads/tokens.
The same resampling seed pairs documents across arms. Each random seed is a
separate arm/interval; its CI describes document sampling, not seed uncertainty.
Compare the displayed seeds as seed variability; do not select a favorable seed.
Within-document layer/query sampling and document task weights must be reviewed;
this tool does not implement task-balanced population inference or paired-delta
noninferiority claims. At least two documents are required, which is a software
minimum, not a claim of adequate statistical power. Full causal support and
short-sequence budget clamp are recorded in the raw rows. Input/loaded/computation dtypes are recorded in the
analysis receipt. Non-real-floating Q/K and float32 conversion overflow fail. Ties use lowest key ID.

The generic `figure_data_contract.py` adapter accepts `figure-bindings-v1`:
`status: frozen`, `sources` mapping names to exact `path/sha256/run_id`, and
`cells` containing all E0 text columns plus `source_id` and `fields` JSON pointers
for estimate/CI endpoints/N. Use the generated bindings as a concrete example.
It exports with `--export NEW.tsv`, or checks with `--check EXISTING.tsv`. It
rejects missing planned cells, duplicate cells, source mismatches, floating
labels, mixed modes/units in a panel and nonfinite/absent statistics. It only
checks declared closure: a binding that omitted an arm from the outset still
requires an independent comparison against the human guide. It never rescales,
rescoring, bootstraps or fabricates missing CI/N for historical JSON. It reads
and hashes the same source snapshot. Publication is no-overwrite and atomic
for a single TSV; O3 additionally atomically publishes the complete directory on supported Linux systems.

Run CPU tests:

```bash
python -m unittest discover -s exp/figures/tests -v
python -O -m unittest discover -s exp/figures/tests -v
```

Independent review must check production QDQ differences, the meaning of causal
support/global reference, original trace identity, document weighting and RoPE
receipt. The PT loading path has not been executed in the current torch-free
CPU environment. Only synthetic NPZ inputs have been run; real figure values,
GPU parity, model quality, memory and latency remain untested.
