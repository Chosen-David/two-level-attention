# O3 CPU diagnostic / figure data bridge: executable handoff to Opus

Review candidate based on main `831fabfb0458d3c86b2703af9d6b63ca8bca8e23`.
No GPU or remote model job launched; no publication or independent acceptance
claimed. Reuse S-T022/E134 (O3/A2) and the existing E0/phase/cost advice. See
[figure inventory](../results/20261014-figure-data-candidate/gap_inventory.md)
and [CLI/input contract](../../exp/figures/o3_retrieval.md).

Opus owns our method. DeepSeek owns external baselines only. Dynamic alpha/beta/
gamma and new dynamic gate remain paused. Do not use this diagnostic as evidence
that production paired-random or A2 natural-generation controls are implemented.

1. First inspect existing completed artifacts and original trace/producer records;
   do not rerun completed cells merely because figure documentation changed.
   Existing `collect_trace_lb_v.py` is reusable; old `meta.json` alone is rejected.
   Recover anonymous document/sample IDs, actual input_ids SHA, exact model/data/
   producer identity and checked RoPE split-half/frequency receipt from real
   evidence. If unavailable, mark BLOCKED; don't create guessed defaults.
2. Freeze a small already-planned prompt-QK diagnostic subset, two or more
   independent documents, >=2 pair seeds, query indices, budget and bootstrap
   scheme before viewing confirmation results. These are missing choices, not
   a request to extend the experimental matrix. Keep private raw trace and
   protocol outside the public repo. No need for a GPU to analyze existing traces.
3. CPU commands after the candidate is reviewed/landed and input is frozen:

```bash
python -m unittest discover -s exp/figures/tests -v
python -O -m unittest discover -s exp/figures/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tla-mpl XDG_CACHE_HOME=/tmp/tla-cache \
python exp/figures/o3_retrieval.py --protocol /approved/private/o3/protocol.json \
  --out /approved/private/o3/analysis-01 --plot
python exp/figures/figure_data_contract.py \
  --bindings /approved/private/o3/analysis-01/figure_bindings.json \
  --check /approved/private/o3/analysis-01/figure_data.tsv
```

Paths above are deliberate private-storage placeholders, not available inputs.
Use a fresh output directory. Existing output is rejected. Use the existing
Torch environment only for PT loading; NumPy NPZ runs on CPU alone. Linux atomic no-overwrite publication exposes the final directory only after
completion; failures retain hidden staging evidence, never a final output.
Outputs must be outside the repo. Protocol summaries omit paths and extra fields;
identity remains explicitly unverified. Analysis receipt is not the producer manifest. Retain the existing
run-cap/treatment/scorer/independent-validation chain.

4. Hand output IDs, summary, TSV, source hashes and SVG to an independent reviewer.
   Review support/global TopK versus real L1/regional/protected selector, float32
   QDQ versus production dtype, GQA oracle, exact trace provenance, document
   sampling/weights and seeds. The figure is explicitly scoring-only and pending;
   no quality, GPU memory, latency or L1 page recall claim. Preserve negative
   findings. Only after scientific/identity validation may a suitably scoped
   panel enter PPT/paper; never insert this run's synthetic test preview.
5. For genuinely missing previously-authorized GPU cells, Opus must recheck
   **current assigned/free capacity**, job ownership, scheduler reservations and
   existing locks with the resource owner. A momentarily idle `nvidia-smi` is
   not permission. Match VRAM, model/dependency compatibility and frozen budget;
   parallelize independent ready cells only on available assigned cards. No
   preemption, duplicate completed cells, extra paid GPUs, occupancy jobs, or
   larger matrix to fill cards. Performance runs require their existing exclusive
   locks and uncontended timing protocol. This advice is not a new GPU dispatch.

Current local evidence: 20 normal and optimized-mode CPU tests, synthetic NPZ
end-to-end runs and deterministic TSV/SVG/PNG. Real PT loading was not tested
because local torch is absent; real trace/model/GPU validation remains open.
This is an implementation candidate, not a completion update to experiment TASKs.
