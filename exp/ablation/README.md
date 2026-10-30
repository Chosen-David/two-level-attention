# NFA ablation work status

Current user scope: HotpotQA and GovReport quality only. Historical reuse first.
F protection-value ablation was cancelled before formal dispatch. All D and
other performance measurements are deferred. New quality settings keep sink4 /
SWA128; historical groups retain their actual settings.

`near_length_paired.py` and `plot_near_length.py` reproduce the independently
admitted historical E comparison (2048/4096/8192). See project results directory
`agent_doc/results/nfa_ablation_20261030/reuse_E_review.md` for exact limitations.
`reuse_inventory.py` and `reuse_frozen.py` are read-only rechecks; their other
historical groups still need independent per-scope admission.

The strict selector / F matrix / idle pool are preserved experimental work, NOT
admitted for formal results or production.69 CPU checks and23 short GPU smoke
requests were completed before cancellation; no6400-request formal queue ran.
Potential outstanding selector issue for any future reuse: when far is empty
and sink is not page-aligned, assigning a page via `page_start >= far_hi` may
classify the mixed sink/near page as far. This needs a dedicated quota test and
review before strict_ablation_v1 is reused. It does not affect production
compute_mask defaults or the historical E/C results. Do not silently promote
these experimental strict semantics to main or relabel historical predictions.
