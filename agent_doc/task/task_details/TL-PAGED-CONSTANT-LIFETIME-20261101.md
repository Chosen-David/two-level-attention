# TL-PAGED-CONSTANT-LIFETIME-20261101

Source: user-requested current-main performance audit. Base: `c11d357f58af4772b6bdfee5bc9fd7e6d3e09aee`. Status: host-validated candidate; native CUDA, selector parity and memory/latency measurements pending.

The new `paged_region_tli.region_selection_compact` passes `k_qat[:, :logical_length]` to `compute_selection_graph` on every non-both-skip decode. The existing Graph constant cache includes that exact length in its key and retains every entry on the model's indexer. Consequently each new decode length adds another entry, even after the request completes. With the default non-sparse selection policy, each includes an int64 arange of the entire logical length. Both-skip exits before this selector and is unaffected.

Fix: paged eager explicitly requests transient constants. Default fixed-capacity Graph callers still retain and reuse every prior address; transient calls can reuse an already pinned matching entry but neither insert nor evict entries or replace the diagnostic alias. Transient capture is rejected. Constant construction, scores, ranking, IDs, quotas and attention kernels are unchanged.

Compared alternatives: `empty_cache()` cannot reclaim live tensor references; a bounded global LRU could invalidate old Graph addresses; capacity bucketing would require broader normalization/mask/shape validation. The scoped lifetime option uses existing PyTorch tensors and kernels, adding no custom kernel or dependency. PyTorch's Graph address-lifetime requirement: https://docs.pytorch.org/docs/2.15/notes/cuda.html#graph-memory-management .

Evidence: the actual baseline constant block and candidate helper executed with NumPy storage for 64 changing 128K lengths under each dense/sparse mode: all 128 constant dictionaries matched. Indexer-owned entries were 128 before and 0 after. Baseline NumPy payload was 69,291,904 bytes across those two modes. Dense ar_tok alone accounts for 67,125,504 bytes per layer across the 64 lengths. These are host ownership/array-payload observations, **not CUDA allocated/reserved/peak memory or measured speedups**. Seven host regression methods pass both ordinary and `python -O`; two native CUDA methods are skipped locally because Torch/CUDA are absent. No model quality claim.

Reproduce on this branch:

```bash
git show c11d357f58af4772b6bdfee5bc9fd7e6d3e09aee:sparse_attn/indexer/tli_indexer.py > /tmp/tli-constants-baseline.py
python agent_doc/results/paged-selector-constants-20261101/probe_constants.py --baseline-source /tmp/tli-constants-baseline.py
python tests/test_paged_selector_constant_lifetime.py -v
python -O tests/test_paged_selector_constant_lifetime.py -v
```

Native gate in a complete supported CUDA environment (no skips accepted for CUDA methods):

```bash
PYTHONPATH=. python -m pytest -q tests/test_paged_selector_constant_lifetime.py tests/test_paged_region_tli.py tests/test_combined_selector_graph.py tests/test_exact_graph_lifetime.py
```

The new native cases compare exact selected IDs/counts at changing lengths and replay two different-capacity graphs after transient eager calls. Before acceptance also pair base/candidate whole paged decode on the same GPU for four frozen gate states, repeated page wraps and a new request; report allocated/reserved/peak memory and TPOT separately. This patch only removes indexer constant retention, not projection scratch, bounded geometry-cache entries, KV residency or allocator reservation.

Overlap: PR10 `2449f56983d7e97c5fe1c4df300c2b50b0c16b81` has neither this Graph constant cache nor this new paged path. PR12/13/17/22 patches do not address this lifetime. No dependency on those unmerged PRs. No guide/manuscript changes.
