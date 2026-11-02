# GraphReady selector host admission

Base: d4d35be8b1807a79f41b08979cffba172c0a55b9. Scope: compute_selected_ids_by_layer only.

Problem: non-both-skip batches construct length_t[B] and pass length_t>0 to torch._assert_async. PyTorch CUDA requires one element, so B>1 fails before per-request selection. query_start_loc[B+1] and seq_idx[B] are only used to assert their own construction, never passed to the consumer. Source contract: https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/TensorCompare.cu (_assert_async_msg_cuda).

Change: validate the existing Python length list directly, before Q gathering. Reject nonpositive active lengths with ValueError. Remove the three unused device metadata tensors and their comparisons/assertions. Keep persistent request IDs, valid slot filtering, per-request selection, both-skip shortcut and first-decode behavior unchanged. This fixes admission only, not the separate current-Q/K, seed, per-head support, residency or Graph contracts. PR10 has another cache ABI; PR26/27/29 still contain this assertion and address independent issues. No unmerged dependency.

Evidence: execute actual coordinator AST with selector stub and a NumPy facade enforcing PyTorch's documented source scalar requirement. Baseline raises at B2; candidate host regression passes normal and -O for B1/2/3/8/32, ragged pending lengths, reordered high persistent IDs, padding holes, removed IDs, nonpositive admission, empty batches and both-skip bypass. Two native coordinator tests included but skipped locally (no Torch/CUDA). Native tests still stub selection and do not certify full-model correctness. py_compile passes. No numerical tolerance changed.

The removed work consists of three device metadata constructions and three assertion expressions per non-both-skip layer. Actual launch counts, GPU latency, synchronization and memory savings have NOT been measured. Prefer existing host integer validation over device reduction/all(), which would keep unnecessary metadata/launches. Lengths are already Python integers from pending/state metadata, so validation introduces no device-to-host read.

Reproduce:

```bash
python tests/test_graph_selector_admission.py
python -O tests/test_graph_selector_admission.py
git show d4d35be8b1807a79f41b08979cffba172c0a55b9:sparse_attn/cache/paged_region_cache.py > /tmp/nfa-admission-baseline.py
NFA_ADMISSION_SOURCE=/tmp/nfa-admission-baseline.py python tests/test_graph_selector_admission.py
compute-sanitizer --tool memcheck python tests/test_graph_selector_admission.py
```

GPU acceptance remains pending: on a fixed device/software stack compare baseline/candidate eager and GraphRunner.step at B1/2/8/32 with unequal lengths and mixed frozen gates. At B>1 baseline admission failure is expected, not a timing baseline. At B1 compare successful complete-step latency and allocations with identical selected IDs/model outputs; include EOS/removal/replacement and new capture. Broader pre-existing failures must be recorded separately, not bypassed or counted as a performance win. Do not use older gate-quality reports as strict selector evidence.
