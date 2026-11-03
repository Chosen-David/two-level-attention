# Prefill core-token interval slices

Base: 40756b6c925ad0caa5741ccbc305299fb8189567. Status: host validated; native GPU/performance pending.

Actual route: frozen_runner.capture_profiles / capture_core_tokens -> shared tail QK and per-head softmax -> _core_tokens_from_probabilities. Dynamic and static gate prefill both use this helper. Decode's request-batched selector is unchanged.

The core score reduction stays probs.mean((0,1)).mean(0). Near/far bounds are host integer contiguous intervals within [0,L]. Replace full-L arange, two boolean position compactions and score gathers by score[lo:hi], then add lo to the same torch.topk local indices. Preserve topk count, input order, dtype, normalization and the final combined D2H. No query/head averaging before scoring, new kernel, threshold change or altered seed budget.

At L128K, the removed int64 arange has 1MiB payload; the two masks/positions and gathered score vectors are additional source-level work. This is NOT measured peak VRAM or GPU speed. The remaining full probability mean, topk and D2H may dominate. Basic slicing is preferable to a new custom kernel or cached per-geometry index arrays here. Existing GQA broadcast/row-chunk candidates remain separate.

Validation: source AST host differential executes 405 configurations (length1/7/64/257/513, H1/4/8, random/ties/zeros, sink0/4/L+1, K0/1/64). Old and new outputs equal with the same host TopK facade; normal and -O pass. Allocation-contract test fails baseline (one arange) and passes candidate (zero). py_compile passes. Native CPU/CUDA pairing included but skipped locally: Torch/Triton/GPU unavailable. Host tie handling is not proof of native torch.topk tied-ID identity. No speed, quality or native numerical acceptance claimed.

Reproduce:
```bash
python tests/test_core_region_slices.py
python -O tests/test_core_region_slices.py
git show 40756b6c925ad0caa5741ccbc305299fb8189567:sparse_attn/policies/frozen_gate.py > /tmp/core-baseline.py
NFA_CORE_SOURCE=/tmp/core-baseline.py python tests/test_core_region_slices.py
PYTHONPATH=. python -m pytest -q tests/test_frozen_gate_profile.py
compute-sanitizer --tool memcheck python tests/test_core_region_slices.py
```
Baseline allocation test is expected to fail; output pairing should pass. Native tests require exact IDs including ties/empty regions; do not weaken tests if a backend uses different tied ordering.

Paired helper timing on an otherwise idle CUDA device (reports raw host wall incl helper D2H; not full prefill):
```bash
python - <<'PY'
import runpy, time, json, torch
m=runpy.run_path('tests/test_core_region_slices.py')
new=m['load_core'](torch); old=lambda p,g,k:m['reference'](p,g,k,torch)
# Timing reference must preserve baseline's single combined D2H.
import os
os.environ['NFA_CORE_SOURCE']='/tmp/core-baseline.py'
b=runpy.run_path('tests/test_core_region_slices.py');old=b['load_core'](torch)
g=m['Geometry'](); rows=[]
for length in (32768,65536,131072):
    torch.manual_seed(20261103)
    p=torch.rand((1,32,256,length),device='cuda')
    assert old(p,g,64)==new(p,g,64)
    for fn in (old,new):
        for _ in range(3):fn(p,g,64)
    for cycle in range(5):
        for label,fn in [('old',old),('new',new),('new',new),('old',old)]:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();fn(p,g,64);torch.cuda.synchronize()
            rows.append(dict(length=length,cycle=cycle,arm=label,
                wall_ms=1000*(time.perf_counter()-start),
                allocated_peak=torch.cuda.max_memory_allocated(),
                reserved_peak=torch.cuda.max_memory_reserved()))
    del p
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(),rows=rows)))
PY
```
This synthetic probability input only measures the helper. Before accepting a performance claim, pair complete profile_regions_and_core_tokens and full Qwen/GLM prefill on identical real inputs, same backend/scale/seed/rows, compare all region statistics, gates, core IDs and model outputs, measure allocated/reserved peaks and TTFT. Include short prompts, static/dynamic gates and request reuse. Profiler should confirm eliminated compaction/gather work rather than infer launch counts from Python.

Overlap: PR31..35 frozen_gate.py freshly identical to main; previous unchanged PR26..30/file and PR10/older ABI comparisons reused. No same-topic patch or dependency found. All original69 branch heads unchanged. No main push/merge or other branch overwrite.

Sources: https://docs.pytorch.org/docs/2.15/generated/torch.nonzero.html (CUDA synchronization); https://docs.pytorch.org/docs/2.15/generated/torch.topk.html (ties not guaranteed stable). No assumption that those docs identify the remote installed version.
