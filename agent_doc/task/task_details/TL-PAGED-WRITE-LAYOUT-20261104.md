# TL-PAGED-WRITE-LAYOUT-20261104

Status: pending native CUDA/Graph and model validation. User-authorized latest-main audit; single agent, no main push/merge.

## Mechanism and scope

Baseline main `40756b6c925ad0caa5741ccbc305299fb8189567`.
`glm4_moe_lite_attn_patch.py` obtains V by splitting the final dimension of
the joint K/V projection, then passes that view to `cache.write_decode` ->
`paged_write_decode` -> `_paged_write_kernel`.
For T=1, the V head stride is `qk_nope_head_dim + v_head_dim`, whereas the
kernel addresses `b * HK * DV + h * DV + vd`. The first head can be correct
while subsequent heads (and batch rows) read intervening K data. CPU indexing
honors strides, so a CPU-only execution of the wrapper's reference path misses
this defect. This is a source-bound counterexample, not a measured model failure.

Normalize K and V with `contiguous()` at the standalone CUDA launch boundary.
Keep the existing kernel ABI, return values, dtype, PAD handling and selection
semantics. Already packed tensors retain identity. The Qwen default fused writer
is a separate path and receives packed single-token projections; it is not changed
or claimed to accept arbitrary strided inputs here.

## Alternatives and performance limits

The existing PyTorch layout primitive is preferred for this correctness patch;
no extra kernel implementation or dependency. A stride-aware paged writer could
avoid the copy, but must also key/guard the cached launch plan for stride changes.
That is a separate measured optimization. For noncontiguous BF16 V, the added
temporary payload is `2 * B * HK * DV` bytes and its copy logically reads/writes
that payload once each; neither is a measured HBM or peak-memory result. Graph
capture must retain the copied storage and replay the copy with fresh input.
No speedup claim. Incorrect baseline output cannot be a valid performance winner.
Source references: PyTorch `torch.split` returns views and `Tensor.contiguous`
returns self when already contiguous:
https://docs.pytorch.org/docs/2.15/generated/torch.split.html
https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.contiguous.html

## Executed validation

- Host test extracts the actual wrapper and write kernel AST; a NumPy pointer
  facade models physical loads/stores, while an independent indexed reference
  checks the entire arena including untouched cells/PAD. It does not run Triton.
- Baseline: 2 failing tests (GLM split V, feature-strided K/V); packed control passes.
  Candidate: all 3 tests / 81 shape-layout cases pass normally and with `python -O`.
  Cases: B1/3/8, HK1/2/8, unequal K/V widths, non-power-of-two D24, invalid slots.
  Packed input identity and original return identity checked.
- Syntax compilation passes. Native CUDA test skipped: this host has no Torch,
  Triton or GPU. This skip is not native acceptance.

## Reproduce and native acceptance

From repository root, in the existing compatible environment:

```bash
python tests/ops/test_paged_write_layout_host.py
python -O tests/ops/test_paged_write_layout_host.py
git show 40756b6c925ad0caa5741ccbc305299fb8189567:sparse_attn/ops/paged_region_kernels.py > /tmp/paged-write-before.py
NFA_WRITE_SOURCE=/tmp/paged-write-before.py python tests/ops/test_paged_write_layout_host.py
python tests/ops/test_paged_write_layout_cuda.py
compute-sanitizer --tool memcheck python tests/ops/test_paged_write_layout_cuda.py
```

Before-source command must fail 2 tests; after-source must pass all81. Native test
requires zero-tolerance arena equality for packed/split/feature views, FP16/BF16,
B1/3/32, HK2/8, three changed-input/changed-slot steps, eager and captured replay.
Also run the repository's GLM model quality/latency protocol before promotion;
compare corrected native layout reference, selected IDs, outputs, cache pages,
EOS/reused requests/recapture, whole decode latency and allocated/reserved memory.
No historical gate quality or prior GLM numbers validate this changed write path.

## Overlap and continuation

All72 remote heads checked;71 development-head dates/ancestry/unique commits are
unchanged from existing branch coverage. Fresh source inspection of this writer
at all71 heads found16 with the function and none normalizing K/V or passing
strides. Other heads lack this function/file and retain earlier ABI classifications.
PR10 head `2449f56983d7e97c5fe1c4df300c2b50b0c16b81` lacks this current writer.
No existing PR repairs this issue. No branch transplanted or overwritten.
Priority l2-fine-topk is historical; ako-pool-score/e175 remain older-ABI candidates.
New contribution is independent of open PR31–38; preserve those changes on merge.
