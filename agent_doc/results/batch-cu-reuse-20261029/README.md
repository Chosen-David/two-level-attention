# Batch fallback bounds reuse — 2026-10-29
Main inspected: 0fd405150557e0688ab967cf569921b0cc96cdbc, eager_decoding.py blob 05919c35978fb8a5be362371ebb669c892e864f5.
Isolated candidate based on PR10 parent cf8eb15d34ea2e4b6e617709427543141fe04170; preserves its previous early-dense fix. PR12/13 retain the same main wrapper. All25 branch heads unchanged; exact-head ancestry/unique commits/code conclusions reused from ../index-prefix-audit-20261029/remote-current.json. All13 PR reviews and new comments checked. No duplicate branch transplant.

## Mechanism and scope
_eager_decoding_attn_batch uses fused batch kernels when eligible; otherwise calls _eager_decoding_attn_gather_batch. That fallback loops B rows, constructs identical device int32 [0,T] B times, then calls single-row gather/dense. Move construction outside the row loop, once per invocation (zero times for B=0). Consumers only read bounds; Triton helpers cast/contiguous to locals, kernels only load CUK. Nothing shares bounds across requests, devices, streams or recaptures. Row slicing, valid_length dispatch, concatenation and numerical kernel/order unchanged.
This removes B-1 tensor construction calls in this fallback only, not all indexer/attention row loops. It does not optimize the already fused batch path, ring storage or GPU arithmetic. Dense may still synchronize on bounds; [B,2] normalization's item() remains a separate candidate. Generic fallback supplies fixed [0,T], with per-row valid_length preserved; it does not newly enable ragged fused kernels.
Existing immutable request metadata reuse is sufficient; new kernel/global cache would add unnecessary lifetime complexity.

## Evidence
100 paired executions of actual wrapper AST with metadata/NumPy doubles: B0/1/2/10/32, T1/8192/16384/32768, None/int/list/tuple/tensor valid_length. Same row order/arguments/output and B0 exception; tensor constructions B -> 1 for B>0; second invocation uses fresh objects. Normal and -O match, wrong T+1 negative rejected, unrelated production AST unchanged, compile passes.
No installed Torch/Triton/CUDA: these are wrapper/control-flow tests, not native tensor, numerical, capture, latency, HBM or memory results. No measured speedup claimed. Model does not establish CUDA asynchronous lifetime; native validation remains required.

## Reproduce
```sh
git show cf8eb15d34ea2e4b6e617709427543141fe04170:sparse_attn/ops/eager_decoding.py > /tmp/cu-baseline.py
python agent_doc/results/batch-cu-reuse-20261029/probe.py /tmp/cu-baseline.py sparse_attn/ops/eager_decoding.py
python -O agent_doc/results/batch-cu-reuse-20261029/probe.py /tmp/cu-baseline.py sparse_attn/ops/eager_decoding.py
# In the existing project CUDA environment; unexecuted locally:
PYTHONPATH=. python agent_doc/results/batch-cu-reuse-20261029/native_check.py /tmp/cu-baseline.py > /tmp/cu-native.json
```
Native script compares exact outputs, alternates paired CUDA-event wrapper timing for B2/10, T256/16384, shared/per-query masks. It is an acceptance harness, NOT a completed result. Before merge additionally run project eager/Graph integration on B1/2/10/32, uneven lengths, frozen gates/EOS, repeated ring wraps and new request capture; compare to parent under identical flags. CPU test cannot certify these. No threshold relaxation.
PR10 remains 6 commits behind main with pre-existing integration conflicts; retain latest-main direct-prefill and SWA=0 fixes when resolving. This narrow descendant does not pretend to resolve unrelated integration.
