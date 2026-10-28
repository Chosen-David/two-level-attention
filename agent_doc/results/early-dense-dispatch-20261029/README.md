# Early dense dispatch — 2026-10-29
Main baseline: 0fd405150557e0688ab967cf569921b0cc96cdbc.
PR10 parent: 78ba04164b1db7888fc1fd0ff8fe56114b812b8c.
eager_decoding.py blob identical in both: 05919c35978fb8a5be362371ebb669c892e864f5.

Actual route: _eager_decoding_attn_gather validates shapes and full mask extent, then tries Triton only at T>=16384. Previously the Torch fallback expanded the block mask with einops.repeat, reshaped it and (per-query-head GQA) constructed arange(R)//G before deciding T<16384 and returning dense. Dense independently constructs its own mask. Move this existing decision immediately after the full_len check. Threshold, validator, dense arguments, scale/heterogeneous V/valid_length, and all long-sequence code stay unchanged. This removes source-level dead preparation; repeat at block_size=1 may be a view, so no universal allocation/kernel saving is claimed. At block_size>1 mask expansion can materialize; native profiling must establish actual lowering.

No new kernel is warranted: reuse the existing dense implementation. PR10 has the same file and does not cover this move. Other main work is not overwritten. The overall PR still has previously known main integration conflicts (auto modes, direct prefill and SWA=0 must be retained); this commit neither resolves those nor imports unrelated branches.

Validation: probe.py executes the actual baseline/candidate function dispatch prefixes with tensor metadata doubles. 192 pairs: 96 short dense routes keep identical dense call metadata/kwargs and remove repeat/head-map preparation; 96 long routes reach unchanged gather preparation. Covers lengths4096/8192/16384/32768, block1/4/64, Hkv1/8,G1/4, shared/per-query masks, valid_length None/shorter, same/heterogeneous V dimension. Normal/-O outputs identical. Cutoff8192 negative control fails. Unrelated module AST unchanged; candidate compiles. This is not Torch, tensor-output, kernel, memory or GPU timing evidence. No Torch/Triton/CUDA available; native required gates not run.

Remote reproduction:
```bash
git show 0fd405150557e0688ab967cf569921b0cc96cdbc:sparse_attn/ops/eager_decoding.py > /tmp/tli-eager-baseline-0fd4051.py
python agent_doc/results/early-dense-dispatch-20261029/probe.py --baseline /tmp/tli-eager-baseline-0fd4051.py --candidate sparse_attn/ops/eager_decoding.py
python -O agent_doc/results/early-dense-dispatch-20261029/probe.py --baseline /tmp/tli-eager-baseline-0fd4051.py --candidate sparse_attn/ops/eager_decoding.py
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
```
Native acceptance: compare baseline/candidate outputs exactly for dense dispatch on Torch CPU/CUDA, finite/NaN inputs and existing error cases; verify threshold16384 and full_len mismatch behavior, per-query GQA and MLA V widths. Compare profiler allocation/operator counts for block1/4/64; time paired eager and any supported Graph route without changing backend. Test B1 plus existing batched fallback B2/8/32 with ragged lengths/EOS. No e2e speedup or physical-memory saving claimed.

Remote inventory: all25 branch heads,13 open PRs/reviews/comments refreshed, unchanged. Reuse exact-head branch ancestry/unique commits/experiment coverage from previous run, not branch names. Main remains substantively analyzed despite unchanged SHA. No direct main push/merge or unrelated repository/guide edits.
