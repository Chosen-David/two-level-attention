# Pool assembly: remove redundant protected concatenations

Analyzed main 0fd405150557e0688ab967cf569921b0cc96cdbc; applied narrowly to existing integration PR10 parent 80eea3f94f854748ecd24b3e4414d4bc3ea1a1ea. Latest main changes are not overwritten; PR10 still needs integration with newer main including direct prefill and dedicated SWA=0 sentinel.
Source: FarSkipGraphWorkspace.select builds [sink,window] and validity in two scratch buffers, then copies both into [candidates,protected]. Replace the latter with [candidates,sink,window] and corresponding valid pieces. Delete intermediate cats, including when L2 is bypassed. Protected tokens STILL participate in L2 softmax. L1, L2 arithmetic/top-k, attention, ring mapping, gate, EOS and mask semantics unchanged.
Two Python torch.cat calls and two scratch allocations per workspace removed (not a measured kernel launch count). Champion hkv8: intermediate 132*8*(8+1)=9504 logical bytes; main geometric allocation 2048*(8+1)=18432 bytes, PR10 fixed-shape allocation9504 bytes. This is scratch accounting, NOT measured process/GPU memory or KV residency improvement.
576 source-AST/NumPy assembly pairs pass normal and -O: Hkv1/8, bs2/64, L1/3/128/1025/4291/32768, sink/window 0/4 and0/128, zero/two pages, zero/one/1024 middle budget; 192 pruning/384 bypass. Captured pool, valid and output positions/masks identical. Top-k output supplied as fixed ordered stand-in; no numerical scorer/attention validation. Negative control swapping sink/window in pool fails.
No Torch/Triton/CUDA in this environment. Candidate AST parses. No GPU speed, physical memory, quality or full integration claim. No numerical threshold relaxed.
Reproduce local assembly checks:
python agent_doc/results/pool-assembly-20261028/probe.py
python -O agent_doc/results/pool-assembly-20261028/probe.py
GPU acceptance (repository environment; do not count skipped CUDA tests as passes):
python -m pytest tests/test_far_skip_hybrid_cache.py tests/test_hybrid_batch_cache.py -q
Pair parent and this commit with the existing e152v4/hybrid benchmark commands and identical champion flags; eager/Graph B1/2/8/32, unequal lengths, EOS, wraps and second-request capture. Native timings remain pending.
All25 remote branch heads/main and13 PR heads unchanged before write; reviews empty/no new comments. Full branch ancestry/date/unique-commit evidence reused from preceding cache-zero audit after exact identity comparison. This change not present in PR10/PR13 inspected assembly. Existing fixed workspace capacity optimization retained. No other branch modified.

