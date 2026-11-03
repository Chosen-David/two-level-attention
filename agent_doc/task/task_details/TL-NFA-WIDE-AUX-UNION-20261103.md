# TL-NFA-WIDE-AUX-UNION-20261103

Status: host-validated fix; native GPU correctness/performance pending.
Base: 40756b6c925ad0caa5741ccbc305299fb8189567.

## Mechanism
Current request-batched selector calls _final_union_physical_kernel in both eager and capture. W=1024; auxiliary run capacity is next_power_of_2(PROT+MAX_SEED). Cache exposes max_prefill_seed_len (default128) without restricting it to W-PROT. With sink4/window128 and seedcapacity1024, N=2180,N_PAD4096,A_PAD2048. reverse_idx=2047-offs becomes negative in upper lanes before an unmasked tl.gather. The two-W merge also cannot represent the whole auxiliary run. This is register-tensor gather misuse/support loss, NOT evidence of an observed global-memory fault. Default132+128 aux fits and is not implicated.

## Fix
Keep the existing bitonic merge unchanged when A_PAD<=W. Otherwise clamp a forward auxiliary gather, build the complete padded union and use existing tl.sort. Dedup, counts and physical mapping remain unchanged. No new dependency, state or launch; large-capacity fallback costs more comparisons, no speedup claim. W=1024 is current producer contract; arbitrary non-power-of-two W is outside this patch.

## Evidence
Standalone test executes actual kernel AST with checked host pointers/gather against independent sorted-set and physical-page oracle. Baseline raises IndexError in wide-aux cases. Fixed80 cases pass normal and -O: SWA0/128/896/1024, seedcapacity0/128/892/1024/2048, first/consumed seeds, active/padded metadata, shuffled physical pages, duplicates. Tests use long rows to isolate this mechanism from existing short-protection PR29; no claim that PR29 is solved. PR34 addresses replay input binding separately and retains this same faulty merge. PR10 older cache ABI is not a dependency. No other branch transplanted.

python tests/test_nfa_union_wide_aux_host.py
python -O tests/test_nfa_union_wide_aux_host.py

Baseline reproduction: pass an extracted main selector file as positional argument. GPU command (requires installed project Torch/Triton/CUDA):
python tests/test_nfa_union_wide_aux_host.py sparse_attn/indexer/nfa_layer_batch_selector.py --cuda
compute-sanitizer --tool memcheck python tests/test_nfa_union_wide_aux_host.py sparse_attn/indexer/nfa_layer_batch_selector.py --cuda

CUDA mode imports only the actual standalone kernel definition, checks eager and poisoned-output Graph replay against the same independent oracle. It has NOT run here (Torch/Triton unavailable). Full-model ragged batches, frozen gates, EOS/re-admission, seed residency and strict-selector quality remain separate acceptance requirements. Paired GPU timing must compare unchanged default path and wide fallback against same-support full-sort reference; do not time host facade or call it GPU acceleration.
