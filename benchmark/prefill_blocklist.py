"""Direct GPU wrapper benchmark; includes packing, excludes selector/model TTFT.

python benchmark/prefill_blocklist.py --length 65536 --rows 2048 --iters 21
Run test_prefill_blocklist first. An unavailable GPU or a kernel error is fatal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import triton
    from sparse_attn.ops.fused_attn_triton import fused_group_attn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=21)
    parser.add_argument("--selected-blocks", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: no CPU timing substitute")
    if not 1 <= args.rows <= args.length or args.iters < 3 or args.selected_blocks < 0:
        parser.error("require 1 <= rows <= length, iters >= 3, selected-blocks >= 0")
    seed, G, hkv, D, grp = 20261015, 4, 8, 128, 64
    torch.manual_seed(seed)
    pos = args.length - args.rows
    q = torch.randn(1, args.rows, hkv * G, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, args.length, hkv, D, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    sels = []
    # Deterministic block sparse workload; not a claim about real selector output.
    for start in range(0, args.rows, grp):
        end = pos + min(start + grp, args.rows)
        bits = torch.zeros(hkv, (end + 63) // 64, device="cuda", dtype=torch.bool)
        for h in range(hkv):
            chosen = torch.randperm(bits.shape[-1], device="cuda")[:args.selected_blocks]
            bits[h, chosen] = True
        sels.append(bits.repeat_interleave(64, dim=-1)[:, :end].contiguous())
    digest = hashlib.sha256()
    for value in (q, k, v, *sels):
        digest.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    kwargs = dict(pos_lo=pos, G=G, softmax_scale=D ** -.5,
                  sink_tok=128, swa_tok=128, grp=grp)

    def run(candidate):
        return fused_group_attn(q, k, v, sels, **kwargs, use_block_list=candidate)

    with torch.inference_mode():
        # JIT and cold launches excluded, explicitly reported separately.
        cold = {}
        for candidate in (False, True):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run(candidate)
            torch.cuda.synchronize()
            cold[str(candidate)] = (time.perf_counter() - t0) * 1000
            del out
        expected, actual = run(False), run(True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del expected, actual
        for _ in range(5):
            run(False)
            run(True)
        torch.cuda.synchronize()
        records = {False: [], True: []}
        for repeat in range(args.iters):
            for candidate in ((False, True) if repeat % 2 == 0 else (True, False)):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                memory_start = torch.cuda.memory_allocated()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                begin.record()
                out = run(candidate)
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000
                records[candidate].append(dict(device_ms=begin.elapsed_time(end), wall_ms=wall_ms,
                                               peak_extra_bytes=torch.cuda.max_memory_allocated() - memory_start))
                del out
        # Separate profiler run so profiling overhead never enters timing.
        kernel_counts = {}
        for candidate in (False, True):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                out = run(candidate)
                torch.cuda.synchronize()
            del out
            names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            if candidate and not any("pack_group_blocks" in n for n in names):
                raise AssertionError("candidate pack kernel absent from profile")
            kernel_counts[str(candidate)] = dict(count=len(names), names=sorted(set(names)))
    source = Path(__file__).resolve().parents[1] / "sparse_attn/ops/fused_attn_triton.py"
    def version_command(command):
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL,
                                           timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
    result = dict(scope="synthetic direct wrapper including packing; excludes selector and model",
                  args=vars(args), seed=seed, input_sha256=digest.hexdigest(),
                  kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                  git_sha=version_command(["git", "-C", str(source.parents[2]), "rev-parse", "HEAD"]),
                  driver=version_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                  device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  dtype=str(q.dtype), q_shape=list(q.shape), q_stride=list(q.stride()),
                  config=dict(G=G, Hkv=hkv, D=D, grp=grp, sink=128, swa=128, BRG=64, BK=64, warps=4, stages=2),
                  cold_launch_ms=cold, equality="exact", kernel_profiles=kernel_counts,
                  samples={str(c): records[c] for c in records},
                  medians={str(c): {key: statistics.median(r[key] for r in records[c])
                                   for key in records[c][0]} for c in records})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
