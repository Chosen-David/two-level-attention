#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-8B decode MLP SiLU+multiply 单 kernel：逐位对拍与微基准。

主对拍与性能口径不包含 gate/up/down GEMM，参考严格是 HF Qwen3MLP 的
``self.act_fn(gate) * self.up_proj(x)`` 中已经完成 up 投影后的 elementwise 段：

    reference = torch.nn.SiLU()(gate_bf16) * up_bf16

覆盖：
  1. Qwen3-8B intermediate_size=12288，B1/B4 seq_len=1 真实 shape；
  2. 全部 65536 个 BF16 gate 位模式，以及 NaN/Inf/正负零/次正规数等
     special gate × special up 交叉组合；按 int16 位模式比较，不用 allclose；
  3. 实例级 patch、prefill 回退和 unregister；
  4. B1/B4 CUDA Graph capture/replay，更新静态输入后验证逐位一致；
  5. HF 两 kernel vs Triton 单 kernel profiler 计数；
  6. eager wrapper CUDA event 时间，以及多节点 Graph replay 的 device time。

用法：
    python benchmark/efficiency/benchmark_qwen3_decode_mlp_silu.py
    python benchmark/efficiency/benchmark_qwen3_decode_mlp_silu.py --skip-perf
    python benchmark/efficiency/benchmark_qwen3_decode_mlp_silu.py --device cuda:1
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_attn.ops.qwen3_mlp_silu_triton import (  # noqa: E402
    fused_silu_mul_decode,
    is_fusable,
)
from sparse_attn.patches.qwen3_mlp_silu_patch import (  # noqa: E402
    register_mlp_silu_patch,
    unregister_mlp_silu_patch,
)

QWEN3_8B_INTERMEDIATE = 12288
_HF_ACT = nn.SiLU(inplace=False)


def hf_reference_elementwise(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """transformers 4.56.2 Qwen3MLP 的两个原始 elementwise 算子。"""
    return _HF_ACT(gate) * up


def _bits_to_bf16(bits, device) -> torch.Tensor:
    """从原始 16-bit 模式构造 BF16，保留负零及不同 NaN payload。"""
    if not isinstance(bits, torch.Tensor):
        bits = torch.tensor(list(bits), dtype=torch.int32)
    bits = bits.to(dtype=torch.int32, device="cpu")
    return bits.to(torch.int16).view(torch.bfloat16).to(device)


def _bit_equal(got: torch.Tensor, ref: torch.Tensor) -> bool:
    """NaN 也按位比较；torch.equal/allclose 不足以表达此门禁。"""
    return bool(torch.equal(got.view(torch.int16), ref.view(torch.int16)))


def _report_bits(label: str, got: torch.Tensor, ref: torch.Tensor) -> bool:
    ok = _bit_equal(got, ref)
    print(f"  [bit-exact {'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        mismatch = got.view(torch.int16) != ref.view(torch.int16)
        indices = mismatch.nonzero()[:8]
        print(f"      mismatch_count={mismatch.sum().item()}")
        for index in indices:
            idx = tuple(index.tolist())
            print(
                f"      at {idx}: got=0x{got.view(torch.int16)[idx].item() & 0xffff:04x} "
                f"ref=0x{ref.view(torch.int16)[idx].item() & 0xffff:04x}"
            )
    return ok


def _make_random(shape, device, seed, scale=4.0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gate = (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).to(device)
    up = (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).to(device)
    return gate, up


def run_bitexact(device) -> list:
    results = []
    for batch in (1, 4):
        shape = (batch, 1, QWEN3_8B_INTERMEDIATE)
        for seed, scale in ((95, 1.0), (96, 16.0)):
            gate, up = _make_random(shape, device, seed, scale)
            got = fused_silu_mul_decode(gate, up)
            ref = hf_reference_elementwise(gate, up)
            results.append(_report_bits(
                f"Qwen3-8B B{batch} random scale={scale}", got, ref))

    special_bits = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f,
        0x0080, 0x8080, 0x3f80, 0xbf80, 0x42b0, 0xc2b0,
        0x42c8, 0xc2c8, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
        0x7fc0, 0xffc0, 0x7fc1, 0xffc1,
    ]
    gate = _bits_to_bf16(special_bits, device).repeat_interleave(len(special_bits))
    up = _bits_to_bf16(special_bits, device).repeat(len(special_bits))
    gate = gate.reshape(1, 1, -1)
    up = up.reshape(1, 1, -1)
    got = fused_silu_mul_decode(gate, up)
    ref = hf_reference_elementwise(gate, up)
    results.append(_report_bits("special gate × special up（含 NaN/Inf/±0）", got, ref))

    all_bits = torch.arange(65536, dtype=torch.int32)
    gate = _bits_to_bf16(all_bits, device).reshape(1, 1, -1)
    up_pattern = _bits_to_bf16(special_bits, device)
    up = up_pattern.repeat((65536 + len(special_bits) - 1) // len(special_bits))[:65536]
    up = up.reshape(1, 1, -1)
    got = fused_silu_mul_decode(gate, up)
    ref = hf_reference_elementwise(gate, up)
    results.append(_report_bits("穷举 65536 个 BF16 gate 位模式", got, ref))
    return results


def run_gate_and_patch(device) -> list:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

    results = []
    gate, up = _make_random((1, 1, QWEN3_8B_INTERMEDIATE), device, 101)
    results.append(("decode contiguous BF16 可融合", is_fusable(gate, up)))
    results.append(("prefill 几何回退", not is_fusable(
        torch.empty(1, 2, 12288, dtype=torch.bfloat16, device=device),
        torch.empty(1, 2, 12288, dtype=torch.bfloat16, device=device),
    )))
    results.append(("FP32 回退", not is_fusable(gate.float(), up.float())))
    noncontiguous = torch.empty(
        1, 1, 24576, dtype=torch.bfloat16, device=device)[:, :, ::2]
    results.append(("非连续输入回退", not is_fusable(
        noncontiguous, torch.empty_like(noncontiguous))))

    config = Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        hidden_act="silu",
    )
    module = Qwen3MLP(config).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 1, 64, device=device).to(torch.bfloat16)
    xp = torch.randn(1, 2, 64, device=device).to(torch.bfloat16)
    with torch.no_grad():
        ref = module(x)
        ref_prefill = module(xp)
    n_patched = register_mlp_silu_patch(module, expected=1, verbose=False)
    with torch.no_grad():
        got = module(x)
        got_prefill = module(xp)
    results.append((f"真实 Qwen3MLP 注册计数={n_patched}", n_patched == 1))
    results.append(("patched decode 完整 MLP 逐位一致", _bit_equal(got, ref)))
    results.append(("patched prefill 回退逐位一致", _bit_equal(got_prefill, ref_prefill)))
    n_restored = unregister_mlp_silu_patch(module)
    results.append(("unregister 恢复原 forward", n_restored == 1))

    for label, ok in results:
        print(f"  [gate/patch {'PASS' if ok else 'FAIL'}] {label}")
    return [ok for _, ok in results]


def run_graph(device) -> list:
    results = []
    for batch in (1, 4):
        shape = (batch, 1, QWEN3_8B_INTERMEDIATE)
        static_gate, static_up = _make_random(shape, device, 110 + batch)
        for _ in range(3):
            fused_silu_mul_decode(static_gate, static_up)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fused_silu_mul_decode(static_gate, static_up)

        new_gate, new_up = _make_random(shape, device, 120 + batch, scale=32.0)
        static_gate.copy_(new_gate)
        static_up.copy_(new_up)
        graph.replay()
        torch.cuda.synchronize()
        ref = hf_reference_elementwise(new_gate, new_up)
        ok = _report_bits(f"Qwen3-8B B{batch} Graph capture/replay", static_out, ref)
        results.append(ok)
    return results


def _profile_cuda_kernel_count(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    try:
        cuda = torch.autograd.DeviceType.CUDA
        return sum(
            1 for event in prof.events()
            if getattr(event, "device_type", None) == cuda
        )
    except Exception:
        return -1


def run_kernel_count(device):
    gate, up = _make_random((1, 1, QWEN3_8B_INTERMEDIATE), device, 131)
    n_hf = _profile_cuda_kernel_count(lambda: hf_reference_elementwise(gate, up))
    n_fused = _profile_cuda_kernel_count(lambda: fused_silu_mul_decode(gate, up))
    print(f"  HF elementwise CUDA kernels={n_hf}（目标 2）")
    print(f"  fused elementwise CUDA kernels={n_fused}（目标 1）")
    return n_hf, n_fused


def _bench_eager(fn, iters: int, reps: int = 7, warmup: int = 25) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def _bench_graph(fn, inner: int, replays: int, reps: int = 7) -> float:
    """每张图捕获 inner 次调用，摊薄 graph.replay() 的 host 发射成本。"""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_outputs = [fn() for _ in range(inner)]
    assert len(captured_outputs) == inner

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / (inner * replays))
    return statistics.median(samples)


def run_perf(device, iters: int, graph_inner: int):
    print(f"  {'case':<20}{'HF eager ms':>14}{'fused eager ms':>16}"
          f"{'HF graph ms':>14}{'fused graph ms':>16}")
    for batch in (1, 4):
        shape = (batch, 1, QWEN3_8B_INTERMEDIATE)
        gate, up = _make_random(shape, device, 140 + batch)
        hf = lambda: hf_reference_elementwise(gate, up)
        fused = lambda: fused_silu_mul_decode(gate, up)
        t_hf_eager = _bench_eager(hf, iters)
        t_fused_eager = _bench_eager(fused, iters)
        replays = max(1, iters // graph_inner)
        t_hf_graph = _bench_graph(hf, graph_inner, replays)
        t_fused_graph = _bench_graph(fused, graph_inner, replays)
        print(f"  {f'Qwen3-8B B{batch}':<20}{t_hf_eager:>14.6f}"
              f"{t_fused_eager:>16.6f}{t_hf_graph:>14.6f}{t_fused_graph:>16.6f}")
        print(f"    speedup: eager={t_hf_eager / t_fused_eager:.3f}x, "
              f"graph={t_hf_graph / t_fused_graph:.3f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--graph-inner", type=int, default=32)
    parser.add_argument("--skip-perf", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用：本脚本须在主 AI 锁卡后运行")
        return 2
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    import transformers
    import triton
    print(f"device={device}, torch={torch.__version__}, "
          f"triton={triton.__version__}, transformers={transformers.__version__}")
    if transformers.__version__ != "4.56.2":
        print(f"警告：数值契约按 transformers 4.56.2 核对，当前为 {transformers.__version__}")

    print("\n[1/5] HF 完整 MLP elementwise 段逐位对拍")
    bit_results = run_bitexact(device)
    print("\n[2/5] gate / 实例 patch / prefill 回退")
    patch_results = run_gate_and_patch(device)
    print("\n[3/5] CUDA Graph capture/replay")
    graph_results = run_graph(device)
    print("\n[4/5] CUDA kernel 计数")
    n_hf, n_fused = run_kernel_count(device)

    if not args.skip_perf:
        print("\n[5/5] eager wrapper 与 Graph device time（CUDA event 中位数）")
        run_perf(device, args.iters, args.graph_inner)

    all_ok = (
        all(bit_results) and all(patch_results) and all(graph_results)
        and n_hf == 2 and n_fused == 1
    )
    print(f"\n==== 汇总：bit-exact {sum(bit_results)}/{len(bit_results)}，"
          f"gate/patch {sum(patch_results)}/{len(patch_results)}，"
          f"graph {sum(graph_results)}/{len(graph_results)}，"
          f"kernel HF/fused={n_hf}/{n_fused}；{'PASS' if all_ok else 'FAIL'} ====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
