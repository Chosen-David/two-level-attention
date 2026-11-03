#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iter91 Qwen3RMSNorm decode 三 kernel 融合 — 真实 shape 微基准 + 逐位对拍脚本。

由用户在 GPU 上运行（本文件交付时只做 py_compile/静态检查，不宣称正确性/性能）。

用法（建议 TLI venv：/home/wangyuanshuo02/two-level-attention/.venv，
     需 torch + triton + transformers==4.56.2）：
    python bench_iter91_rmsnorm.py                 # 全部：对拍+gate+patch+graph+计数+延迟
    python bench_iter91_rmsnorm.py --skip-perf     # 跳过延迟微基准
    python bench_iter91_rmsnorm.py --device cuda:1

跑卡纪律（用户全局纪律）：起跑前 nvidia-smi used<500MiB 且
query-compute-apps 查无他方进程；采数期间旁路 30s 监控。

对拍口径：
  - 参考 = transformers 4.56.2 Qwen3RMSNorm.forward 的逐字复刻（hf_reference_forward）；
  - 逐级 bit-exact：square ↔ HF 的 fp32 pow(2) 输出、mean ↔ HF variance、
    out ↔ HF 最终输出 —— 差异能定位到具体 stage；
  - gate/prefill 回退、实例级 patch 注入（q_norm/k_norm/两层 decoder norm）、
    CUDA graph capture/replay 安全性与逐位一致性、kernel 计数（HF vs 3-kernel）、
    CUDA event 延迟（中位数）。

真实 shape（来自本地 config）：
  - Qwen3-8B      : H=4096, q_heads=32, kv_heads=8,  head_dim=128, eps=1e-6
  - Qwen3-32B     : H=5120, q_heads=64, kv_heads=8,  head_dim=128, eps=1e-6
  - Qwen3-30B-A3B : H=2048, q_heads=32, kv_heads=4,  head_dim=128, eps=1e-6
"""

import argparse
import statistics
import sys

import torch
import torch.nn as nn

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_attn.ops.qwen3_rmsnorm_triton import (  # noqa: E402
    fused_rmsnorm_decode,
    is_fusable,
)
from sparse_attn.patches.qwen3_rmsnorm_patch import (  # noqa: E402
    register_rmsnorm_patch,
    unregister_rmsnorm_patch,
)

# 真实模型 shape（名称, hidden, q_heads, kv_heads, head_dim, eps）
MODEL_SHAPES = [
    ("Qwen3-8B", 4096, 32, 8, 128, 1e-6),
    ("Qwen3-32B", 5120, 64, 8, 128, 1e-6),
    ("Qwen3-30B-A3B", 2048, 32, 4, 128, 1e-6),
]


# ---------------------------------------------------------------------------
# 参考：transformers 4.56.2 Qwen3RMSNorm.forward 逐字复刻
# （modeling_qwen3.py L59-64，不引入任何额外改动）
# ---------------------------------------------------------------------------
def hf_reference_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


def _make_inputs(shape, n, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # 真实幅度：hidden ~ N(0, 1.5^2)，weight ~ N(1, 0.5^2)（训练后 norm weight
    # 通常在 1 附近）；先 fp32 生成再转 bf16，与生产 dtype 路径一致
    x = (torch.randn(shape, generator=g) * 1.5).to(torch.bfloat16).to(dev)
    w = (torch.randn(n, generator=g) * 0.5 + 1.0).to(torch.bfloat16).to(dev)
    return x, w


# ===========================================================================
# 1. 逐位对拍（含 square / mean 逐级定位）
# ===========================================================================
def run_bitexact(dev, verbose=True):
    results = []  # (case, ok_square, ok_mean, ok_out)
    seed = 0
    for name, hidden, qh, kvh, hd, eps in MODEL_SHAPES:
        arms = [
            (f"{name} input_layernorm  (1,1,{hidden})", (1, 1, hidden), hidden),
            (f"{name} post_attn_norm   (1,1,{hidden})", (1, 1, hidden), hidden),
            (f"{name} q_norm           (1,{qh},1,{hd})", (1, qh, 1, hd), hd),
            (f"{name} k_norm           (1,{kvh},1,{hd})", (1, kvh, 1, hd), hd),
            (f"{name} batched norm     (4,1,{hidden})", (4, 1, hidden), hidden),
            (f"{name} batched q_norm   (4,{qh},1,{hd})", (4, qh, 1, hd), hd),
        ]
        for label, shape, n in arms:
            seed += 1
            x, w = _make_inputs(shape, n, dev, seed)
            out, square, mean = fused_rmsnorm_decode(x, w, eps, return_intermediates=True)

            # HF 参考及其中间量（与 hf_reference_forward 完全同链）
            h32 = x.to(torch.float32)
            ref_sq = h32.pow(2)
            ref_var = ref_sq.mean(-1, keepdim=True)
            ref_out = hf_reference_forward(x, w, eps)

            ok_sq = torch.equal(square, ref_sq)
            ok_mean = torch.equal(mean, ref_var)
            ok_out = torch.equal(out, ref_out)
            results.append((label, ok_sq, ok_mean, ok_out))
            if verbose:
                status = "PASS" if (ok_sq and ok_mean and ok_out) else "FAIL"
                print(f"  [bit-exact {status}] {label}: square={ok_sq} mean={ok_mean} out={ok_out}")
                if not ok_out and (ok_sq and ok_mean):
                    # 差异定位：out 段（rsqrt / 舍入链）——打印最大 ULP 级差
                    diff = (out.view(torch.int16).int() - ref_out.view(torch.int16).int()).abs()
                    print(f"      out 段差异：max |bf16 bit-diff| = {diff.max().item()}")
    return results


# ===========================================================================
# 2. gate / prefill 回退
# ===========================================================================
def run_gate_fallback(dev):
    results = []
    x, w = _make_inputs((2, 256, 4096), 4096, dev, seed=99)  # prefill（seq_len>1）
    results.append(("prefill shape 不融合", not is_fusable(x, w)))
    x2, w2 = _make_inputs((1, 1, 4096), 4096, dev, seed=100)
    results.append(("decode shape 融合", is_fusable(x2, w2)))
    # fp32 weight → 回退
    results.append(("fp32 weight 不融合", not is_fusable(x2, w2.float())))
    # 非连续 → 回退（注意 (1,1,N) 上对 size-1 维的切片仍算 contiguous，
    # 必须用 stride(-1)!=1 的步长切片才能真正构造非连续输入）
    xt = torch.randn(1, 1, 8192, device=dev).to(torch.bfloat16)[:, :, ::2]
    assert xt.shape[-1] == 4096 and not xt.is_contiguous()
    results.append(("非连续输入 不融合", not is_fusable(xt, w2)))
    for label, ok in results:
        print(f"  [gate {'PASS' if ok else 'FAIL'}] {label}")
    return [ok for _, ok in results]


# ===========================================================================
# 3. patch 注入（真实 Qwen3RMSNorm 实例树：q/k norm + 两层 decoder norm + final）
# ===========================================================================
class _FakeAttention(nn.Module):
    def __init__(self, hidden, head_dim, eps):
        super().__init__()
        self.q_norm = _new_rmsnorm(head_dim, eps)
        self.k_norm = _new_rmsnorm(head_dim, eps)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden, head_dim, eps):
        super().__init__()
        self.input_layernorm = _new_rmsnorm(hidden, eps)
        self.post_attention_layernorm = _new_rmsnorm(hidden, eps)
        self.self_attn = _FakeAttention(hidden, head_dim, eps)


class _FakeModel(nn.Module):
    def __init__(self, hidden, head_dim, eps, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeDecoderLayer(hidden, head_dim, eps) for _ in range(n_layers)]
        )
        self.norm = _new_rmsnorm(hidden, eps)  # 末层 final norm


def _new_rmsnorm(n, eps):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    m = Qwen3RMSNorm(n, eps=eps)
    m.weight.data.uniform_(-0.5, 1.5)  # 随机非平凡权重
    return m.to(torch.bfloat16)


def run_patch_test(dev):
    results = []
    hidden, head_dim, eps, qh, kvh = 4096, 128, 1e-6, 32, 8
    model = _FakeModel(hidden, head_dim, eps, n_layers=2).to(dev)
    expected = 2 * 4 + 1  # 2 层 ×(q+k+两层 norm) + final norm

    n_patched = register_rmsnorm_patch(model, expected=expected, verbose=False)
    results.append((f"patch 计数 == {expected}", n_patched == expected))

    # decode：4 类 norm 全部逐位对拍
    for sub, shape, n in [
        ("input_layernorm", (1, 1, hidden), hidden),
        ("post_attention_layernorm", (1, 1, hidden), hidden),
        ("q_norm", (1, qh, 1, head_dim), head_dim),
        ("k_norm", (1, kvh, 1, head_dim), head_dim),
    ]:
        x, _ = _make_inputs(shape, n, dev, seed=7)
        mod = getattr(model.layers[0], sub) if sub in (
            "input_layernorm", "post_attention_layernorm") else getattr(model.layers[0].self_attn, sub)
        got = mod(x)
        ref = hf_reference_forward(x, mod.weight, mod.variance_epsilon)
        ok = torch.equal(got, ref)
        results.append((f"patched {sub} decode 逐位一致", ok))
        print(f"  [patch {'PASS' if ok else 'FAIL'}] {sub} (decode)")

    # prefill：走原 forward（回退路径）
    xp, _ = _make_inputs((1, 512, hidden), hidden, dev, seed=8)
    got = model.layers[0].input_layernorm(xp)
    ref = hf_reference_forward(xp, model.layers[0].input_layernorm.weight, eps)
    ok = torch.equal(got, ref)
    results.append(("patched prefill 回退原 forward 逐位一致", ok))
    print(f"  [patch {'PASS' if ok else 'FAIL'}] prefill fallback")

    # 撤销后恢复原 forward
    n_restored = unregister_rmsnorm_patch(model)
    ok = n_restored == expected and torch.equal(
        model.layers[0].input_layernorm(xp),
        hf_reference_forward(xp, model.layers[0].input_layernorm.weight, eps),
    )
    results.append((f"unregister 恢复 {expected} 个模块", ok))
    print(f"  [patch {'PASS' if ok else 'FAIL'}] unregister/restore")
    return [ok for _, ok in results]


# ===========================================================================
# 4. CUDA graph capture / replay 安全 + 逐位一致
# ===========================================================================
def run_graph_test(dev):
    results = []
    x, w = _make_inputs((1, 1, 4096), 4096, dev, seed=11)
    eps = 1e-6

    # —— 4a. 裸 ops 包装：capture 融合路径 ——
    static_x = x.clone()
    for _ in range(3):  # 预热（Triton JIT 编译 + allocator 稳态）
        fused_rmsnorm_decode(static_x, w, eps)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = fused_rmsnorm_decode(static_x, w, eps)  # 返回张量必须保引用
    # 换输入重放：确认 replay 真实重算且逐位一致
    x2, _ = _make_inputs((1, 1, 4096), 4096, dev, seed=12)
    static_x.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    ref = hf_reference_forward(x2, w, eps)
    ok = torch.equal(static_out, ref)
    results.append(("graph capture/replay(ops) 逐位一致", ok))
    print(f"  [graph {'PASS' if ok else 'FAIL'}] ops 融合 capture+replay")

    # —— 4b. patched 模块 forward：capture（含 gate 元数据路径）——
    model = _FakeModel(4096, 128, 1e-6, n_layers=1).to(dev)
    register_rmsnorm_patch(model, expected=1 * 4 + 1, verbose=False)
    mod = model.layers[0].input_layernorm
    for _ in range(3):
        mod(static_x)
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        static_out2 = mod(static_x)
    x3, _ = _make_inputs((1, 1, 4096), 4096, dev, seed=13)
    static_x.copy_(x3)
    g2.replay()
    torch.cuda.synchronize()
    ref3 = hf_reference_forward(x3, mod.weight, mod.variance_epsilon)
    ok2 = torch.equal(static_out2, ref3)
    results.append(("graph capture/replay(patched module) 逐位一致", ok2))
    print(f"  [graph {'PASS' if ok2 else 'FAIL'}] patched module capture+replay")
    unregister_rmsnorm_patch(model)
    return [ok for _, ok in results]


# ===========================================================================
# 5. kernel 计数（torch.profiler）：HF eager vs 三 kernel 融合
# ===========================================================================
def _profile_cuda_kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = []
    try:
        dev_type_cuda = torch.autograd.DeviceType.CUDA
        for e in prof.events():
            if getattr(e, "device_type", None) == dev_type_cuda:
                kernels.append((e.name, getattr(e, "self_device_time_total", 0.0)))
    except Exception:
        pass
    return kernels


def run_kernel_count(dev):
    x, w = _make_inputs((1, 1, 4096), 4096, dev, seed=21)
    eps = 1e-6
    print("  --- HF eager 一次 forward 的 CUDA kernel ---")
    ks_hf = _profile_cuda_kernels(lambda: hf_reference_forward(x, w, eps))
    for name, us in ks_hf:
        print(f"    {us:8.1f} us  {name[:90]}")
    print(f"    合计 {len(ks_hf)} 个")
    print("  --- 融合路径（kernel A + torch.mean + kernel C）---")
    ks_f = _profile_cuda_kernels(lambda: fused_rmsnorm_decode(x, w, eps))
    for name, us in ks_f:
        print(f"    {us:8.1f} us  {name[:90]}")
    print(f"    合计 {len(ks_f)} 个（目标 3）")
    return len(ks_f)


# ===========================================================================
# 6. 延迟微基准（CUDA event，中位数）
# ===========================================================================
def _bench_fn(fn, iters, reps=5, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return statistics.median(times)


def run_perf(dev, iters):
    print(f"  {'case':<44}{'HF ms':>10}{'fused ms':>10}{'ratio':>8}")
    for name, hidden, qh, kvh, hd, eps in MODEL_SHAPES:
        for label, shape, n in [
            (f"{name} norm (1,1,{hidden})", (1, 1, hidden), hidden),
            (f"{name} q_norm (1,{qh},1,{hd})", (1, qh, 1, hd), hd),
            (f"{name} k_norm (1,{kvh},1,{hd})", (1, kvh, 1, hd), hd),
        ]:
            x, w = _make_inputs(shape, n, dev, seed=31)
            t_hf = _bench_fn(lambda: hf_reference_forward(x, w, eps), iters)
            t_f = _bench_fn(lambda: fused_rmsnorm_decode(x, w, eps), iters)
            print(f"  {label:<44}{t_hf:>10.4f}{t_f:>10.4f}{t_hf / t_f:>8.2f}x")


def main():
    ap = argparse.ArgumentParser(description="iter91 Qwen3RMSNorm decode 融合对拍/微基准")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用：本脚本必须在 GPU 上运行")
        return 2
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    print(f"device={dev}, torch={torch.__version__}")
    try:
        import triton
        import transformers
        print(f"triton={triton.__version__}, transformers={transformers.__version__}")
    except Exception as e:
        print(f"依赖缺失：{e}")
        return 2

    print("\n[1/6] 逐位对拍（square/mean/out 逐级定位）")
    r1 = run_bitexact(dev)
    print("\n[2/6] gate / prefill 回退")
    r2 = run_gate_fallback(dev)
    print("\n[3/6] patch 注入（真实 Qwen3RMSNorm 实例树）")
    r3 = run_patch_test(dev)
    print("\n[4/6] CUDA graph capture/replay")
    r4 = run_graph_test(dev)
    print("\n[5/6] kernel 计数（profiler）")
    n_kernels = run_kernel_count(dev)

    if not args.skip_perf:
        print("\n[6/6] 延迟微基准（CUDA event 中位数）")
        run_perf(dev, args.iters)

    all_ok = (
        all(sq and mn and ot for _, sq, mn, ot in r1)
        and all(r2) and all(r3) and all(r4)
    )
    n_fail = sum(1 for _, sq, mn, ot in r1 if not (sq and mn and ot))
    print(f"\n==== 汇总：bit-exact {len(r1) - n_fail}/{len(r1)} PASS，"
          f"gate {sum(r2)}/{len(r2)}，patch {sum(r3)}/{len(r3)}，graph {sum(r4)}/{len(r4)}，"
          f"融合 kernel 数 {n_kernels}（目标 3）====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
