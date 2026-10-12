# -*- coding: utf-8 -*-
"""iter8 原型验证：compute_score coarse 段 Triton GQA K 复用。

目标段（锚点 tli_indexer.py compute_score L733-757）：
  b_k_min/b_k_max = repeat(k_min/k_max)[T,H,d']→(h g) 物化 + clamp 双 gemv +
  score_coarse_avg 的 repeat(k_avg) + gemv——胶水主导（iter7 附带发现：
  60 launches/call 中 gemv 本体仅 32μs，repeat 物化/copy/胶水 ~120μs）。

Triton 化：kernel 每 program 载 k_min/k_max 块 [BT,d']，循环 g 个 q 头做
clamp(max=0)·k_min + clamp(min=0)·k_max 双点积（GQA 组内 K 复用，免 repeat
物化）；avg 段复用 _score_fine_kernel 同构（einsum(b_q, k_avg)）。

对拍口径：与原版 torch 表达式 bf16 输出 ULP 统计（同 iter6 fine 口径）。
"""
import sys
sys.path.insert(0, '/tmp/e127_kernel_wt')
import os
os.environ['TLI_E127_FAST'] = '1'
import time
import torch
import triton
import triton.language as tl
from einops import repeat, einsum

torch.manual_seed(20261021)
dev = 'cuda:0'


@triton.jit
def _score_coarse_kernel(
    KMIN, KMAX, Q, OUT,   # KMIN/KMAX [T,Hkv,d'] bf16；Q [H,d'] fp32；OUT [H,T]
    T, stride_kt, stride_kh, stride_qh, stride_oh, stride_ot,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
):
    """coarse minmax 上界分：s[h,t] = Σ_d min(q,0)·k_min + max(q,0)·k_max。
    GQA 组内 k_min/k_max 块寄存器复用（读带宽 /g，免 repeat 物化）。"""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask_t = offs_t < T
    kmin = tl.load(KMIN + offs_t[:, None] * stride_kt + pid_h * stride_kh + offs_d[None, :],
                   mask=mask_t[:, None], other=0.0).to(tl.float32)   # [BT, D]
    kmax = tl.load(KMAX + offs_t[:, None] * stride_kt + pid_h * stride_kh + offs_d[None, :],
                   mask=mask_t[:, None], other=0.0).to(tl.float32)
    for j in tl.static_range(G):
        h = pid_h * G + j
        q = tl.load(Q + h * stride_qh + offs_d)          # [D] fp32（已缩放）
        qpos = tl.minimum(q, 0.0)                        # clamp(max=0)：负部 → 配 k_min
        qneg = tl.maximum(q, 0.0)                         # clamp(min=0)：正部 → 配 k_max
        s = tl.sum(kmin * qpos[None, :], axis=1) + tl.sum(kmax * qneg[None, :], axis=1)
        tl.store(OUT + h * stride_oh + offs_t * stride_ot,
                 s.to(OUT.dtype.element_ty), mask=mask_t)


def fast_coarse(k_min, k_max, b_q, out_dtype, BT=64):
    """k_min/k_max [T,Hkv,d']；b_q [H,d'] fp32 → out [H,T] out_dtype。"""
    T, Hkv, D = k_min.shape
    H = b_q.shape[0]
    g = H // Hkv
    out = torch.empty((H, T), device=k_min.device, dtype=out_dtype)
    _score_coarse_kernel[(triton.cdiv(T, BT), Hkv)](
        k_min, k_max, b_q, out, T,
        k_min.stride(0), k_min.stride(1), b_q.stride(0),
        out.stride(0), out.stride(1),
        Hkv=Hkv, G=g, D=D, BT=BT, num_warps=4)
    return out


def fast_avg(k_avg, b_q, out_dtype, BT=64):
    """avg 段 = einsum(b_q, repeat(k_avg))，与 _score_fine_kernel 同构。"""
    T, Hkv, D = k_avg.shape
    H = b_q.shape[0]
    g = H // Hkv
    out = torch.empty((H, T), device=k_avg.device, dtype=out_dtype)
    _score_fine_kernel[(triton.cdiv(T, BT), Hkv)](
        k_avg, b_q, out, T,
        k_avg.stride(0), k_avg.stride(1), b_q.stride(0),
        out.stride(0), out.stride(1),
        Hkv=Hkv, G=g, D=D, BT=BT, num_warps=4)
    return out


# 原版表达式（逐位复刻锚点口径）
def torch_coarse(k_min, k_max, b_q, out_dtype):
    g = b_q.shape[0] // k_min.shape[1]
    b_k_min = repeat(k_min, "t h d -> t (h g) d", g=g).to(torch.float32)
    b_k_max = repeat(k_max, "t h d -> t (h g) d", g=g).to(torch.float32)
    b_score = (
        einsum(b_q.clamp(max=0), b_k_min, "h d, kt h d -> h kt")
        + einsum(b_q.clamp(min=0), b_k_max, "h d, kt h d -> h kt")
    )
    return b_score.to(out_dtype)


def torch_avg(k_avg, b_q, out_dtype):
    g = b_q.shape[0] // k_avg.shape[1]
    b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=g).to(torch.float32)
    return einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(out_dtype)


# 复用已集成的 _score_fine_kernel（iter6，验证同构性）
from sparse_attn.indexer.tli_indexer import _score_fine_kernel  # inplace 合并后 kernel 在基类

H, Hkv, G = 32, 8, 4
print(f"{'config':<28}{'ULP比例':>10}{'maxrel':>10}{'maxabs':>10}{'tri ms':>9}{'torch ms':>10}")
for D in [32, 128]:            # d'：tail 子空间 32 / full 128
    for T in [1024, 16384, 65536 - 13]:
        for use_avg in [False, True]:
            k_min = torch.randn(T, Hkv, D, device=dev, dtype=torch.bfloat16)
            k_max = k_min + torch.rand(T, Hkv, D, device=dev, dtype=torch.bfloat16)
            b_q = torch.randn(H, D, device=dev, dtype=torch.float32)
            k_avg = torch.randn(T, Hkv, D, device=dev, dtype=torch.bfloat16) if use_avg else None

            o_t = fast_coarse(k_min, k_max, b_q, torch.bfloat16)
            o_g = torch_coarse(k_min, k_max, b_q, torch.bfloat16)
            diff = (o_t.float() != o_g.float())
            n_ulp = int(diff.sum().item())
            rel = ((o_t.float() - o_g.float()).abs() /
                   o_g.float().abs().clamp_min(1e-6)).max().item()
            mabs = (o_t.float() - o_g.float()).abs().max().item()

            rows = [f"coarse D={D:<3}T={T:<6}"]
            if use_avg:
                oa_t = fast_avg(k_avg, b_q, torch.bfloat16)
                oa_g = torch_avg(k_avg, b_q, torch.bfloat16)
                da = (oa_t.float() != oa_g.float())
                n_ulp = max(n_ulp, int(da.sum().item()))
                rel = max(rel, ((oa_t.float() - oa_g.float()).abs() /
                                oa_g.float().abs().clamp_min(1e-6)).max().item())
                mabs = max(mabs, (oa_t.float() - oa_g.float()).abs().max().item())
                rows.append("avg")

            # 性能（coarse + avg 一起计）
            torch.cuda.synchronize()
            for _ in range(3):
                fast_coarse(k_min, k_max, b_q, torch.bfloat16)
                if use_avg:
                    fast_avg(k_avg, b_q, torch.bfloat16)
            ts = []
            for _ in range(20):
                t0 = time.perf_counter()
                fast_coarse(k_min, k_max, b_q, torch.bfloat16)
                if use_avg:
                    fast_avg(k_avg, b_q, torch.bfloat16)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            t_tri = ts[10] * 1e3
            for _ in range(3):
                torch_coarse(k_min, k_max, b_q, torch.bfloat16)
                if use_avg:
                    torch_avg(k_avg, b_q, torch.bfloat16)
            ts = []
            for _ in range(20):
                t0 = time.perf_counter()
                torch_coarse(k_min, k_max, b_q, torch.bfloat16)
                if use_avg:
                    torch_avg(k_avg, b_q, torch.bfloat16)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            t_tor = ts[10] * 1e3
            print(f"{'+'.join(rows):<28}{n_ulp / o_t.numel():>10.2e}"
                  f"{rel:>10.2e}{mabs:>10.2e}{t_tri:>9.3f}{t_tor:>10.3f}")

# allclose 任务容差核验
print("\nallclose(1e-2, 1e-2) 全配置：")
for D in [32, 128]:
    for T in [16384, 65536]:
        k_min = torch.randn(T, Hkv, D, device=dev, dtype=torch.bfloat16)
        k_max = k_min + torch.rand(T, Hkv, D, device=dev, dtype=torch.bfloat16)
        b_q = torch.randn(H, D, device=dev, dtype=torch.float32)
        ok = torch.allclose(fast_coarse(k_min, k_max, b_q, torch.bfloat16).float(),
                            torch_coarse(k_min, k_max, b_q, torch.bfloat16).float(),
                            atol=1e-2, rtol=1e-2)
        print(f"  coarse D={D} T={T}: allclose={ok}")
