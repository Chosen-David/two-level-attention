"""Qwen3-8B B1 decode 的 Q/K RoPE 单 launch 实现。

数值口径严格对应 transformers 4.56.2 ``apply_rotary_pos_emb``：
BF16 输入使用 FP32 opmath，两次乘法各自写回 BF16，最终加法再写回
BF16。这里不做代数合并，也不把两次乘法和加法融合成一次 FP32 FMA。
"""

import torch
import triton
import triton.language as tl
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


@triton.jit
def _rope_head(
    X,
    Y,
    COS,
    SIN,
    head,
    x_stride_h,
    x_stride_d,
    cos_stride_d,
    sin_stride_d,
    D: tl.constexpr,
):
    """处理一个 [1, 1, D] head；X 可为 transpose 后的非连续视图。"""
    d = tl.arange(0, D)
    half: tl.constexpr = D // 2
    pair_d = tl.where(d < half, d + half, d - half)

    x = tl.load(X + head * x_stride_h + d * x_stride_d)
    x_pair = tl.load(X + head * x_stride_h + pair_d * x_stride_d)
    cos = tl.load(COS + d * cos_stride_d).to(tl.float32)
    sin = tl.load(SIN + d * sin_stride_d).to(tl.float32)

    # HF: q * cos 与 rotate_half(q) * sin 均先以 FP32 计算，再分别
    # 物化为 BF16。显式转回 FP32 后才可执行最后的 BF16 add。
    x_cos = (x.to(tl.float32) * cos).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(d < half, -x_pair, x_pair)
    rotated_sin = (
        rotated.to(tl.float32) * sin
    ).to(tl.bfloat16).to(tl.float32)
    out = (x_cos + rotated_sin).to(tl.bfloat16)

    # Y 是连续 [1, H, 1, D]，B1/S1 因而只需 head 与 d 的地址。
    tl.store(Y + head * D + d, out)


@triton.jit
def _qwen3_qk_rope_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_OUT,
    K_OUT,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    cos_stride_d,
    sin_stride_d,
    HQ: tl.constexpr,
    D: tl.constexpr,
):
    """Q/K 同一 launch：pid [0, HQ) 处理 Q，其余处理 K。"""
    pid = tl.program_id(0)
    if pid < HQ:
        _rope_head(
            Q, Q_OUT, COS, SIN, pid,
            q_stride_h, q_stride_d, cos_stride_d, sin_stride_d, D=D,
        )
    else:
        _rope_head(
            K, K_OUT, COS, SIN, pid - HQ,
            k_stride_h, k_stride_d, cos_stride_d, sin_stride_d, D=D,
        )


def _is_supported(q, k, cos, sin):
    """仅做 host 元数据检查，不读取设备值、不引入同步。"""
    return (
        q.shape == (1, 32, 1, 128)
        and k.shape == (1, 8, 1, 128)
        and cos.shape == (1, 1, 128)
        and sin.shape == (1, 1, 128)
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and q.is_cuda
        and k.device == q.device
        and cos.device == q.device
        and sin.device == q.device
        and q.stride(3) == 1
        and k.stride(3) == 1
        and cos.stride(2) == 1
        and sin.stride(2) == 1
    )


def _persistent_outputs(owner, q, k):
    """返回由 attention 模块持久持有的连续输出，保证 Graph 指针生命周期。"""
    key = (q.shape, k.shape, q.dtype, k.dtype, q.device, k.device)
    saved = getattr(owner, '_iter89_qk_rope_buffers', None)
    if saved is not None and saved[0] == key:
        return saved[1], saved[2]

    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    # 不用仅限 forward 的局部临时引用：capture 后 owner 仍持有同一地址。
    # 每层 owner 不同，层间不会互踩；同层 replay 串行复用固定输出地址。
    owner._iter89_qk_rope_buffers = (key, q_out, k_out)
    return q_out, k_out


def qwen3_decode_rope(q, k, cos, sin, *, owner):
    """Qwen3-8B B1 decode RoPE；非目标几何原样回退 HF 实现。

    ``owner`` 必须为生命周期覆盖 CUDA Graph 的 attention 模块（或测试中的
    等效持久对象）。生产分支仅在 S=B=1 时调用；这里再次检查完整形状，
    防止其他 Qwen3 配置误入专用 kernel。
    """
    if not _is_supported(q, k, cos, sin):
        return apply_rotary_pos_emb(q, k, cos, sin)

    q_out, k_out = _persistent_outputs(owner, q, k)
    # 禁用 FMA 收缩：否则 q*cos 会被并入最终加法，第一次 BF16
    # 中间舍入会被编译器消除（已实测产生数百个单 ULP 差异）。
    _qwen3_qk_rope_kernel[(40,)](
        q, k, cos, sin, q_out, k_out,
        q.stride(1), q.stride(3),
        k.stride(1), k.stride(3),
        cos.stride(2), sin.stride(2),
        HQ=32, D=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, k_out
