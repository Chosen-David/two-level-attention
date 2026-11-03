"""Qwen3 decode MLP 的 SiLU(gate) * up 单 kernel 融合。

数值契约以 transformers 4.56.2 ``Qwen3MLP.forward`` 和 PyTorch 2.8.0
CUDA 实现为准。HF 的 elementwise 段是两个独立算子：

    silu_bf16 = self.act_fn(gate_bf16)       # 第一个 BF16 写回
    out_bf16 = silu_bf16 * up_bf16           # 第二个 BF16 写回

PyTorch ``ActivationSiluKernel.cu`` 的 BF16 dispatch 使用
``opmath_t = float``，实际表达式是 ``x_acc / (1 + exp(-x_acc))``，不是
``x * sigmoid(x)``。``c10::cuda::compat::exp(float)`` 调用 ``::expf``；
Triton 这里对应 libdevice ``__nv_expf``。除法和最终乘法使用非 FTZ PTX RN，
避免 libdevice 的 FTZ 指令吞掉次正规数；不使用近似除法或编译器收缩。

必须保留的舍入链：
    gate_fp32 -> exp/add/div (FP32 RN) -> silu_fp32 -> BF16 RTNE
              -> 无损回 FP32 -> 与 up_fp32 做 RN 乘法 -> BF16 RTNE。

仅融合 decode 几何 ``(..., 1, intermediate_size)`` 的 contiguous BF16
CUDA 输入；其它几何/dtype/布局由 patch 层回退 HF。这里不做 GEMM、
RMSNorm、residual、attention 或 selector 改动。
"""

import torch
import triton
import triton.language as tl

# 精确算术映射（Triton 3.4 NVIDIA backend）：
#   exp    -> __nv_expf       （不是 __nv_fast_expf）
#   add_rn -> __nv_fadd_rn    （FP32 round-to-nearest-even）
#   div/mul -> 非 FTZ PTX RN（libdevice 的 fdiv/fmul 会降成 .ftz 指令）
from triton.language.extra.cuda.libdevice import (
    add_rn as _fadd_rn,
    exp as _expf,
)


@triton.jit
def _div_rn_no_ftz(x, y):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=r,r,r", [x, y],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_rn_no_ftz(x, y):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [x, y],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _silu_mul_bf16_kernel(
    GATE_PTR,
    UP_PTR,
    OUT_PTR,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # int64 基址避免大 prefill/测试张量发生 32-bit 偏移溢出；生产 decode
    # 的 grid 仅由静态 numel 决定，capture 时无 host sync。
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate_bf16 = tl.load(GATE_PTR + offsets, mask=mask, other=0.0)
    up_bf16 = tl.load(UP_PTR + offsets, mask=mask, other=0.0)
    gate_fp32 = gate_bf16.to(tl.float32)
    up_fp32 = up_bf16.to(tl.float32)

    # PyTorch SiLU：x_acc / (opmath_t(1) + compat::exp(-x_acc))。
    # 保留原表达式与运算顺序，禁止重写成 x*sigmoid(x)。
    exp_neg_x = _expf(-gate_fp32)
    denominator = _fadd_rn(1.0, exp_neg_x)
    silu_fp32 = _div_rn_no_ftz(gate_fp32, denominator)

    # 独立 HF SiLU kernel 的输出是 BF16；必须先物化这次舍入，再进行下一步
    # 乘法。即使融合后不写全局中间数组，也不能省略这个类型转换。
    silu_bf16 = silu_fp32.to(tl.bfloat16)

    # c10::BFloat16::operator* 在 FP32 中相乘并 RTNE 回 BF16。
    product_fp32 = _mul_rn_no_ftz(silu_bf16.to(tl.float32), up_fp32)
    out_bf16 = product_fp32.to(tl.bfloat16)
    tl.store(OUT_PTR + offsets, out_bf16, mask=mask)


def is_fusable(gate: torch.Tensor, up: torch.Tensor) -> bool:
    """仅做元数据判断，CUDA Graph capture 期间无数据依赖分支或同步。"""
    if not isinstance(gate, torch.Tensor) or not isinstance(up, torch.Tensor):
        return False
    if gate.shape != up.shape or gate.dim() < 2 or gate.shape[-2] != 1:
        return False
    if gate.numel() == 0:
        return False
    if gate.dtype is not torch.bfloat16 or up.dtype is not torch.bfloat16:
        return False
    if not gate.is_cuda or not up.is_cuda or gate.device != up.device:
        return False
    return gate.is_contiguous() and up.is_contiguous()


def fused_silu_mul_decode(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """返回 ``BF16(SiLU(gate)) * up`` 的逐位等价 BF16 输出。

    调用前必须满足 ``is_fusable``。Qwen3-8B 的 intermediate_size=12288；
    BLOCK=256 时 B1/B4 分别启动 48/192 个 program，较 BLOCK=1024 提供更
    充足的 SM 并行度。该算子在 decode 小张量上主要省一次 launch 和一次
    BF16 中间全局读写，不涉及任何 GEMM 合并。
    """
    assert is_fusable(gate, up), (
        "fused_silu_mul_decode 仅接受相同 shape 的 (...,1,N) contiguous "
        "BF16 CUDA gate/up；其它情况必须由 patch 回退 HF"
    )
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _silu_mul_bf16_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out
