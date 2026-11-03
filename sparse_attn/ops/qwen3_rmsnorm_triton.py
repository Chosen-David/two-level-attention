"""
iter91：Qwen3 decode（seq_len==1）Qwen3RMSNorm 三 kernel 融合 — Triton ops 层。

==========================================================================
目标
==========================================================================
把 HF transformers 4.56.2 `Qwen3RMSNorm.forward` 的 eager 多 kernel 链
    to(fp32) -> pow(2) -> mean -> +eps -> rsqrt -> mul -> to(bf16) -> weight mul
（每个算子各一个 kernel，约 8 次启动）压成 **3 个 kernel**：

    [A] `_square_fp32_kernel` ：从原 BF16 x 直接生成 FP32
        square = fp32(x) * fp32(x)              ← 融合掉 to(fp32)+pow(2)
    [B] `torch.mean(square, dim=-1, keepdim=True)`
        ← 调用 PyTorch 原生归约 kernel，**归约 kernel 与求和顺序零改动**
    [C] `_norm_scale_kernel` ：读原 BF16 x / FP32 mean / BF16 weight，
        一次算完 rsqrt、归一化、BF16 舍入、weight 缩放 ← 融合掉其余全部

==========================================================================
逐位语义契约（相对 transformers 4.56.2 原实现，BF16 输入）
==========================================================================
HF 原文（modeling_qwen3.py L59-64）：
    input_dtype = hidden_states.dtype                       # bf16
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return self.weight * hidden_states.to(input_dtype)

本实现的复刻链（kernel A / torch.mean / kernel C 三段）：
    x32        = x.to(fp32)                    # bf16->fp32 无损（bf16 ⊂ fp32）
    variance   = torch.mean(x32 * x32, -1)     # 同值同 shape 同 kernel → 逐位同
    rstd       = rsqrt(variance + fp32(eps))   # 见下方「舍入复刻」第 3 条
    norm       = x32 * rstd                    # fp32 乘法，RTNE
    bf16_round = norm.to(bf16)                 # RTNE 舍入（= .to(input_dtype)）
    out        = (weight * bf16_round).to(bf16)
                 # torch 的 bf16*bf16 走 fp32 opmath（bf16 乘积在 fp32 中精确），
                 # kernel C 显式按 fp32 opmath 计算后再 RTNE 回 bf16

「舍入复刻」逐条依据：
  1. pow(x, 2)：PyTorch 对整数指数 2 有 a*a 特化（pow_tensor_scalar_kernel），
     kernel A 的 fp32(x)*fp32(x) 与之逐位一致；
  2. mean：square 与 HF pow 输出值、shape、布局、dtype 全同，
     调用同一 torch.mean kernel → 归约顺序逐位一致（不用 tl.sum 替代）；
  3. rsqrt：PyTorch CUDA rsqrt(fp32) 用 ::rsqrt（即 CUDA rsqrtf /
     libdevice __nv_rsqrtf）；Triton 侧同样调 libdevice rsqrt →
     预期逐位一致（GPU 上由 bit-exact 门禁裁决，见风险清单）；
  4. eps 加法：PyTorch fp32 tensor + python float 标量 → 标量 RTNE 转 fp32
     后在 fp32 相加；kernel C 将 eps 显式 .to(tl.float32) 后相加；
  5. fp32→bf16 舍入：torch .to(bf16) 与 Triton .to(tl.bfloat16) 均为 RTNE；
  6. weight 乘法：显式升 fp32 相乘再 RTNE 回 bf16，复刻 torch 的
     bf16 elementwise opmath（opmath_type<BFloat16> = float）。

==========================================================================
约束（与用户 iter91 指令一致）
==========================================================================
  - 仅 `seq_len == 1` 的 decode（BF16、CUDA、contiguous）走融合；
    decoder norm 几何为 (B,1,H)，Q/K norm 在 attention patch 中的真实
    pre-transpose 几何为 (B,1,num_heads,D)；prefill/其它布局回退 HF；
  - 覆盖 Q/K norm（head_dim=128，行数 = B*num_heads）与 decoder 两层 norm
    （hidden_size，行数 = B）；不含 residual / RoPE / GEMM；
  - Graph capture 安全：无 .item()/host sync/数据依赖分支；grid 由静态
    shape 决定。注意既有 CUDA graph 纪律：capture 后调用方必须持有返回的
    输出张量引用（intermediate square/mean 属 graph 私有池，随图重放复用）。

==========================================================================
已知语义风险（不在本文件内裁决，GPU 对拍脚本裁定）
==========================================================================
  R1 rsqrt 逐位性：torch ::rsqrt vs Triton libdevice rsqrt 预期同源，
     但未在 GPU 验证（bench 的 bit-exact 门禁会暴露）；
  R2 fp32→bf16 舍入方向：现代 Triton（3.x）默认 RTNE，与 torch 一致，
     极老版本曾默认截断（本环境 Triton 3.4，风险低但列出）；
  R3 torch.mean 的 kernel 选择对 shape/layout 敏感：本实现保证 square
     与 HF 的 pow 输出布局全同，理论消除该风险，仍以 GPU 对拍为准。
"""

import torch
import triton
import triton.language as tl

# libdevice rsqrt 导入：与 Liger-Kernel 相同的兼容链（triton>=3.0 常规路径
# → NGC 容器路径 → 旧版 triton.language.math 回退）
try:
    from triton.language.extra.libdevice import rsqrt as _tl_rsqrt
except ModuleNotFoundError:  # pragma: no cover - NGC 容器等特殊环境
    try:
        from triton.language.extra.cuda.libdevice import rsqrt as _tl_rsqrt
    except ModuleNotFoundError:  # pragma: no cover - 旧版 triton
        from triton.language.math import rsqrt as _tl_rsqrt


# ---------------------------------------------------------------------------
# Triton kernel A：BF16 x → FP32 square（融合 to(fp32) + pow(2) 两步）
# ---------------------------------------------------------------------------
@triton.jit
def _square_fp32_kernel(
    X_PTR,          # *bf16，二维化后输入 (rows, N)，本 program 处理第 pid 行
    SQ_PTR,         # *fp32，输出 square，shape 与原 hidden_states 全同
    x_row_stride,   # 输入行步长（contiguous 下 == N）
    sq_row_stride,  # 输出行步长
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 负责一行：decode 场景行数极少（B 或 B*num_heads），
    # launch 开销主导，行级并行 + 单块整行加载即是最优结构。
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 读原 BF16 x → 无损升 fp32 → 平方（与 HF 的 to(fp32)+pow(2) 逐位同）
    x = tl.load(X_PTR + row * x_row_stride + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    sq = x32 * x32

    tl.store(SQ_PTR + row * sq_row_stride + cols, sq, mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel C：BF16 x + FP32 mean + BF16 weight → BF16 out
# 严格复刻 x32*rsqrt(mean+eps) → .to(bf16) → (weight * bf16_round) 的舍入链
# ---------------------------------------------------------------------------
@triton.jit
def _norm_scale_kernel(
    X_PTR,           # *bf16，原输入（不是 square！kernel C 重读原 x）
    MEAN_PTR,        # *fp32，torch.mean 输出展平后的逐行均值
    W_PTR,           # *bf16，weight（一维 N，contiguous）
    OUT_PTR,         # *bf16，输出，shape 与原 hidden_states 全同
    x_row_stride,
    mean_row_stride,
    out_row_stride,
    n_cols,
    eps,             # python float（fp32 运行时标量）
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # --- 1) x32 = x.to(fp32)（无损）---
    x = tl.load(X_PTR + row * x_row_stride + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # --- 2) rstd = rsqrt(mean + fp32(eps)) ---
    # torch：fp32 tensor + python float 标量 → 标量先 RTNE 转 fp32 再相加
    mean = tl.load(MEAN_PTR + row * mean_row_stride)  # fp32 标量
    eps32 = eps.to(tl.float32)
    rstd = _tl_rsqrt(mean + eps32)

    # --- 3) norm = x32 * rstd（fp32 RTNE 乘法，与 torch mul 同） ---
    norm = x32 * rstd

    # --- 4) bf16_round = norm.to(bf16)：RTNE 舍入，复刻 .to(input_dtype) ---
    bf16_round = norm.to(tl.bfloat16)

    # --- 5) out = (weight * bf16_round).to(bf16) ---
    # torch 的 bf16 elementwise 乘法走 fp32 opmath（bf16×bf16 在 fp32 精确），
    # 再 RTNE 回 bf16；这里显式同路径：bf16_round 无损回 fp32 → w32 相乘 →
    # RTNE 回 bf16。
    r32 = bf16_round.to(tl.float32)
    w = tl.load(W_PTR + cols, mask=mask, other=0.0)
    w32 = w.to(tl.float32)
    out = (w32 * r32).to(tl.bfloat16)

    tl.store(OUT_PTR + row * out_row_stride + cols, out, mask=mask)


@triton.jit
def _rmsnorm_single_kernel(
    X_PTR, W_PTR, OUT_PTR, x_row_stride, out_row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """实验臂：单行内完成平方、归约和输出，不改变最终舍入链。"""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(X_PTR + row * x_row_stride + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    mean = tl.sum(x32 * x32, axis=0) / n_cols
    rstd = _tl_rsqrt(mean + eps.to(tl.float32))
    bf16_round = (x32 * rstd).to(tl.bfloat16)
    w = tl.load(W_PTR + cols, mask=mask, other=0.0)
    out = (w.to(tl.float32) * bf16_round.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT_PTR + row * out_row_stride + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Python 包装层
# ---------------------------------------------------------------------------
_MAX_FUSED_COLS = 8192  # BLOCK 上限保护：覆盖 8B(4096)/32B(5120→BLOCK 8192)/30B(2048)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pick_num_warps(block: int) -> int:
    # 与 Liger calculate_settings 同量级的选择：
    # 小 BLOCK（q/k norm 128）4 warp；hidden 4096/5120→8192 用 8/16 warp
    if block <= 512:
        return 4
    if block <= 4096:
        return 8
    return 16


def is_fusable(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """seq_len==1 decode 融合 gate（不满足任何一条都回退 HF 原 forward）。

    判据（全部为元数据判断，无 host sync，graph capture 安全）：
      - 输入至少 2 维且最后一维 == weight.numel()
      - decoder norm (B,1,H) 的 shape[-2]==1；attention patch 在 transpose
        前调用 Q/K norm，其真实几何是 (B,1,H,D)，故四维时以 shape[1]==1
        判定 seq_len==1。两种几何均可零拷贝 view 成 (rows,N)。
      - x / weight 均 BF16（Qwen3 生产 dtype）且 CUDA 且 contiguous
      - N <= 8192（BLOCK 寄存器预算保护）
    """
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if hidden_states.dim() < 2:
        return False
    is_decode = hidden_states.shape[-2] == 1 or (
        hidden_states.dim() == 4 and hidden_states.shape[1] == 1)
    if not is_decode:
        return False
    n = weight.numel()
    if hidden_states.shape[-1] != n:
        return False
    if n > _MAX_FUSED_COLS or n == 0:
        return False
    if hidden_states.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if not hidden_states.is_cuda:
        return False
    if not (hidden_states.is_contiguous() and weight.is_contiguous()):
        return False
    return True


def fused_rmsnorm_decode(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    return_intermediates: bool = False,
):
    """三 kernel 融合的 Qwen3RMSNorm 前向（仅 seq_len==1 decode，先过 is_fusable）。

    返回与 HF Qwen3RMSNorm.forward 逐位对齐的 BF16 输出（shape 同输入）。
    return_intermediates=True 时额外返回 (square, mean)，供对拍脚本逐级
    定位 bit 差异（square 对应 HF 的 fp32 pow(2) 输出、mean 对应 variance）。

    Graph capture 注意：intermediate square/mean 为函数内临时分配，
    capture 模式下进入 graph 私有池由图持有；返回的 out 张量必须由
    调用方保持引用（否则图持悬垂指针——既有 CUDA graph 生命周期纪律）。
    """
    assert is_fusable(hidden_states, weight), (
        "fused_rmsnorm_decode 仅接受 seq_len==1 的 BF16 contiguous CUDA 输入；"
        "调用方应先用 is_fusable 过滤，prefill/其它布局走 HF 原 forward"
    )

    n = weight.numel()
    x2 = hidden_states.view(-1, n)          # gate 已保证 contiguous，view 零拷贝
    rows = x2.shape[0]                       # decoder norm: B；Q/K norm: B*num_heads
    block = _next_pow2(n)
    num_warps = _pick_num_warps(block)

    square = torch.empty(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
    out = torch.empty_like(hidden_states)
    sq2 = square.view(-1, n)
    out2 = out.view(-1, n)

    # ---- kernel A：BF16 x → FP32 square ----
    _square_fp32_kernel[(rows,)](
        x2, sq2,
        x2.stride(0), sq2.stride(0),
        n,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )

    # ---- kernel B：原 torch.mean，kernel 与归约顺序完全不改 ----
    # square 与 HF pow(2) 输出的值/shape/dtype/布局全同 → 逐位同 mean。
    mean = torch.mean(square, dim=-1, keepdim=True)
    mean_flat = mean.view(-1)                # (..., 1) contiguous → 展平零拷贝

    # ---- kernel C：norm + bf16 舍入 + weight 缩放 ----
    _norm_scale_kernel[(rows,)](
        x2, mean_flat, weight, out2,
        x2.stride(0), mean_flat.stride(0), out2.stride(0),
        n,
        eps,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )

    if return_intermediates:
        return out, square, mean
    return out


def fused_rmsnorm_decode_single(hidden_states, weight, eps):
    """单 kernel 实验入口；生产仍默认使用逐位归约一致的三 kernel 路径。"""
    assert is_fusable(hidden_states, weight)
    n = weight.numel()
    x2 = hidden_states.view(-1, n)
    out = torch.empty_like(hidden_states)
    out2 = out.view(-1, n)
    block = _next_pow2(n)
    _rmsnorm_single_kernel[(x2.shape[0],)](
        x2, weight, out2, x2.stride(0), out2.stride(0), n, eps,
        BLOCK_SIZE=block, num_warps=_pick_num_warps(block))
    return out
