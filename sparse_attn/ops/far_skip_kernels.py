"""Data-movement kernels for compact decode; no score/softmax approximations.

Opt in with TLI_FAR_SKIP_KERNELS=triton. CUDA validation and latency on the
deployment GPU are required before promoting this experimental backend.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['Position'])
def write_pair(K, V, NewK, NewV, Position, H: tl.constexpr, D: tl.constexpr,
               K_H: tl.constexpr, K_S: tl.constexpr, K_D: tl.constexpr,
               V_H: tl.constexpr, V_S: tl.constexpr, V_D: tl.constexpr,
               NK_H: tl.constexpr, NK_D: tl.constexpr,
               NV_H: tl.constexpr, NV_D: tl.constexpr,
               SINK: tl.constexpr, TAIL: tl.constexpr,
               TENSOR_POSITION: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.load(Position) if TENSOR_POSITION else Position
    slot = tl.where(position < SINK, position, SINK + tl.maximum(position-SINK, 0) % TAIL)
    x = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    h, d = x // D, x % D
    k = tl.load(NewK + h*NK_H + d*NK_D, h < H, 0)
    v = tl.load(NewV + h*NV_H + d*NV_D, h < H, 0)
    tl.store(K + h*K_H + slot*K_S + d*K_D, k, h < H)
    tl.store(V + h*V_H + slot*V_S + d*V_D, v, h < H)


@triton.jit(do_not_specialize=['N', 'S0', 'S1', 'V0', 'V1'])
def order_keys(Scores, Valid, Keys, N, S0, S1, V0, V1,
               HAS_VALID: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Scores + row*S0 + col*S1, col < N, 0).to(tl.float32)
    # Explicit canonicalization: an optimized x+0 may retain negative zero.
    value = tl.where(value == 0., 0., value)
    bits = value.to(tl.int32, bitcast=True).to(tl.int64)
    ordered = bits ^ ((bits >> 31) & 0x7fffffff)
    ordered = tl.where(value != value, 0x7fffffff, ordered)
    keys = ordered * N + (N - 1 - col.to(tl.int64))
    if HAS_VALID:
        valid = tl.load(Valid + row*V0 + col*V1, col < N, 0)
        keys = tl.where(valid, keys, -9223372036854775807 - 1)
    tl.store(Keys + row*N + col, keys, col < N)


@triton.jit(do_not_specialize=['N', 'X_S', 'X_H', 'X_D', 'Y_S', 'Y_H', 'Y_D',
                             'P_H', 'P_N', 'V_H', 'V_N', 'DIM_STRIDE'])
def gather_rows(X, Y, Positions, Valid, Dims, OutX, OutY,
                N, D: tl.constexpr, X_S, X_H, X_D, Y_S, Y_H, Y_D,
                P_H, P_N, V_H, V_N, DIM_STRIDE,
                SINK: tl.constexpr, TAIL: tl.constexpr,
                HAS_VALID: tl.constexpr, HAS_DIMS: tl.constexpr, PAIR: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    n = tl.program_id(1)*BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)
    valid = n < N
    if HAS_VALID:
        valid = valid & tl.load(Valid + h*V_H + n*V_N, n < N, 0)
    pos = tl.load(Positions + h*P_H + n*P_N, valid, 0)
    if TAIL > 0:
        # tl % is signed remainder. Clamp inactive prefix lanes before modulo;
        # the prefix branch subsequently restores their absolute positions.
        mapped = SINK + tl.maximum(pos-SINK, 0) % TAIL
        pos = tl.where(pos < SINK, pos, mapped)
    dim = d
    if HAS_DIMS:
        dim = tl.load(Dims + d*DIM_STRIDE, d < D, 0)
    mask = valid[:, None] & (d[None, :] < D)
    off = pos[:, None]*X_S + h*X_H + dim[None, :]*X_D
    x = tl.load(X + off, mask, 0)
    dst = (h*N + n[:, None])*D + d[None, :]
    tl.store(OutX + dst, x, (n[:, None] < N) & (d[None, :] < D))
    if PAIR:
        yoff = pos[:, None]*Y_S + h*Y_H + dim[None, :]*Y_D
        y = tl.load(Y + yoff, mask, 0)
        tl.store(OutY + dst, y, (n[:, None] < N) & (d[None, :] < D))
