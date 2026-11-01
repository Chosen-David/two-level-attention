"""
DSA TopK Indexer — fused FP8 Triton scoring + bundled CUDA radix top-K.

Dispatch on `max_seq_padded = block_table.shape[1] * PAGE_SIZE`:
  - `max_seq_padded <= TOPK`: passthrough — every valid token id goes in the output.
  - `max_seq_padded >  TOPK`: FP8 score kernel → FilteredTopK + page-table transform.

Fixed Constants: HEAD_COUNT=64, HEAD_DIM=128, TOPK=2048, PAGE_SIZE=64.
"""

import torch
import triton
import triton.language as tl

from .topk_ext import top_k_page_table_transform

PAGE_SIZE = 64
HEAD_COUNT = 64
HEAD_DIM = 128
TOPK = 2048

# ---------------------------------------------------------------------------
# Passthrough path (max_seq_padded <= TOPK)
# ---------------------------------------------------------------------------

@triton.jit
def _passthrough_single_page_kernel(
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    stride_block_table: tl.constexpr,
    stride_out: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
):
    """
    Max pages=1 fast path: one program per batch, single block_table load.
    """
    batch_id = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + batch_id)
    physical_page_id = tl.load(block_table_ptr + batch_id * stride_block_table).to(tl.int32)

    offsets = tl.arange(0, TOPK)
    valid = offsets < tl.minimum(seq_len, PAGE_SIZE)
    physical_token = physical_page_id * PAGE_SIZE + offsets
    out = tl.where(valid, physical_token, tl.full(physical_token.shape, -1, dtype=tl.int32))
    tl.store(out_ptr + batch_id * stride_out + offsets, out)


@triton.jit
def _passthrough_multi_page_kernel(
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    stride_block_table: tl.constexpr,
    stride_out: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Small-workload fast path: emit all valid token ids from block_table.
    When max_seq_padded <= TOPK, seq_len <= TOPK, so the top-K answer is just every
    valid token id in [0, seq_len). Writes -1 for positions >= seq_len to fill topk_indices.

    BLOCK_K is a tuning parameter: the number of tokens each program emits.
    """
    batch_id = tl.program_id(0)
    k_tile_id = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch_id)

    offsets = k_tile_id * BLOCK_K + tl.arange(0, BLOCK_K)
    valid = offsets < seq_len

    logical_page = (offsets // PAGE_SIZE).to(tl.int32)
    offset_in_page = (offsets % PAGE_SIZE).to(tl.int32)
    safe_page_id = tl.where(valid, logical_page, 0)
    physical_page = tl.load(block_table_ptr + batch_id * stride_block_table + safe_page_id)
    physical_token = physical_page * PAGE_SIZE + offset_in_page

    out = tl.where(valid, physical_token, tl.full(physical_token.shape, -1, dtype=tl.int32))
    tl.store(out_ptr + batch_id * stride_out + offsets, out, mask=offsets < stride_out)


def _run_passthrough(batch_size, max_pages, block_table, seq_lens, topk_indices):
    """
    When max_seq_padded <= TOPK, the top-K answer is just every valid token id in [0, seq_len).
    """
    if max_pages == 1:
        _passthrough_single_page_kernel[(batch_size,)](
            block_table_ptr=block_table,
            seq_lens_ptr=seq_lens,
            out_ptr=topk_indices,
            stride_block_table=block_table.stride(0),
            stride_out=topk_indices.stride(0),
            PAGE_SIZE=PAGE_SIZE,
            TOPK=TOPK,
            num_warps=4,
        )
        return

    # Each program emits BLOCK_K tokens with a total of TOPK outputs.
    BLOCK_K = 256
    grid = (batch_size, triton.cdiv(TOPK, BLOCK_K))
    _passthrough_multi_page_kernel[grid](
        block_table_ptr=block_table,
        seq_lens_ptr=seq_lens,
        out_ptr=topk_indices,
        stride_block_table=block_table.stride(0),
        stride_out=topk_indices.stride(0),
        PAGE_SIZE=PAGE_SIZE,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Top-K path (max_seq_padded > TOPK).
#
# Score kernel uses TOKENS_PER_TILE=PAGE_SIZE=64, one page per program.
# num_warps=2 keeps threads/block at 64, which matches the tcgen05
# 64x64x128 MMA tile and lets 32 blocks fit per SM on B200 — so the
# (BATCH_SIZE, max_num_pages) grid clears in a single wave.
# ---------------------------------------------------------------------------

@triton.jit
def _score_kernel(
    q_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    weights_ptr,
    seq_lens_ptr,
    block_table_ptr,
    scores_ptr,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_page: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_k_scale_page: tl.constexpr,
    stride_k_scale_token: tl.constexpr,
    stride_weights_batch: tl.constexpr,
    stride_weights_head: tl.constexpr,
    stride_block_table_batch: tl.constexpr,
    stride_block_table_page: tl.constexpr,
    stride_scores_batch: tl.constexpr,
    stride_scores_token: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_TILE: tl.constexpr,  # fixed to PAGE_SIZE, one page per program.
):
    """
    Score one KV page per program: FP8 Q·Kᵀ → ReLU → weighted head-sum → per-token scale.
    Writes TOKENS_PER_TILE scores into scores_buf; -1e30 past seq_len.
    """
    batch_id = tl.program_id(0)
    logical_page = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + batch_id)
    token_start = logical_page * TOKENS_PER_TILE

    if token_start >= seq_len:
        pad_token_offsets = tl.arange(0, TOKENS_PER_TILE)
        pad_score_offsets = batch_id * stride_scores_batch + (token_start + pad_token_offsets) * stride_scores_token
        tl.store(scores_ptr + pad_score_offsets, tl.full([TOKENS_PER_TILE], -1e30, tl.float32))
        return

    physical_page = tl.load(block_table_ptr + batch_id * stride_block_table_batch + logical_page * stride_block_table_page).to(tl.int64)

    head_offsets = tl.arange(0, HEAD_COUNT)
    token_offsets = tl.arange(0, TOKENS_PER_TILE)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_id * stride_q_batch,
        shape=(HEAD_COUNT, HEAD_DIM),
        strides=(stride_q_head, stride_q_dim),
        offsets=(0, 0),
        block_shape=(HEAD_COUNT, HEAD_DIM),
        order=(1, 0),
    )
    q_fp8 = tl.load(q_block_ptr)

    k_block_ptr = tl.make_block_ptr(
        base=k_fp8_ptr + physical_page * stride_k_page,
        shape=(TOKENS_PER_TILE, HEAD_DIM),
        strides=(stride_k_token, stride_k_dim),
        offsets=(0, 0),
        block_shape=(TOKENS_PER_TILE, HEAD_DIM),
        order=(1, 0),
    )
    k_fp8 = tl.load(k_block_ptr)

    scale_offsets = physical_page * stride_k_scale_page + token_offsets * stride_k_scale_token
    scale = tl.load(k_scale_ptr + scale_offsets)
    weight_offsets = batch_id * stride_weights_batch + head_offsets * stride_weights_head
    weight = tl.load(weights_ptr + weight_offsets)

    scores_tile = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)
    scores_tile = tl.maximum(scores_tile, 0.0)
    scores_tile = scores_tile * weight[:, None]
    per_token_score = tl.sum(scores_tile, axis=0) * scale

    abs_token_ids = token_start + token_offsets
    valid_tokens = abs_token_ids < seq_len
    per_token_score = tl.where(valid_tokens, per_token_score, -1e30)

    score_offsets = batch_id * stride_scores_batch + abs_token_ids * stride_scores_token
    tl.store(scores_ptr + score_offsets, per_token_score)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _make_kv_views(k_index_cache_fp8):
    """
    Zero-copy strided views into the deep_gemm SOA cache.

    The cache's declared shape is [P, 64, 1, 132] int8, but actual memory is
    SOA per page: [8192 fp8 bytes][256 scale bytes]. Build:
      - k_fp8:   [P, 64, 128] fp8,     stride (8448, 128, 1)
      - k_scale: [P, 64]      float32, stride (2112, 1), storage_offset=2048
    """
    num_pages, page_size, _, head_dim_plus_scale_factor = k_index_cache_fp8.shape
    head_dim = head_dim_plus_scale_factor - 4             # 128
    stride_page_i8 = page_size * head_dim_plus_scale_factor      # 8448 fp8 bytes per page
    stride_page_f32 = stride_page_i8 // 4  # 2112 float32 elements per page

    cache_fp8_flat = k_index_cache_fp8.view(torch.float8_e4m3fn).view(num_pages, stride_page_i8)
    k_fp8 = torch.as_strided(
        cache_fp8_flat,
        size=(num_pages, page_size, head_dim),
        stride=(stride_page_i8, head_dim, 1),
    )
    cache_f32_flat = k_index_cache_fp8.view(torch.float32).view(num_pages, stride_page_f32)
    k_scale = cache_f32_flat[:, page_size * head_dim // 4:]
    return k_fp8, k_scale


_topk_scratch_state = {}


def _get_topk_scratch(batch_size, max_seq_padded, device):
    """
    Cache a [B, next_pow2(max_seq_padded)] bf16 scratch for the score output.
    """
    key = (batch_size, max_seq_padded)
    buf = _topk_scratch_state.get(key)
    if buf is not None:
        return buf
    max_n = _next_pow2(max_seq_padded)
    buf = torch.empty((batch_size, max_n), dtype=torch.bfloat16, device=device)
    _topk_scratch_state[key] = buf
    return buf


def _run_topk(batch_size, max_seq_padded, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """
    Score every token via the FP8 score kernel, then FilteredTopK + page-table transform.
    """
    scores_buf = _get_topk_scratch(batch_size, max_seq_padded, q_index_fp8.device)
    k_fp8, k_scale = _make_kv_views(k_index_cache_fp8)
    max_num_pages = block_table.shape[1]

    _score_kernel[(batch_size, max_num_pages)](
        q_index_fp8,
        k_fp8,
        k_scale,
        weights,
        seq_lens,
        block_table,
        scores_buf,
        stride_q_batch=q_index_fp8.stride(0),
        stride_q_head=q_index_fp8.stride(1),
        stride_q_dim=q_index_fp8.stride(2),
        stride_k_page=k_fp8.stride(0),
        stride_k_token=k_fp8.stride(1),
        stride_k_dim=k_fp8.stride(2),
        stride_k_scale_page=k_scale.stride(0),
        stride_k_scale_token=k_scale.stride(1),
        stride_weights_batch=weights.stride(0),
        stride_weights_head=weights.stride(1),
        stride_block_table_batch=block_table.stride(0),
        stride_block_table_page=block_table.stride(1),
        stride_scores_batch=scores_buf.stride(0),
        stride_scores_token=scores_buf.stride(1),
        HEAD_COUNT=HEAD_COUNT,
        HEAD_DIM=HEAD_DIM,
        TOKENS_PER_TILE=PAGE_SIZE,
        num_warps=2,
        num_stages=2,
    )

    top_k_page_table_transform(scores_buf, block_table, seq_lens, TOPK, output_page_table=topk_indices)


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """
    DSA TopK Indexer: passthrough when max_seq_padded ≤ TOPK, else FP8 score → radix top-K + page-table remap.
    """
    batch_size = q_index_fp8.shape[0]
    max_seq_padded = block_table.shape[1] * PAGE_SIZE

    if max_seq_padded <= TOPK:
        _run_passthrough(batch_size, block_table.shape[1], block_table, seq_lens, topk_indices)
        return

    _run_topk(batch_size, max_seq_padded, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)