"""Batch context and power-of-two bucket selection."""
from sparse_attn.baseline_runtime import (
    BatchCtx,
    batch_decode_ctx,
    batch_decode_scope,
    clear_batch_decode_ctx,
    set_batch_decode_ctx,
)


def batch_bucket_for(batch, max_batch=None):
    """Return the smallest power-of-two bucket >= batch, optionally capped."""
    if batch < 1:
        raise ValueError("batch must be positive")
    bucket = 1
    while bucket < batch:
        bucket <<= 1
    return bucket if max_batch is None else min(bucket, max_batch)


def pad_to_bucket(req_ids, max_batch=None):
    bucket = batch_bucket_for(len(req_ids), max_batch)
    return req_ids + [-1] * (bucket - len(req_ids))
