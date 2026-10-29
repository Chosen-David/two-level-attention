"""Single-request selector intermediate for later exact batch consumers.

This carries existing producer tensors, without gathering K, scoring, normalizing
or ranking. The producer owns the active-prefix invariant; consumers must retain
per-query normalization and verify their supported dtype/ranking domain.
"""
from typing import NamedTuple
import torch


class CandidateInputs(NamedTuple):
    k_fine: torch.Tensor       # [1,S,Hkv,Dfine], existing index feature storage
    q_fine: torch.Tensor       # [Hq,Dfine], FP32 after model-dtype scale rounding
    ids: torch.Tensor          # [Hkv,R], int64; active candidates form a prefix
    keep: torch.Tensor         # [Hkv,R], bool; includes pre-existing tail ordering
    counts: torch.Tensor       # [Hkv], int64 device counts
    length: torch.Tensor       # scalar int64 effective length
    lower: torch.Tensor        # scalar int64 ranking interval lower bound
    upper: torch.Tensor        # scalar int64 ranking interval upper bound
    quota: torch.Tensor        # scalar int64 middle-token quota
    capacity: int
    group_size: int
    output_dtype: torch.dtype
    rank_width: int
    sink: int
    swa: int


def candidate_inputs(k_fine, q_fine, ids, keep, counts, length, lower, upper,
                     quota, *, capacity, group_size, output_dtype, rank_width,
                     sink, swa):
    """Validate metadata only and retain exact tensor identities (capture safe).

    No tensor values are read to CPU, no active-prefix reorder/dedup is performed.
    These are candidates for the selector denominator, not final attention IDs;
    protection is still added separately by the attention consumer.
    """
    if (type(capacity) is not int or capacity < 1
            or type(group_size) is not int or group_size < 1
            or type(rank_width) is not int or not 0 <= rank_width <= capacity
            or any(type(x) is not int or x < 0 for x in (sink, swa))):
        raise ValueError('invalid candidate producer geometry')
    if (k_fine.ndim != 4 or k_fine.shape[0] != 1
            or k_fine.shape[1] != capacity or min(k_fine.shape[2:]) < 1
            or ids.ndim != 2 or ids.shape[0] != k_fine.shape[2]
            or keep.shape != ids.shape or counts.shape != ids.shape[:1]
            or q_fine.shape != (ids.shape[0] * group_size, k_fine.shape[-1])):
        raise ValueError('incompatible candidate producer shapes')
    if (ids.dtype != torch.int64 or counts.dtype != torch.int64
            or keep.dtype != torch.bool or q_fine.dtype != torch.float32
            or output_dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or k_fine.dtype != output_dtype):
        raise ValueError('invalid candidate producer dtypes')
    tensors = (k_fine, q_fine, ids, keep, counts, length, lower, upper, quota)
    if (any(t.device != k_fine.device for t in tensors)
            or any(t.ndim != 0 or t.dtype != torch.int64
                   for t in (length, lower, upper, quota))):
        raise ValueError('candidate metadata requires same-device int64 scalars')
    return CandidateInputs(*tensors, capacity, group_size, output_dtype,
                           rank_width, sink, swa)
