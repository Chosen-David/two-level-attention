"""Eager-mode decode runner used for debug and prefill staging."""
from sparse_attn.ops.eager_decoding import eager_decoding_attn


def eager_decode(*args, **kwargs):
    return eager_decoding_attn(*args, **kwargs)
