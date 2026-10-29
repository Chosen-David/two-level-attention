# -*- coding: utf-8 -*-
"""Operator pipeline (Phase 0 stub).

P0 routes everything to legacy. Phases 1-7 progressively enable:
  'off'     -> legacy qwen3_attn_patch path (default now)
  'serial'  -> serial graph + slot-based selection
  'overlap' -> fork/join graph with protected-stream overlap
"""


def operator_mode() -> str:
    import os
    return os.environ.get("TLI_OPERATOR_MODE", "off")


def decode_indexed(*args, **kwargs):
    raise NotImplementedError(
        "decode_indexed is not available when TLI_OPERATOR_MODE=off")


def decode_ring(*args, **kwargs):
    raise NotImplementedError(
        "decode_ring is not available when TLI_OPERATOR_MODE=off")
