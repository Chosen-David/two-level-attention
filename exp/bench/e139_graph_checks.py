"""E139 hard gates: explicit exceptions remain active under python -O."""
import torch


def require_masks(actual, expected, rows):
    if actual is None or expected is None:
        raise RuntimeError("selector fell back instead of returning masks")
    if len(actual) != rows or len(expected) != rows or rows <= 0:
        raise RuntimeError("missing or unexpected mask rows")
    for a, b in zip(actual, expected):
        if (a.shape != b.shape or a.dtype != torch.bool or b.dtype != torch.bool
                or a.device != b.device or not torch.equal(a, b)):
            raise RuntimeError("mask equality gate failed")


def require_outputs(actual, expected):
    if actual is None or expected is None or actual.numel() == 0:
        raise RuntimeError("missing output")
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.device != expected.device):
        raise RuntimeError("output schema gate failed")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise RuntimeError("nonfinite output")
    if not torch.allclose(actual.float(), expected.float(), rtol=1e-2, atol=1e-3):
        raise RuntimeError("output allclose gate failed")


def require_graph(before, after, event):
    if after['fallback'] != before['fallback'] or after[event] <= before[event]:
        raise RuntimeError("required Graph route did not execute cleanly: " + event)
