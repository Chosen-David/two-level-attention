"""Fail-closed mask gates shared by the active E139 validation scripts."""
import torch


def require_equal_masks(label, expected, actual, group_ends, kv_heads=8):
    """Validate complete output schemas before comparing bits (also under -O).

    These scripts generate Hkv=8 inputs. A fallback (None), missing rows, wrong
    mask schema, or any bit mismatch must terminate before performance output.
    """
    count = len(group_ends)
    for name, masks in (("reference", expected), ("actual", actual)):
        if not isinstance(masks, (list, tuple)) or len(masks) != count:
            got = len(masks) if isinstance(masks, (list, tuple)) else type(masks).__name__
            raise SystemExit(f"validation failed: {label}: {name} rows={got}, expected={count}")
        for row, (mask, end) in enumerate(zip(masks, group_ends)):
            if (not isinstance(mask, torch.Tensor)
                    or tuple(mask.shape) != (kv_heads, end)
                    or mask.dtype != torch.bool):
                raise SystemExit(f"validation failed: {label}: {name} row={row} invalid mask schema")
    diff = 0
    for row, (a, b) in enumerate(zip(expected, actual)):
        if a.device != b.device:
            raise SystemExit(f"validation failed: {label}: row={row} device mismatch")
        diff += int(not torch.equal(a, b))
    print(f"{label} diff_rows = {diff}/{count}", flush=True)
    if diff:
        raise SystemExit(f"validation failed: {label}: {diff}/{count} mask rows differ")
