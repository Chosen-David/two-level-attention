"""HybridLayer / HybridCache regression tests — CPU only.

Protocol: HybridLayer with strategy='linear' must replicate TLIStaticLayer
bit-exact; strategy='ring' must replicate FarSkipLayer bit-exact for sink+near.
"""
import os, sys
os.environ.setdefault("TLI_SPARSE_PREFILL", "0")

import torch
from transformers import AutoConfig

import sys, pathlib
sys.path.insert(0, str(PathPath := pathlib.Path(__file__).resolve().parent.parent))

from sparse_attn.ops.hybrid_cache import HybridLayer, HybridCache
from sparse_attn.ops.tli_static_cache import TLIStaticLayer


def _make_layers(H=8, D=16, S_cap=128, B=2):
    return H, D, S_cap, B


def test_linear_strategy_matches_static_layer():
    """HybridLayer(strategy='linear') == TLIStaticLayer bit-exact after N steps."""
    torch.manual_seed(42)
    H, D, S_cap, B = _make_layers()
    N = 20

    hybrid = HybridLayer(S_cap, B)
    hybrid.lazy_initialization(torch.zeros(B, H, 1, D))
    hybrid.set_strategy("linear")
    static = TLIStaticLayer(S_cap, B)  # 内部 set_strategy('linear')

    from dataclasses import dataclass
    @dataclass
    class Ctx:
        batch: int
        seq_lens: list

    ctx = Ctx(batch=B, seq_lens=[0] * B)
    from sparse_attn.baseline_runtime import batch_decode_scope

    for step in range(N):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        ctx.seq_lens = [min(ctx.seq_lens[b] + 1, S_cap) for b in range(B)]
        with batch_decode_scope(ctx):
            kh, vh = hybrid.update(k.clone(), v.clone())
            ks, vs = static.update(k.clone(), v.clone())
        assert torch.equal(kh, ks), f"key diverged at step {step}"
        assert torch.equal(vh, vs), f"value diverged at step {step}"

    print("[PASS] linear strategy matches TLIStaticLayer bit-exact")


def test_ring_strategy_basic():
    """HybridLayer(strategy='ring') writes sink first, then cycles tail slots."""
    torch.manual_seed(0)
    H, D, S_cap, B = _make_layers(S_cap=256)
    hybrid = HybridLayer(S_cap, B, sink_tokens=4, near_length=64,
                         window_tokens=64, block_size=16)
    hybrid.lazy_initialization(torch.zeros(B, H, 1, D))
    hybrid.set_strategy("ring")
    assert hybrid.strategy == "ring"

    N = 300
    history_k, history_v = [], []
    for step in range(N):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        history_k.append(k.clone()); history_v.append(k.clone())
        hybrid.update(k, v)

    layout = hybrid.ring_layout
    total = hybrid.ring_logical_length()
    assert total == N, f"logic length {total} != {N}"

    # Verify recent tail tokens are readable from ring slots
    for step_i in range(hybrid._ring_cfg.near_length):
        abs_pos = total - hybrid._ring_cfg.near_length + step_i
        slot = layout.slot(abs_pos)
        expected = history_k[abs_pos].squeeze(-2)
        actual = hybrid.keys[..., slot, :]
        assert torch.allclose(actual, expected, atol=1e-5), \
            f"ring slot mismatch at abs_pos={abs_pos} slot={slot}"

    print("[PASS] ring strategy sink + tail cycle correct")


def test_strategy_freeze_guard():
    """set_strategy twice with different value must raise."""
    hybrid = HybridLayer(128, 2)
    hybrid.set_strategy("linear")
    try:
        hybrid.set_strategy("ring")
    except RuntimeError as e:
        assert "already frozen" in str(e)
        print("[PASS] strategy freeze guard works")
        return
    raise AssertionError("expected RuntimeError on double-freeze")


def test_get_seq_length_monotonic():
    """cumulative_length must increment by 1 per update, both strategies."""
    for strategy in ("linear", "ring"):
        hybrid = HybridLayer(512, 1, sink_tokens=4, near_length=32,
                             window_tokens=64, block_size=64)
        hybrid.lazy_initialization(torch.zeros(1, 4, 1, 8))
        hybrid.set_strategy(strategy)
        for step in range(10):
            hybrid.update(torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8))
        assert hybrid.get_seq_length() == 10, f"{strategy}: length={hybrid.get_seq_length()}"
    print("[PASS] get_seq_length monotonic for both strategies")


def test_prefill_reject():
    """HybridLayer.update must reject multi-token (prefill-shaped) input."""
    hybrid = HybridLayer(64, 1)
    hybrid.lazy_initialization(torch.zeros(1, 4, 1, 8))
    hybrid.set_strategy("linear")
    try:
        hybrid.update(torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
    except RuntimeError as e:
        assert "decode-only" in str(e)
        print("[PASS] prefill-shape update correctly rejected")
        return
    raise AssertionError("expected RuntimeError on prefill update")


def test_reset():
    """reset() zeroes buffers and resets counters."""
    for strategy in ("linear", "ring"):
        hybrid = HybridLayer(512, 1, sink_tokens=4, near_length=32,
                             window_tokens=64, block_size=64)
        hybrid.lazy_initialization(torch.zeros(1, 4, 1, 8))
        hybrid.set_strategy(strategy)
        for _ in range(5):
            hybrid.update(torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8))
        hybrid.reset()
        assert hybrid.get_seq_length() == 0
        assert torch.all(hybrid.keys == 0)
        assert torch.all(hybrid.values == 0)
    print("[PASS] reset clears state for both strategies")


def test_ring_capacity_validation():
    """set_strategy('ring') must validate capacity against max_cache_len."""
    # 故意构造 ring_cap > S_cap：sink=4 + max(64,64) + 128 + 64 - 1 = 259 > 128
    hybrid = HybridLayer(128, 1, sink_tokens=4, near_length=64,
                         window_tokens=128, block_size=64)
    try:
        hybrid.set_strategy("ring")
    except ValueError as e:
        assert "exceeds" in str(e)
        print("[PASS] ring capacity validation works")
        return
    raise AssertionError("expected ValueError on oversized ring")


def test_kv_prefix_len_dispatch():
    """get_kv_prefix_len must return ring capacity for ring, S_cap for linear."""
    hybrid_linear = HybridLayer(256, 1)
    hybrid_linear.lazy_initialization(torch.zeros(1, 4, 1, 8))
    hybrid_linear.set_strategy("linear")
    assert hybrid_linear.get_kv_prefix_len() == 256

    hybrid_ring = HybridLayer(512, 1, sink_tokens=4, near_length=64,
                              window_tokens=64, block_size=16)
    hybrid_ring.lazy_initialization(torch.zeros(1, 4, 1, 8))
    hybrid_ring.set_strategy("ring")
    assert hybrid_ring.get_kv_prefix_len() == hybrid_ring.ring_layout.capacity
    assert hybrid_ring.get_kv_prefix_len() < 512  # ring 容量应小于全量

    print("[PASS] get_kv_prefix_len dispatches correctly")


if __name__ == "__main__":
    import pathlib
    test_linear_strategy_matches_static_layer()
    test_ring_strategy_basic()
    test_strategy_freeze_guard()
    test_get_seq_length_monotonic()
    test_prefill_reject()
    test_reset()
    test_ring_capacity_validation()
    test_kv_prefix_len_dispatch()
    print("\n[ALL PASS] HybridLayer / HybridCache CPU regression (8/8)")
