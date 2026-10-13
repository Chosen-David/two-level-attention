# -*- coding: utf-8 -*-
"""GPT 审计 B 批 MInference 主题红绿测试（135 padding mask / 136 chunk cache）。

TL-MINF-PADMASK-135（B3）
  红：qwen3_minference_patch wrapper 把 attention_mask 放进 prefill_kwargs，
      但稀疏层（vertical_and_slash / block_sparse）与浅层 flash_attn_func
      均不消费 → padding 改变有效 token 输出（被 pad 行错读全 KV），静默
      错误输出。本测试 monkeypatch 守卫为空（模拟修复前形态）+ mock 官方
      kernel，实证 padded mask 原样流入 prefill_kwargs 且无人校验。
  绿：修复后 wrapper 在进入稀疏 prefill 分支前检测「非平凡 padding mask」
      → 显式 RuntimeError（中文，含审计编号），拒收不静默。纯因果 4D mask
      （eager 全量 prefill 正常态）与 mask=None 不误杀。

TL-MINF-CHUNKCACHE-136（B4）
  红：仅按 q_len==1 分 decode——q_len>1 且 cache 已有 token（chunked
      prefill with prefix cache）仍走稀疏 prefill，下游 kernel 按
      Q_len==K_len 方阵协议错位（复刻：守卫关闭时 q_len=5 + cache=3 直接
      进入 sparse 分支无任何报错）。
  绿：修复后检测「past_key_values 更新前已缓存 token 且 q_len>1」→ 显式
      RuntimeError（检测点在 cache update 之前，get_seq_length() 为更新前
      长度）；首段全量 prefill（cache=0）与 decode（q_len=1）不误杀。

运行要求：本机树存在 minference_vendor 工作目录（minference 官方包为
vendored 依赖）；缺包时 SKIP（不 FAIL）。

用法（本仓根）：
  PYTHONPATH=$PWD python3 -m exp.trace.test_gpt_b_batch_minference
"""
import os
import sys
import types
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
VENDOR = os.path.join(REPO, "exp", "vendor_baselines", "minference_vendor")
sys.path.insert(0, VENDOR)
sys.path.insert(0, REPO)

import torch  # noqa: E402

try:
    import qwen3_minference_patch as patch_mod  # noqa: E402
except Exception as e:  # noqa: BLE001——缺 vendored 依赖时如实 SKIP
    print(f"[MINFERENCE-TEST] SKIP: minference_vendor 依赖不可导入（{e}）")
    sys.exit(0)

from qwen3_minference_patch import (  # noqa: E402
    _has_nontrivial_padding, _prefill_guard_violation,
    _qwen3_minference_wrapper,
)


def _check(cond, msg=""):
    """与 E119 套件同纪律：显式判定（非 assert），-O 下不失效。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _causal_mask(q_len, kv_len, dtype=torch.float32, bsz=1):
    """构造 4.56 create_causal_mask 形态的纯因果加性 mask
    （0=可见，dtype min=屏蔽，上三角）。"""
    m = torch.zeros(bsz, 1, q_len, kv_len, dtype=dtype)
    neg = torch.finfo(dtype).min
    for i in range(q_len):
        m[:, :, i, (kv_len - q_len) + i + 1:] = neg
    return m


def _pad(causal, tail=3):
    """在纯因果 mask 上叠加右 padding（末 KV 列屏蔽）。"""
    out = causal.clone()
    out[:, :, :, -tail:] = torch.finfo(causal.dtype).min
    return out


class _FakeCache:
    """get_seq_length/update 语义对齐 HF DynamicCache（无参调用返回当前
    已缓存长度；update 返回展开后的 k/v——首段全量 prefill 会真正走到）。
    136 守卫须在 update 之前判定（更新后 get_seq_length 恒>0 会误杀）。"""

    def __init__(self, n):
        self.n = n

    def get_seq_length(self, layer_idx=0):
        return self.n

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states


def _call_wrapper(self, hidden_states, attention_mask=None,
                  past_key_values=None):
    """以 mock 官方 kernel 调 wrapper，返回 ((status, payload), captured)。"""
    captured = {}

    def fake_prefill(q, k, v, prefill_kwargs):
        captured["q_len"] = q.shape[2]
        captured["kv_len"] = k.shape[2]
        captured["attention_mask"] = prefill_kwargs.get("attention_mask")
        return torch.zeros_like(q)

    # cos/sin 形态对齐 Qwen3RotaryEmbedding 产物 [bs, seq, head_dim]
    seq = hidden_states.shape[1]
    pe = (torch.zeros(1, seq, 8), torch.zeros(1, seq, 8))
    with mock.patch.object(patch_mod, "minference_prefill_forward",
                           side_effect=fake_prefill), \
         mock.patch.object(patch_mod, "repeat_kv", side_effect=lambda t, n: t):
        try:
            out = _qwen3_minference_wrapper(
                self, hidden_states, pe,
                attention_mask, past_key_values, None)
            return ("ok", out), captured
        except RuntimeError as e:
            return ("raise", str(e)), captured


def _fake_self(q_len, kv_len):
    """构造能走完稀疏 prefill 分支的最小 Qwen3Attention 替身（CPU）。"""
    bsz, heads, head_dim = 1, 2, 8
    s = types.SimpleNamespace()
    s.layer_idx = 1
    s.head_dim = head_dim
    s.config = types.SimpleNamespace(num_hidden_layers=4)
    s._minference_attn_config = {"is_search": False, "starting_layer": -1,
                                 "best_pattern": [[{}]] * 4}

    def proj(x):
        return torch.zeros(*x.shape[:-1], heads * head_dim)
    s.q_proj = s.k_proj = s.v_proj = s.o_proj = proj
    s.q_norm = s.k_norm = (lambda x: x)
    s.num_heads = heads
    s.num_key_value_heads = heads
    s.flash_forward = (lambda *a, **k: (torch.zeros(0), None))
    return s


def test_135_padmask():
    """135：红（守卫关闭 padded mask 静默流入 kernel）/ 绿（显式拒收）。"""
    q_len, kv_len = 6, 6
    hs = torch.zeros(1, q_len, 8)
    causal = _causal_mask(q_len, kv_len)
    padded = _pad(causal, tail=2)

    # ---- 单元级：_has_nontrivial_padding 语义 ----
    _check(_has_nontrivial_padding(None) is False, "135: None 应判平凡")
    _check(_has_nontrivial_padding(causal) is False,
           "135: 纯因果 mask（无 padding）不得误杀")
    _check(_has_nontrivial_padding(padded) is True,
           "135: 右 padding 应判非平凡")
    left_pad = causal.clone()
    left_pad[:, :, :, :2] = torch.finfo(torch.float32).min
    _check(_has_nontrivial_padding(left_pad) is True,
           "135: 左 padding 应判非平凡（末行见屏蔽前缀列）")
    bool_padded = torch.ones(1, 1, q_len, kv_len, dtype=torch.bool)
    bool_padded[:, :, -1, -2:] = False   # bool 语义 True=可见
    _check(_has_nontrivial_padding(bool_padded) is True,
           "135: bool mask padding 应判非平凡")
    bool_causal = torch.ones(1, 1, q_len, kv_len, dtype=torch.bool)
    bool_causal[:, :, 0, 1:] = False     # 纯因果 bool（首行上三角屏蔽）
    _check(_has_nontrivial_padding(bool_causal) is False,
           "135: bool 纯因果不得误杀")

    # ---- wrapper 级绿：padded mask → RuntimeError（进入 sparse 分支前）----
    s = _fake_self(q_len, kv_len)
    (status, msg), _ = _call_wrapper(s, hs, attention_mask=padded,
                                     past_key_values=None)
    _check(status == "raise" and "TL-MINF-PADMASK-135" in msg,
           f"135: padded mask 应被显式拒收: {status} {msg}")

    # ---- wrapper 级红（审计缺陷复现）：守卫关闭 → padded mask 原样流入
    #      prefill_kwargs，mock kernel 正常返回 = 无人校验的静默错输出 ----
    with mock.patch.object(patch_mod, "_prefill_guard_violation",
                           return_value=None):
        (status2, _), captured = _call_wrapper(
            s, hs, attention_mask=padded, past_key_values=None)
        _check(status2 == "ok",
               f"135 红证: 守卫关闭时 padded mask 走完全程无报错: {status2}")
        _check(captured["attention_mask"] is not None and
               not torch.equal(captured["attention_mask"],
                               torch.zeros_like(padded)),
               "135 红证: padded mask 流入 prefill_kwargs 而官方 kernel "
               "不消费（审计静默错输出路径）")

    # ---- wrapper 级绿（不误杀）：mask=None / 纯因果 → 不因 135 报错 ----
    for m in (None, causal):
        (st, m2), cap = _call_wrapper(s, hs, attention_mask=m,
                                      past_key_values=None)
        _check(st == "ok", f"135: 合法 mask {type(m)} 不得被误杀: {st} {m2}")
    print("B3 PASS  padding mask fail-closed（红证：守卫关闭时 padded mask "
          "静默流入 kernel；绿证：显式 RuntimeError + 纯因果/None 不误杀）")


def test_136_chunkcache():
    """136：红（q_len>1+prefix cache 静默走稀疏 prefill）/ 绿（显式拒收）。"""
    q_len = 5
    hs = torch.zeros(1, q_len, 8)

    # ---- 单元级：_prefill_guard_violation ----
    v = _prefill_guard_violation(q_len, _FakeCache(3), None)
    _check(v is not None and v[0] == "TL-MINF-CHUNKCACHE-136",
           f"136: cache=3 + q_len=5 应判违例: {v}")
    _check(_prefill_guard_violation(q_len, _FakeCache(0), None) is None,
           "136: 首段全量 prefill（cache=0）不得误杀")
    _check(_prefill_guard_violation(1, _FakeCache(3), None) is None,
           "136: decode（q_len=1）不得误杀")
    _check(_prefill_guard_violation(q_len, None, None) is None,
           "136: 无 cache 不得误杀")

    # ---- wrapper 级绿：q_len>1 + 更新前 cache 已有 token → RuntimeError
    #      （守卫在 wrapper 内位于 cache.update 调用之前，用更新前
    #      get_seq_length() 判定——见 _prefill_guard_violation docstring）----
    s = _fake_self(q_len, q_len)
    (status, msg), _ = _call_wrapper(s, hs, attention_mask=None,
                                     past_key_values=_FakeCache(3))
    _check(status == "raise" and "TL-MINF-CHUNKCACHE-136" in msg and
           "方阵" in msg,
           f"136: chunked prefill 应被显式拒收（中文说明方阵协议）: "
           f"{status} {msg}")

    # ---- wrapper 级红（审计缺陷复现）：守卫关闭 → q_len=5 + cache=3 直接
    #      走稀疏 prefill 分支（mock kernel 收到 q_len=5、cache 无人校验） ----
    with mock.patch.object(patch_mod, "_prefill_guard_violation",
                           return_value=None):
        (status2, _), captured = _call_wrapper(
            s, hs, attention_mask=None, past_key_values=_FakeCache(3))
        _check(status2 == "ok" and captured["q_len"] == 5,
               ("136 红证: 守卫关闭时 q_len=5+cache=3 静默走稀疏 prefill"
                f"（Q_len≠K_len 方阵协议错位）: {status2} {captured}"))

    # ---- wrapper 级绿（不误杀）：首段全量 prefill（cache=0）正常走完 ----
    (st, m2), cap = _call_wrapper(s, hs, attention_mask=None,
                                  past_key_values=_FakeCache(0))
    _check(st == "ok" and cap["q_len"] == 5,
           f"136: 首段全量 prefill 不得误杀: {st} {m2} {cap}")

    # ---- decode 分支（q_len=1）不触守卫：flash_forward 直通 ----
    (st, _), _ = _call_wrapper(s, torch.zeros(1, 1, 8), attention_mask=None,
                               past_key_values=_FakeCache(3))
    _check(st == "ok", "136: decode 单 token 不得误杀")
    print("B4 PASS  chunked prefill with prefix cache fail-closed（红证：守"
          "卫关闭时静默走稀疏分支；绿证：更新前 seq_length 判定显式 "
          "RuntimeError + 首段 prefill/decode 不误杀）")


def main():
    test_135_padmask()
    test_136_chunkcache()
    print("\nGPT-B-BATCH-MINFERENCE(135/136) RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
