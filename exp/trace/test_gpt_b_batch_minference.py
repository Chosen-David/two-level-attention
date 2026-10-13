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

TL-MINF-CHUNKCACHE-136 跨层误杀（C 批追加，GPT advice 2026-10-13_1446）
  缺陷：guard 无参调用 get_seq_length()（transformers 4.56 默认
      layer_idx=0）——Qwen3 逐层循环共享同一 DynamicCache，首次完整
      prefill 时 layer0 放行并写入后 layer1 进守卫读到 layer0 的非零
      长度 → layer>=1 全误杀（B4 原验收 FakeCache 忽略层号为盲点）。
  绿：guard 增参 layer_idx（wrapper 传 self.layer_idx），136 判定与
      报错读数均读本层 get_seq_length(layer_idx) 更新前长度；共享 cache
      首次 prefill layer0→layer1 均放行、本层 prefix chunked prefill
      照旧拒收、decode 直通不变。

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
    """get_seq_length/update 语义对齐 HF DynamicCache（GPT advice
    2026-10-13_1446 跨层误杀修复后）：**按层存储**——get_seq_length
    (layer_idx) 只读本层长度（无参调用默认 layer_idx=0，同 HF 4.56
    Cache API）；update() 真正把本层新 token append 进本层计数
    （HF 逐层循环共享同一 cache：layer0 prefill 写入后 layer1 更新前
    本层仍为 0）。构造参数 n = 所有层的初始缓存长度（prefix cache
    全层就绪形态；n=0 即首段完整 prefill 起点）。"""

    def __init__(self, n=0):
        self._base = n
        self.per_layer = {}

    def get_seq_length(self, layer_idx=0):
        # 本层未写过 → 构造基值；写过 → update 追加后的本层真实长度
        return self.per_layer.get(layer_idx, self._base)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states [bsz, heads, q_len, head_dim]（wrapper 已 transpose）
        self.per_layer[layer_idx] = (self.get_seq_length(layer_idx) +
                                     key_states.shape[2])
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


def _fake_self(q_len, kv_len, layer_idx=1):
    """构造能走完稀疏 prefill 分支的最小 Qwen3Attention 替身（CPU）。
    layer_idx 可参数化（跨层三态测试需构造 layer0/layer1/layer2 替身
    共享同一 cache，模拟 Qwen3 逐层解码循环）。"""
    bsz, heads, head_dim = 1, 2, 8
    s = types.SimpleNamespace()
    s.layer_idx = layer_idx
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


def test_136_crosslayer_shared_cache():
    """136 跨层误杀回归（GPT advice 2026-10-13_1446，C 批追加项）。

    缺陷：guard 无参调用 past_key_values.get_seq_length()——transformers
    4.56 Cache API 默认 layer_idx=0。Qwen3 逐层循环共享同一 DynamicCache：
    首次完整 prefill 时 layer0 检查放行、其 attention 内部 update() 写入
    后，layer1 进守卫读到 layer0 的非零长度 → 误抛 TL-MINF-CHUNKCACHE-136
    （layer>=1 全误杀）。B4 原验收的 FakeCache 忽略层号 + update 不改状态，
    测试盲点。

    三态验证（wrapper 级，FakeCache 按 HF 语义按层存储 + 真 update）：
      ① 共享 cache 首次 prefill：layer0 放行并写入后，layer1 进守卫
         （q_len>1）→ 必须放行（红证点：基线上无参 get_seq_length 默认
         读 layer0 而误拒）；
      ② 当前层已有 prefix（chunked prefill with prefix cache）且 q_len>1
         → 在 update 之前拒绝（既有 fail-closed 语义保留，报错读数 =
         本层更新前长度）；
      ③ 单 token decode 直通不变。
    """
    q_len = 5
    hs = torch.zeros(1, q_len, 8)

    # ---- ① 共享 cache 首次完整 prefill：layer0 → layer1 都必须放行 ----
    cache = _FakeCache(0)          # 首段全量 prefill 起点：全层皆空
    s0 = _fake_self(q_len, q_len, layer_idx=0)
    s1 = _fake_self(q_len, q_len, layer_idx=1)
    (st0, m0), _ = _call_wrapper(s0, hs, attention_mask=None,
                                 past_key_values=cache)
    _check(st0 == "ok",
           f"136 跨层①: layer0 首段 prefill 应放行: {st0} {m0}")
    _check(cache.get_seq_length(0) == q_len,
           "136 跨层①: layer0 update 应真正写入本层（HF 逐层共享 cache）")
    _check(cache.get_seq_length(1) == 0,
           "136 跨层①: layer1 更新前本层应为 0（其他层已写入不构成本层"
           " chunked 判定依据）")
    # 红证点：基线 guard 无参 get_seq_length()（默认 layer_idx=0）读到
    # layer0 已写入的 5 → layer1 首次 prefill 被跨层误拒。
    (st1, m1), _ = _call_wrapper(s1, hs, attention_mask=None,
                                 past_key_values=cache)
    _check(st1 == "ok",
           ("136 跨层①红证点: 共享 cache 下 layer1 首段 prefill 被误拒"
            f"（无参 get_seq_length 默认 layer_idx=0，读到 layer0 已写入"
            f"长度）——guard 须按本层 layer_idx 读更新前长度: {st1} {m1}"))
    _check(cache.get_seq_length(1) == q_len,
           "136 跨层①: layer1 放行后 update 应写入本层")

    # ---- ② 当前层已有 prefix 且 q_len>1 → update 前拒绝（同层读数）----
    hs_chunk = torch.zeros(1, 3, 8)   # 新 chunk q_len=3（layer1 已缓存 5）
    (st2, m2), _ = _call_wrapper(s1, hs_chunk, attention_mask=None,
                                 past_key_values=cache)
    _check(st2 == "raise" and "TL-MINF-CHUNKCACHE-136" in m2 and "方阵" in m2,
           f"136 跨层②: 本层已有 prefix 的 chunked prefill 应显式拒收: "
           f"{st2} {m2}")
    _check("第 1 层" in m2 and "5 个 token" in m2,
           f"136 跨层②: 报错读数须为本层（layer1）更新前长度 5: {m2}")
    _check(cache.get_seq_length(1) == q_len,
           "136 跨层②: 拒绝须发生在 update 之前（本层长度不得被改写）")

    # ---- ③ 单 token decode 直通不变（q_len=1 不触守卫）----
    (st3, _), _ = _call_wrapper(s1, torch.zeros(1, 1, 8), attention_mask=None,
                                past_key_values=cache)
    _check(st3 == "ok", "136 跨层③: decode 单 token 不得误杀")
    print("B4+ PASS  136 跨层误杀回归（advice 2026-10-13_1446）：共享 "
          "cache 首次 prefill layer0→layer1 均放行（红证点：基线无参 "
          "get_seq_length 默认 layer_idx=0 跨层误拒）；本层 prefix "
          "chunked prefill 在 update 前拒收且读数=本层更新前长度；decode "
          "单 token 直通不变")


def main():
    test_135_padmask()
    test_136_chunkcache()
    test_136_crosslayer_shared_cache()
    print("\nGPT-B-BATCH-MINFERENCE(135/136) RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
