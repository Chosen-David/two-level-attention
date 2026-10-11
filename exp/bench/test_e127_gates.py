#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E127 优化门禁套件（每项优化一个 test_*，违反即回退）。

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_e127_gates.py            # 全部
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_e127_gates.py metrics eager

门禁口径：
  - 值逐位一致的门禁用 ==（不是 allclose）
  - attention 输出对拍用 allclose(1e-2/1e-2)（bf16 合理容差，任务指定）
  - 惰性/无同步路径用 torch.cuda.set_sync_debug_mode("error") 验证零 host 同步
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


# ---------------------------------------------------------------- metrics 惰性化
def test_metrics_lazy():
    """iter1：add_select_result 惰性路径 —— get_select_tokens 逐位一致 + 零同步。"""
    from sparse_attn.metrics import Metrics
    torch.manual_seed(20261011)
    dev = torch.device('cuda:0')
    # 构造覆盖：低于阈值（不记录）/ 高于阈值（记录）/ 多行混合 / 不同 H
    calls = []
    for S, H in [(100, 8), (3000, 8), (5000, 32), (8192, 8)]:
        q_ids = torch.tensor([S - 1], device=dev)
        mask = torch.rand(1, 1, H, S, device=dev) > 0.9
        calls.append((q_ids, mask, 1))
    # 多行混合：第 0 行低于阈值，第 1 行高于
    q_ids2 = torch.tensor([100, 9000], device=dev)
    mask2 = torch.rand(1, 2, 8, 9000, device=dev) > 0.85
    calls.append((q_ids2, mask2, 64))

    results = {}
    for mode in ["0", "1"]:
        os.environ["TLI_E127_FAST"] = mode
        m = Metrics()
        for q_ids, mask, bs in calls:
            m.add_select_result(q_ids, mask, bs)
        results[mode] = m.get_select_tokens()
    check("metrics_lazy: get_select_tokens 逐位一致",
          results["0"] == results["1"] and results["0"] is not None,
          f"old={results['0']} lazy={results['1']}")

    # 零同步验证：error 模式下惰性路径不得触发 host 同步
    os.environ["TLI_E127_FAST"] = "1"
    m = Metrics()
    q_ids = torch.tensor([4096], device=dev)
    mask = torch.rand(1, 1, 8, 4096, device=dev) > 0.9
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(20):
            m.add_select_result(q_ids, mask, 1)
        no_sync = True
    except Exception as e:
        no_sync = False
        print(f"  同步异常: {type(e).__name__}: {e}")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    check("metrics_lazy: add_select_result 零 host 同步", no_sync)
    os.environ.pop("TLI_E127_FAST", None)


# ---------------------------------------------------------------- eager gather
def test_eager_gather():
    """iter2+3：eager_decoding_attn gather 路径 —— allclose(1e-2) 对拍 + 语义守卫。

    输入用与生产一致的形态：mask [1,1,Hkv,S]（共享口径）与 [1,1,H,S]
    （per_q_head 口径）都要过；全 False 行的 B7 fail-closed 守卫两条路径
    都必须触发同一 RuntimeError。
    【iter3】gather 路径零同步化：B7/W 超界/cu 契约违规改为 GPU 标志 +
    生成结束 eager_decoding_check_errors() 尾部 raise（文案/语义守恒，
    时机后移）；小序列路由（T<16K）仍当场走 dense raise。
    """
    from sparse_attn.ops.eager_decoding import (
        eager_decoding_attn, eager_decoding_check_errors)
    torch.manual_seed(20261012)
    dev = torch.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    ok, worst = True, 0.0
    for S in [4097, 8192, 16384, 65536 - 13]:
        for mask_kind in ["shared", "per_qh"]:
            k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
            q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
            sel = torch.rand(1, 1, Hkv if mask_kind == "shared" else H,
                             S, device=dev) > 0.98
            sel[..., -128:] = True   # 保证每行非空（B7 守卫不触发）
            cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
            os.environ["TLI_E127_FAST"] = "0"
            o_ref = eager_decoding_attn(q, k, v, sel, 1, cu)
            os.environ["TLI_E127_FAST"] = "1"
            o_fast = eager_decoding_attn(q, k, v, sel, 1, cu)
            os.environ.pop("TLI_E127_FAST", None)
            eager_decoding_check_errors(dev)   # 正常路径：不 raise + 清零
            close = torch.allclose(o_ref.float(), o_fast.float(), atol=1e-2, rtol=1e-2)
            d = (o_ref.float() - o_fast.float()).abs().max().item()
            worst = max(worst, d)
            ok = ok and close
            if not close:
                print(f"  MISMATCH S={S} {mask_kind} maxdiff={d}")
    check("eager_gather: allclose(1e-2) 全配置对拍", ok, f"worst_abs_diff={worst:.2e}")

    # B7 fail-closed 守卫：小序列（T<16K 路由 dense）双模式当场 raise
    for mask_kind in ["shared", "per_qh"]:
        S, rows = 4096, (Hkv if mask_kind == "shared" else H)
        k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
        bad = torch.zeros(1, 1, rows, S, dtype=torch.bool, device=dev)
        cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
        raised = {}
        for mode in ["0", "1"]:
            os.environ["TLI_E127_FAST"] = mode
            try:
                eager_decoding_attn(q, k, v, bad, 1, cu)
                raised[mode] = False
            except RuntimeError:
                raised[mode] = True
        os.environ.pop("TLI_E127_FAST", None)
        check(f"eager_gather: B7 全 False 行守卫({mask_kind}) 小序列双路径当场 raise",
              raised["0"] and raised["1"])

    # 【iter3】B7 大序列：gather 主路径不再当场 raise（零同步），改为
    # eager_decoding_check_errors 尾部 raise（文案同 B7）。
    S = 65536 - 13
    k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
    bad = torch.zeros(1, 1, Hkv, S, dtype=torch.bool, device=dev)
    cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
    os.environ["TLI_E127_FAST"] = "1"
    o = eager_decoding_attn(q, k, v, bad, 1, cu)      # 不当场 raise
    raised_tail = False
    try:
        eager_decoding_check_errors(dev)
    except RuntimeError as e:
        raised_tail = "全 False 行" in str(e)
    check("eager_gather: B7 大序列 gather 尾部 raise（零同步语义）", raised_tail)

    # 【iter3】W 超上界（漏选保护）：全 True mask counts=S > W_upper → code=2
    o = eager_decoding_attn(q, k, v, torch.ones_like(bad), 1, cu)
    raised_w = False
    try:
        eager_decoding_check_errors(dev)
    except RuntimeError as e:
        raised_w = "TLI_E127_W_UPPER" in str(e)
    check("eager_gather: W 超上界 fail-closed 尾部 raise", raised_w)

    # 【iter3】cu 契约破坏 → code=3
    cu_bad = torch.tensor([5, S], dtype=torch.int32, device=dev)
    sel = torch.rand(1, 1, Hkv, S, device=dev) > 0.98
    sel[..., -128:] = True
    o = eager_decoding_attn(q, k, v, sel, 1, cu_bad)
    raised_cu = False
    try:
        eager_decoding_check_errors(dev)
    except RuntimeError as e:
        raised_cu = "cu_seqlens_k 契约破坏" in str(e)
    check("eager_gather: cu 契约破坏 fail-closed 尾部 raise", raised_cu)

    # 【iter3】零 host 同步验证：error 模式下 gather 主路径全程无同步
    sel2 = torch.rand(1, 1, H, S, device=dev) > 0.98
    sel2[..., -128:] = True
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(20):
            eager_decoding_attn(q, k, v, sel, 1, cu)
            eager_decoding_attn(q, k, v, sel2, 1, cu)
        no_sync = True
    except Exception as e:
        no_sync = False
        print(f"  同步异常: {type(e).__name__}: {e}")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    eager_decoding_check_errors(dev)
    check("eager_gather: gather 主路径零 host 同步（40 次调用）", no_sync)
    os.environ.pop("TLI_E127_FAST", None)


if __name__ == '__main__':
    selected = sys.argv[1:] or ["metrics", "eager"]
    if "metrics" in selected:
        test_metrics_lazy()
    if "eager" in selected:
        test_eager_gather()
    print(f"\n=== {len(PASS)} PASS / {len(FAIL)} FAIL ===")
    for f in FAIL:
        print(f"FAILED: {f}")
    sys.exit(1 if FAIL else 0)
