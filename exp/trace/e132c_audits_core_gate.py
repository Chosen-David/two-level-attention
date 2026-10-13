"""E132c A 批红绿测试：GPT 四份增量审计 TLI 核心代码门禁（6+1 项）。

覆盖审计项（修复 = fail-closed 显式拒绝；红证 = 修复前 SHA 上 FAIL）：
  A1  TL-E134-PROJ-L2FULL-137   proj_basis × l2_subspace=full 组合未实现 → 构造期 ValueError
  A2  TL-E135-OPTIMIZED-GATE-144 sink/swa 边界 assert → 显式 ValueError（python -O 下 assert 被剥除）
  A3  TL-E135-PROTECTED-BUDGET-143 sink/swa 非块对齐 → ValueError（保护区不占预算契约）
  A4  TL-E137-PARAM-VALIDATION-147 dyn_dgate_rows/eps 边界：arguments.py type 校验 + 构造函数双处校验
  A5  TL-E137-LLAMA-NOOP-146     LlamaAttention + --tli_dyn_dgate → 注册期显式拒绝
  A6  TL-E137-PREFILL-GATE-145   decode-frozen 生命周期分离：calibrate 只写 pending，
                                 decode 首步冻结；enable_layer_skip × dyn_dgate 同开 → ValueError

用法（普通与 -O 双跑，两者结果一致才算绿——A2 红证专项依赖 -O 剥 assert）：
  python3 exp/trace/e132c_audits_core_gate.py
  python3 -O exp/trace/e132c_audits_core_gate.py

不依赖 GPU / 不加载真模型（A5 用 dummy 类替换 patch.LlamaAttention 全局引用，
绕开 transformers 权重构造）。不挂任何大套件。
"""
import argparse
import os
import sys
import tempfile
import types

# 仓库根 = 脚本位于 <repo>/exp/trace/ 上两级
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import torch  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class Failure(Exception):
    pass


def check(cond, msg):
    """显式失败抛异常（不用 assert——assert 在 python -O 下会被剥除，本脚本
    自身也必须 -O 双跑一致）。"""
    if not cond:
        raise Failure(msg)


def make_args(**over):
    """构造可直接实例化 TLIIndexer 的最小 args namespace（CPU 假参数，
    不触发任何 tensor/GPU 分配）。enable_layer_skip 显式 False——绕开
    D' 掩码文件 IO；生产动态 gate 臂同款口径（--tli_enable_layer_skip false）。"""
    base = dict(
        method="tli",
        tia_block_size=64,
        tia_level1_topk=16,
        tia_level2_topk=64,
        tia_level2_cmp_ratio=2,
        tia_enable_async_topk=False,
        tli_subspace="tail",
        tli_enable_layer_skip=False,
        tli_sink_tokens=128,
        tli_swa_size=128,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def expect_value_error(fn, needle, label):
    """期待 ValueError 且消息含 needle；其他异常/无异常均 FAIL。"""
    try:
        fn()
    except ValueError as e:
        check(needle in str(e),
              f"{label}: ValueError 消息应含 {needle!r}，实际 {str(e)!r}")
        return
    raise Failure(f"{label}: 期待 ValueError 未抛出")


# ====================================================================== #
# A1：proj_basis × l2_subspace=full → 构造期 ValueError
# ====================================================================== #

def t_a1():
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    with tempfile.TemporaryDirectory() as td:
        # 假投影基 [n_layers, Hkv, 128, r]（裸 tensor，与 CLI 声明格式一致）
        bp = os.path.join(td, "fake_basis.pt")
        torch.save(torch.randn(2, 2, 128, 4), bp)
        args = make_args(tli_proj_basis=bp, tli_l2_subspace="full")
        expect_value_error(
            lambda: TLIIndexer(args),
            "未实现",
            "A1 proj_basis×l2_subspace=full 应构造期拒绝")
        # 对照：l2_subspace=same 时该组合合法（投影路径既有口径，不被误伤）
        TLIIndexer(make_args(tli_proj_basis=bp, tli_l2_subspace="same"))


# ====================================================================== #
# A2：sink/swa 边界 assert → 显式 ValueError（-O 下仍生效）
# ====================================================================== #

def t_a2():
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    expect_value_error(
        lambda: TLIIndexer(make_args(tli_sink_tokens=-1)),
        "sink_tokens", "A2 sink_tokens=-1 应 ValueError")
    expect_value_error(
        lambda: TLIIndexer(make_args(tli_swa_size=0)),
        "swa", "A2 swa_size=0 应 ValueError")
    # 合法边界不受影响
    TLIIndexer(make_args(tli_sink_tokens=0))    # 0 合法（无 sink 强制区）


# ====================================================================== #
# A3：sink/swa 非块对齐 → ValueError；128/64 及倍数过
# ====================================================================== #

def t_a3():
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    expect_value_error(
        lambda: TLIIndexer(make_args(tli_sink_tokens=16)),
        "整数倍", "A3 sink=16（非 64 倍数）应 ValueError")
    expect_value_error(
        lambda: TLIIndexer(make_args(tli_swa_size=96)),
        "整数倍", "A3 swa=96（非 64 倍数）应 ValueError")
    # 历史默认与 64 倍数集合不受影响
    TLIIndexer(make_args(tli_sink_tokens=128, tli_swa_size=128))
    TLIIndexer(make_args(tli_sink_tokens=64, tli_swa_size=192))
    TLIIndexer(make_args(tli_sink_tokens=0, tli_swa_size=64))


# ====================================================================== #
# A4：dyn_dgate_rows/eps 边界——argparse 层 + 构造函数层双处校验
# ====================================================================== #

def _tli_parser():
    from sparse_attn.arguments import add_sparse_attn_args
    p = argparse.ArgumentParser(allow_abbrev=False)
    add_sparse_attn_args(p)
    return p


def expect_arg_reject(p, argv, label):
    try:
        p.parse_args(argv)
    except SystemExit as e:
        check(e.code not in (None, 0), f"{label}: argparse 拒绝应非零退出码")
        return
    raise Failure(f"{label}: argparse 应拒绝 {argv} 未拒绝")


def t_a4():
    from sparse_attn.arguments import add_sparse_attn_args
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    # ---- argparse 层（CLI 入口）----
    for bad_rows in ("0", "-1"):
        expect_arg_reject(
            _tli_parser(), ["--method", "tli", "--tli_dyn_dgate_rows", bad_rows],
            f"A4 rows={bad_rows}")
    for bad_eps in ("nan", "inf", "-0.1", "1.1"):
        expect_arg_reject(
            _tli_parser(), ["--method", "tli", "--tli_dyn_dgate_eps", bad_eps],
            f"A4 eps={bad_eps}")
    # 合法边界：rows=1 / eps=0 / eps=1
    ns = _tli_parser().parse_args(
        ["--method", "tli", "--tli_dyn_dgate_rows", "1",
         "--tli_dyn_dgate_eps", "0.0"])
    check(ns.tli_dyn_dgate_rows == 1 and ns.tli_dyn_dgate_eps == 0.0,
          "A4 合法边界 rows=1/eps=0 应被接受")
    ns = _tli_parser().parse_args(["--method", "tli", "--tli_dyn_dgate_eps", "1.0"])
    check(ns.tli_dyn_dgate_eps == 1.0, "A4 合法边界 eps=1 应被接受")
    # ---- 构造函数层（getattr 兜底路径：非 CLI 入口直接传 namespace）----
    for bad_rows in (0, -1):
        expect_value_error(
            lambda r=bad_rows: TLIIndexer(
                make_args(tli_dyn_dgate=True, tli_dyn_dgate_rows=r)),
            "rows", f"A4 构造层 rows={bad_rows} 应 ValueError")
    for bad_eps in (float("nan"), float("inf"), -0.1, 1.1):
        expect_value_error(
            lambda e=bad_eps: TLIIndexer(
                make_args(tli_dyn_dgate=True, tli_dyn_dgate_eps=e)),
            "eps", f"A4 构造层 eps={bad_eps} 应 ValueError")
    # 合法边界构造通过
    TLIIndexer(make_args(tli_dyn_dgate=True, tli_dyn_dgate_rows=1,
                         tli_dyn_dgate_eps=0.0))
    TLIIndexer(make_args(tli_dyn_dgate=True, tli_dyn_dgate_eps=1.0))


# ====================================================================== #
# A5：LlamaAttention + dyn_dgate → 注册期显式拒绝（mock，不加载真模型）
# ====================================================================== #

class _DummyLlamaModule(torch.nn.Module):
    """占位模块——通过替换 patch.LlamaAttention 全局引用匹配 isinstance，
    不构造 transformers 真权重。"""


def t_a5():
    import torch.nn as nn
    from sparse_attn.patches import patch as patch_mod

    model = nn.Module()
    model.add_module("layers_0_self_attn", _DummyLlamaModule())
    saved = patch_mod.LlamaAttention
    patch_mod.LlamaAttention = _DummyLlamaModule   # 全局引用替换 → isinstance 命中
    try:
        args = make_args(tli_dyn_dgate=True)
        try:
            patch_mod.register_patch(model, args)
        except RuntimeError as e:
            check("E137" in str(e) and "Llama" in str(e),
                  f"A5 拒绝消息应含 'E137'/'Llama'，实际 {str(e)!r}")
        else:
            raise Failure("A5 Llama×dyn_dgate 注册应显式拒绝（静默 no-op 会产出"
                          "带动态 gate 标签的 dense 语义结果）")
        # 反门：dyn_dgate=False 时不被 A5 误伤——dummy 模块按既有逻辑
        # 正常 patch（返回计数 1），证明 A5 门只在动态 gate 声明时触发
        n = patch_mod.register_patch(model, make_args(tli_dyn_dgate=False))
        check(n == 1, f"A5 反门：dyn_dgate=False 应正常 patch（n=1），实际 n={n}")
    finally:
        patch_mod.LlamaAttention = saved


# ====================================================================== #
# A6：decode-frozen 生命周期分离
# ====================================================================== #

def t_a6():
    from sparse_attn.indexer.tli_indexer import TLIIndexer

    # 组合语义 fail-closed：enable_layer_skip × dyn_dgate 同开 → ValueError
    expect_value_error(
        lambda: TLIIndexer(make_args(tli_dyn_dgate=True, tli_enable_layer_skip=True)),
        "互斥", "A6 静态 D'×动态 D' 同开应构造期 ValueError")

    # 生命周期分离主链（eps=1.0 → r_l<eps 恒真 → pending 必置位）
    idx = TLIIndexer(make_args(tli_dyn_dgate=True, tli_dyn_dgate_eps=1.0))
    idx.layer_idx = 0
    idx.clear()
    check(idx.skip_far is False and idx._dyn_skip_pending is False
          and idx._dyn_frozen is False, "A6 clear() 后三状态应全复位")

    # 真实 calibrate（CPU 小张量）：决策只落 pending，skip_far 仍 False
    torch.manual_seed(20261013)
    q = torch.randn(1, 4, 512, 128)   # [B,H,S,D] post-RoPE 布局
    k = torch.randn(1, 2, 512, 128)   # [B,Hkv,S,D]
    idx.calibrate_dyn_dgate(q, k)
    check(idx._dyn_skip_pending is True,
          "A6 eps=1.0 下 calibrate 应置 _dyn_skip_pending=True")
    check(idx.skip_far is False,
          "A6 calibrate 后 skip_far 应仍为 False（prefill 期决策不生效——"
          "否则稀疏 prefill chunk 会消费完整 prompt 标定的非因果路由）")

    # decode 首步冻结生效 + 哨兵防重复
    freeze = getattr(idx, "freeze_dyn_dgate", None)
    check(freeze is not None and callable(freeze), "A6 freeze_dyn_dgate 未实现")
    freeze()
    check(idx.skip_far is True, "A6 decode 首步冻结后 skip_far 应生效")
    freeze()
    check(idx.skip_far is True and idx._dyn_frozen is True,
          "A6 重复 freeze 应幂等（哨兵防重复）")

    # clear() 复位后可重新标定冻结（模拟 calibrate 设 pending 的最小路径）
    idx.clear()
    check(idx.skip_far is False and idx._dyn_skip_pending is False
          and idx._dyn_frozen is False, "A6 二次 clear() 应复位三状态")
    idx._dyn_skip_pending = True
    freeze()
    check(idx.skip_far is True, "A6 新请求 pending → 冻结链应可重复")

    # 不触发路径：eps=0.0 → r_l<0 恒假 → pending False → 冻结后 skip_far False
    idx2 = TLIIndexer(make_args(tli_dyn_dgate=True, tli_dyn_dgate_eps=0.0))
    idx2.layer_idx = 1
    idx2.calibrate_dyn_dgate(q, k)
    check(idx2._dyn_skip_pending is False, "A6 eps=0.0 下 pending 应为 False")
    idx2.freeze_dyn_dgate()
    check(idx2.skip_far is False, "A6 eps=0.0 冻结后 skip_far 应为 False")


# ====================================================================== #

def main():
    tests = [
        ("A1", t_a1, "proj_basis × l2_subspace=full 组合 fail-closed"),
        ("A2", t_a2, "sink/swa 边界 assert → ValueError"),
        ("A3", t_a3, "sink/swa 块对齐校验"),
        ("A4", t_a4, "dyn_dgate rows/eps 边界双处校验"),
        ("A5", t_a5, "Llama×dyn_dgate 注册拒绝"),
        ("A6", t_a6, "decode-frozen 生命周期分离"),
    ]
    py_opt = "ON" if __debug__ is False else "OFF"
    print(f"== E132c A 批核心门禁红绿测试（python -O={py_opt}，repo={_REPO}）==")
    n_pass = n_fail = 0
    for name, fn, desc in tests:
        try:
            fn()
            print(f"  [PASS] {name}  {desc}")
            n_pass += 1
        except Failure as e:
            print(f"  [FAIL] {name}  {desc}: {e}")
            n_fail += 1
        except Exception as e:   # 非预期异常同计 FAIL（红证含崩溃形态）
            print(f"  [FAIL] {name}  {desc}: 非预期异常 {type(e).__name__}: {e}")
            n_fail += 1
    print(f"== 结果: {n_pass} PASS / {n_fail} FAIL ==")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
