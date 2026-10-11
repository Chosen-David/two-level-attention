#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E126 红绿测试：计时插桩不影响数值 + anchor 身份门。

  T1 (CPU)  anchor 硬校验过 + 插桩构建成功 + 生产文件字节零改动（md5 冻结锚点）
  T2 (GPU)  计时开/关同输入逐位一致：每臂 timing-off 两跑 A1/A2 + timing-on 一跑 B，
            mask 与 attention 输出 md5 逐位相等（A1≠A2 时降 delta-delta 判定并报告）
  T3 (GPU)  mark 序列 == 各臂期望序列；span 全部非负；span 总和 ≤ 每步 wall×2+2ms
  T4 (GPU)  TLI_STAGE_TIMING=0 → 全程零事件入列表

运行：CUDA_VISIBLE_DEVICES=<gpu> python3 exp/trace/test_e126_stage_timing.py
（无 GPU 时 T2/T3/T4 SKIP，rc=0 只覆盖 T1；全部显式检查不依赖 assert 语义，
 python -O 下同样生效——045 口径）
"""
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

FAILURES = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {msg}", flush=True)
    if not cond:
        FAILURES.append(msg)


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "e126_stage_timing", os.path.join(HERE, "e126_stage_timing.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["e126_stage_timing"] = m
    spec.loader.exec_module(m)
    return m


def main():
    torch_ok = True
    try:
        import torch
        have_cuda = torch.cuda.is_available()
    except Exception:
        torch_ok = False
        have_cuda = False
    if not torch_ok:
        print("[SKIP] 无 torch 环境", flush=True)
        return 0

    H = load_harness()

    # ---------------- T1（CPU）：anchor + 身份 + 生产文件不动 ----------------
    tli_path = os.path.join(ROOT, "sparse_attn", "indexer", "tli_indexer.py")
    src = open(tli_path, encoding="utf-8").read()
    try:
        inst = H.build_instrumented_source(src, tli_path)
        check(True, "T1a anchor 硬校验全部通过，插桩源构建成功")
    except SystemExit as e:
        check(False, f"T1a anchor 失配 fail-closed：{e}")
        return 1
    n_marks = inst.count("self._e126_mark(")
    check(n_marks >= len(H.ANCHORS) + 3, f"T1b 插桩 mark 数={n_marks} ≥ anchors+returns")
    md5_after = H.file_md5(tli_path)
    check(md5_after == H.EXPECTED_TLI_MD5,
          f"T1c 生产文件未改动 md5={md5_after} == 冻结锚点 {H.EXPECTED_TLI_MD5}")
    check("self._e126_mark" not in src, "T1d 生产源文本零污染（无任何 mark 注入）")

    if not have_cuda:
        print("[SKIP] 无 CUDA——T2/T3/T4 跳过（T1 已覆盖 CPU 侧身份门）", flush=True)
        return 1 if FAILURES else 0

    # ---------------- GPU 侧：加载插桩模块 + eager ----------------
    os.environ["TLI_STAGE_TIMING"] = "1"
    mod = H.load_instrumented(ROOT)
    import importlib
    eager_fn = importlib.import_module(
        "sparse_attn_e126.ops.eager_decoding").eager_decoding_attn
    torch.set_grad_enabled(False)
    device = "cuda"

    # ---------------- T4：TLI_STAGE_TIMING=0 → 零事件 ----------------
    os.environ["TLI_STAGE_TIMING"] = "0"
    mod4 = H.load_instrumented(ROOT, pkg_tag="sparse_attn_e126_t4")
    arm_cfg = H.ARM_CONFIGS["mavg"]
    k_all = H.gen_structured_kv(4096 + 8, 7, device)
    v_all = k_all.clone()
    mod4._E126_EVENTS.clear()
    H.run_sim(mod4, eager_fn, arm_cfg, k_all, v_all, 4096, n_steps=4,
              device=device, timing=False)
    check(len(mod4._E126_EVENTS) == 0, "T4 计时关（env=0）→ 全程零事件入列表")

    # ---------------- T2/T3：五臂 timing on/off 逐位一致 + 序列/span ----------------
    for arm, arm_cfg in H.ARM_CONFIGS.items():
        k_all = H.gen_structured_kv(4096 + 8, 7, device)
        v_all = k_all.clone()
        runs = {}
        spans_nonneg = True
        try:
            for tag_run, timing in (("A1", False), ("A2", False), ("B", True)):
                _, _, samples, _, gate, last = H.run_sim(
                    mod, eager_fn, arm_cfg, k_all, v_all, 4096, n_steps=8,
                    device=device, timing=timing, collect_gate=True)
                runs[tag_run] = (last["mask_md5"], last["out_md5"], gate, samples)
        except RuntimeError as e:
            check(False, f"T3[{arm}] run_sim 序列校验失配：{e}")
            del k_all, v_all
            torch.cuda.empty_cache()
            continue
        # T2：逐位一致
        m_a1, o_a1 = runs["A1"][0], runs["A1"][1]
        m_a2, o_a2 = runs["A2"][0], runs["A2"][1]
        m_b, o_b = runs["B"][0], runs["B"][1]
        if m_a1 == m_a2 and o_a1 == o_a2:
            # 基线确定 → 插桩必须逐位零扰动
            check(m_b == m_a1 and o_b == o_a1,
                  f"T2[{arm}] 基线确定 → 计时开/关 mask+out md5 逐位一致")
        else:
            # 基线本身非确定（原子加成序等）→ delta-delta：插桩扰动不得超过基线噪声
            check(m_b == m_a1 or m_b == m_a2,
                  f"T2[{arm}] 基线两跑已不一致（mask md5 A1≠A2），timing-on 结果须落在 "
                  f"基线噪声集合内（否则插桩引入额外扰动）")
        # T3：序列校验已由 run_sim 内部逐 step fail-fast 完成（失配即抛 RuntimeError
        # 传播出 run_sim，被下方 except 捕获）；此处再验 span 非负
        try:
            samples = runs["B"][3]
            for sp in samples:
                for name, v in sp.items():
                    if v is None or v < 0:
                        spans_nonneg = False
            exp_seq = H.expected_tag_sequence(arm_cfg)
            check(True, f"T3[{arm}] mark 序列 == 期望序列"
                        f"（{len(exp_seq)} 标记/步 × 8 步，run_sim 内逐 step 校验）")
        except RuntimeError as e:
            check(False, f"T3[{arm}] mark 序列失配：{e}")
        check(spans_nonneg, f"T3[{arm}] 全部阶段 span 非负")
        del k_all, v_all
        torch.cuda.empty_cache()

    # T3 补充：span 总和 ≤ wall（单独小跑，抓 wall 返回值）
    arm = "mavg"
    arm_cfg = H.ARM_CONFIGS[arm]
    k_all = H.gen_structured_kv(4096 + 8, 7, device)
    v_all = k_all.clone()
    wall_ns, s8, samples, _, _, _ = H.run_sim(
        mod, eager_fn, arm_cfg, k_all, v_all, 4096, n_steps=8, device=device,
        timing=True)
    spans_le_wall = True
    for i, sp in enumerate(samples):
        span_sum = sum(v for v in sp.values() if v is not None)
        if span_sum > wall_ns[i] / 1e6 * 2 + 2.0:
            spans_le_wall = False
    check(spans_le_wall, "T3 span 总和 ≤ 每步 wall×2+2ms（事件口径不虚涨）")

    print("\n==== E126 红绿测试总结 ====", flush=True)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} 项：", flush=True)
        for f in FAILURES:
            print("  - " + f, flush=True)
        return 1
    print("全部通过", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
