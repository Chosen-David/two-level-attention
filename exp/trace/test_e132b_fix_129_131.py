#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E132b 修复红绿测试：GPT 审计 1325/1133 四项缺陷（全部 CPU 可跑，不用 GPU）。

覆盖：
  - TL-E127-FEATURE-STRIDE-129：fine/avg fast path 门禁末维 stride 检查
    （CPU 上用 tensor 子类伪造 is_cuda + monkeypatch _fast_score_fine 哨兵，
    直接验证门禁布尔逻辑本身——红证：非连续末维输入仍进 fast 分支）
  - TL-E131S2-REPORT-REPRO-131：gen_report repo 相对 CLI + 源码身份实算 +
    不引用已删除 fast.py + results 逐位重放 + meta 透传
  - TL-E131-AGG-FAILOPEN-119：聚合前 fail-closed 校验（篡改 arm/S0/NaN/
    gate/sidecar/重复 run 必须退出非零；真数据聚合结果与已提交报告一致）
  - TL-E131S2-BARRIER-RESIDENCY-130：watchdog env 钩子框架 + limitation 声明

用法：python3 exp/trace/test_e132b_fix_129_131.py
  修复前运行 = 红证（gate/校验类断言失败）；修复后运行 = 全绿。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS_DIR = REPO / "exp" / "results_efficiency"
ARM_FILES = ["mavg", "aavg", "mminmax", "cavgsim", "cavgsim_steps80",
             "cclustersim", "mavg_e127"]

_results = []  # (name, ok, info)


def check(name, ok, info=""):
    _results.append((name, bool(ok), info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {info}" if info else ""))


# ======================================================================
# 129：fine/avg fast path 门禁末维 stride 检查
# ======================================================================
class _CudaFaker(torch.Tensor):
    """CPU 张量伪装 is_cuda=True（仅测试门禁布尔逻辑，不 launch kernel）。"""
    is_cuda = property(lambda self: True)


def _faker(t):
    return t.as_subclass(_CudaFaker)


def _mk_indexer(**over):
    """object.__new__ 骨架：只设 compute_score 本类路径触碰的属性
    （绕开父类捷径需 _need_avg_score()=True：far_method='avg'）。"""
    import sparse_attn.indexer.tli_indexer as ti
    idx = object.__new__(ti.TLIIndexer)
    idx.enable_subspace = False
    idx._basis = None
    idx.moba_gate = False
    idx.skip_far = False
    idx.alpha = 0.0
    idx.beta = 0.0
    idx.far_method = "avg"      # _need_avg_score() → True
    idx.near_method = "minmax"
    idx._triton_ok = True
    for k, v in over.items():
        setattr(idx, k, v)
    return idx


def _run_compute_score(idx, q, k_qat, k_avg=None):
    """调 compute_score，返回 (score_dict, fast_calls)。fast_calls 记录
    哨兵 _fast_score_fine 被调用时的 (kq.stride(-1), q.stride(-1))。"""
    T = k_qat.shape[1]
    Hkv = k_qat.shape[2]
    index_dict = {
        "k_min": torch.randn(1, T, Hkv, 128),
        "k_max": torch.randn(1, T, Hkv, 128),
        "k_qat": k_qat,
        "k_avg": k_avg,
        "valid_length": None,
    }
    calls = []

    def sentinel(kq, b_q, out_dtype):
        calls.append((kq.stride(-1), b_q.stride(-1)))
        return torch.zeros((b_q.shape[0], kq.shape[0]), dtype=out_dtype)

    idx._fast_score_fine = sentinel
    out = idx.compute_score(q, None, index_dict, 0.125)
    return out, calls


def _t129_gate_kqat_stride():
    """红证 1：k_qat 末维 stride!=1（permute 合法布局）仍进 fine fast 分支。"""
    import sparse_attn.indexer.tli_indexer as ti
    ti._HAS_TRITON = True
    T, Hkv, H, D = 16, 8, 32, 128
    idx = _mk_indexer()
    q = torch.randn(1, 1, H, D)
    # [1,T,128,Hkv] contiguous → permute(0,1,3,2)：[1,T,Hkv,128]，末维 stride=Hkv
    k_qat = _faker(torch.randn(1, T, 128, Hkv).permute(0, 1, 3, 2))
    assert k_qat.stride(-1) == Hkv != 1
    _, calls = _run_compute_score(idx, q, k_qat, k_avg=None)
    check("129/fine 门禁拒绝 k_qat 末维非连续", calls == [],
          f"fast 分支被非连续 k_qat 触发（stride(-1)={k_qat.stride(-1)}）" if calls else "")


def _t129_gate_bq_full_stride():
    """红证 2：b_q_full 末维 stride!=1（转置稠密 q）仍进 fine fast 分支。"""
    import sparse_attn.indexer.tli_indexer as ti
    ti._HAS_TRITON = True
    T, Hkv, H, D = 16, 8, 32, 128
    idx = _mk_indexer()
    # [D,H] contiguous 转置 → [H,D] stride=(1,H)：dense 非重叠合法布局，
    # mul + .to(fp32) 保留 stride（实测验证）
    q = torch.randn(D, H).t().unsqueeze(0).unsqueeze(0)
    b_q_full_probe = (q.squeeze(0)[0] * 0.125).to(torch.float32)
    assert b_q_full_probe.stride(-1) == H != 1
    k_qat = _faker(torch.randn(1, T, Hkv, 128))  # 连续，仅 q 非连续
    _, calls = _run_compute_score(idx, q, k_qat, k_avg=None)
    check("129/fine 门禁拒绝 b_q_full 末维非连续", calls == [],
          f"fast 分支被非连续 b_q_full 触发（stride(-1)={b_q_full_probe.stride(-1)}）" if calls else "")


def _t129_gate_avg_bq_stride():
    """红证 3：avg 段门禁缺 b_q 末维 stride 检查（k_avg 连续 + q 非连续）。"""
    import sparse_attn.indexer.tli_indexer as ti
    ti._HAS_TRITON = True
    T, Hkv, H, D = 16, 8, 32, 128
    idx = _mk_indexer()
    q = torch.randn(D, H).t().unsqueeze(0).unsqueeze(0)
    k_qat = torch.randn(1, T, Hkv, 128)          # 纯 CPU：fine 门禁自然关
    k_avg = _faker(torch.randn(1, T, Hkv, 128))   # 连续伪 cuda
    _, calls = _run_compute_score(idx, q, k_qat, k_avg=k_avg)
    check("129/avg 门禁拒绝 b_q 末维非连续", calls == [],
          f"avg fast 分支被非连续 b_q 触发" if calls else "")


def _t129_fallback_value_equiv():
    """绿证：门禁拒绝后回退原版 einsum，值与连续布局参考一致（fail-closed 值同）。"""
    import sparse_attn.indexer.tli_indexer as ti
    ti._HAS_TRITON = True
    T, Hkv, H, D = 16, 8, 32, 128
    torch.manual_seed(7)
    base = torch.randn(1, T, 128, Hkv)
    q = torch.randn(1, 1, H, D)
    # 参考：连续 CPU 布局（门禁自然关，原版 einsum）
    idx0 = _mk_indexer()
    ref, ref_calls = _run_compute_score(idx0, q, base.permute(0, 1, 3, 2).contiguous())
    # 非连续伪 cuda：门禁（修复后）拒绝 → 回退原版表达式
    idx1 = _mk_indexer()
    out, calls = _run_compute_score(idx1, q, _faker(base.permute(0, 1, 3, 2)))
    ok = (calls == []
          and torch.allclose(ref["score_fine"], out["score_fine"], atol=1e-5, rtol=1e-5)
          and torch.isfinite(out["score_fine"]).all())
    check("129/回退值同原版 einsum（fail-closed）", ok)


# ======================================================================
# 131：gen_report repo 相对 + 身份实算 + 透传 + 逐位重放
# ======================================================================
GEN = REPO / "exp" / "bench" / "e131s2_gen_report.py"


def _run_gen(tmp, out_json, out_md):
    return subprocess.run(
        [sys.executable, str(GEN), "--repo", str(REPO),
         "--out-json", str(out_json), "--out-md", str(out_md)],
        capture_output=True, text=True, cwd=str(REPO))


def _t131_replay_bitexact():
    """硬验收：现有 raw JSON 重放 → results 与已提交正式产物逐位一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "e131_phase2_optimized.json"
        out_md = Path(tmp) / "e131_phase2_report.md"
        p = _run_gen(tmp, out_json, out_md)
        if p.returncode != 0:
            check("131/gen_report 可运行（repo 相对 CLI）", False,
                  f"rc={p.returncode} {p.stderr[-300:]}")
            return
        check("131/gen_report 可运行（repo 相对 CLI）", True)
        regen = json.loads(out_json.read_text())
        committed = json.loads((RESULTS_DIR / "e131_phase2_optimized.json").read_text())
        same = (json.dumps(regen["results"], sort_keys=True)
                == json.dumps(committed["results"], sort_keys=True))
        check("131/results 表与正式产物逐位一致", same)
        # 不引用已删除的 tli_indexer_fast.py（生成器自身输出面）
        jtxt = out_json.read_text()
        mdtxt = out_md.read_text()
        check("131/生成输出不引用 tli_indexer_fast.py",
              ("tli_indexer_fast.py" not in jtxt.replace("（E127 inplace 合并后）", ""))
              and "tli_indexer_fast.py" not in mdtxt,
              "生成器输出仍引用已删除文件" if "tli_indexer_fast" in jtxt or "tli_indexer_fast" in mdtxt else "")
        # 源码身份从实际文件字节计算
        si = regen["meta"].get("source_identity", {})
        ok_hash = True
        for rel in ("sparse_attn/indexer/tli_indexer.py",
                    "sparse_attn/indexer/greedy_triton.py"):
            want = si.get(rel)
            if want is None:
                ok_hash = False
                break
            b = (REPO / rel).read_bytes()
            if want.get("md5") != hashlib.md5(b).hexdigest() \
                    or want.get("sha256") != hashlib.sha256(b).hexdigest():
                ok_hash = False
        check("131/源码身份（md5/sha256）实算自文件字节", ok_hash)


def _t131_meta_passthrough():
    """noise_band/port_note：输出已存在时透传既有 meta 字段。"""
    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "e131_phase2_optimized.json"
        out_md = Path(tmp) / "e131_phase2_report.md"
        shutil.copy(RESULTS_DIR / "e131_phase2_optimized.json", out_json)
        p = _run_gen(tmp, out_json, out_md)
        if p.returncode != 0:
            check("131/meta 透传（noise_band/port_note）", False,
                  f"rc={p.returncode} {p.stderr[-300:]}")
            return
        regen = json.loads(out_json.read_text())
        committed = json.loads((RESULTS_DIR / "e131_phase2_optimized.json").read_text())
        ok = (regen["meta"].get("noise_band") == committed["meta"].get("noise_band")
              and regen["meta"].get("port_note") == committed["meta"].get("port_note"))
        check("131/meta 透传（noise_band/port_note）", ok)


# ======================================================================
# 119：_e131_build_report 聚合前 fail-closed 校验
# ======================================================================
AGG = REPO / "exp" / "bench" / "_e131_build_report.py"


def _agg_inputs(workdir):
    for arm in ARM_FILES:
        for ext in (".json", ".json.sha256"):
            shutil.copy(RESULTS_DIR / f"e131_arm_bench_{arm}{ext}", workdir)


def _resign(path):
    """篡改后重签 sidecar（bare-hash 格式与原文件一致）——使 sha 校验通过，
    红证聚焦在 arm/S0/NaN/gate 等语义校验本身。"""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(h + "\n")


def _run_agg(workdir):
    return subprocess.run([sys.executable, str(AGG)], cwd=str(workdir),
                          capture_output=True, text=True)


def _tamper(workdir, fname, fn):
    p = Path(workdir) / f"e131_arm_bench_{fname}.json"
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    _resign(p)


def _t119_reject_tampered():
    """红证：篡改 arm / S0 / NaN / gate / sidecar / 重复 run → 退出非零。"""
    cases = []

    def mk(name, fn, fname="mavg"):
        def case():
            with tempfile.TemporaryDirectory() as td:
                _agg_inputs(td)
                _tamper(td, fname, fn)
                p = _run_agg(td)
                return p.returncode != 0, (f"rc={p.returncode} {p.stderr.strip()[-200:]}")
        cases.append((name, case))

    mk("arm 篡改（mavg→aavg）",
       lambda d: (d["runs"][0].__setitem__("arm", "aavg"),
                  d["meta"].__setitem__("arm", "aavg")))
    mk("S0 篡改（65536→8192）",
       lambda d: d["runs"][0].__setitem__("S0", 8192))
    mk("steps 篡改（30→999）",
       lambda d: d["runs"][0].__setitem__("steps", 999))
    mk("NaN 数值（step_gpu_sum_ms）",
       lambda d: d["runs"][0].__setitem__("step_gpu_sum_ms", float("nan")))
    mk("gate out_finite=false",
       lambda d: d["runs"][0]["gate"].__setitem__("out_finite", False))

    def sidecar_case():
        with tempfile.TemporaryDirectory() as td:
            _agg_inputs(td)
            # 篡改字节但【不】重签 sidecar → sha256 校验必须拦下
            p = Path(td) / "e131_arm_bench_mavg.json"
            d = json.loads(p.read_text())
            d["runs"][0]["step_gpu_sum_ms"] = 0.123
            p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
            r = _run_agg(td)
            return r.returncode != 0, f"rc={r.returncode}"

    cases.append(("sidecar sha256 不匹配", sidecar_case))

    def dup_case():
        with tempfile.TemporaryDirectory() as td:
            _agg_inputs(td)
            p = Path(td) / "e131_arm_bench_mavg.json"
            d = json.loads(p.read_text())
            d["runs"].append(d["runs"][0])
            p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
            _resign(p)
            r = _run_agg(td)
            return r.returncode != 0, f"rc={r.returncode}"

    cases.append(("重复 run（runs=2）", dup_case))

    def missing_case():
        with tempfile.TemporaryDirectory() as td:
            _agg_inputs(td)
            os.remove(Path(td) / "e131_arm_bench_mminmax.json")
            r = _run_agg(td)
            return r.returncode != 0, f"rc={r.returncode}"

    cases.append(("缺失臂输入文件", missing_case))

    for name, case in cases:
        ok, info = case()
        check(f"119/拒绝发布：{name}", ok, "" if ok else info)


def _t119_green_reproduce():
    """绿证：真数据聚合 rc=0，且输出与已提交 e131_phase1_report.json
    逐字段一致（除 generated 时间戳）。"""
    with tempfile.TemporaryDirectory() as td:
        _agg_inputs(td)
        p = _run_agg(td)
        if p.returncode != 0:
            check("119/真数据聚合复现已提交报告", False,
                  f"rc={p.returncode} {p.stderr[-300:]}")
            return
        got = json.loads((Path(td) / "e131_phase1_report.json").read_text())
        ref = json.loads((RESULTS_DIR / "e131_phase1_report.json").read_text())
        got["meta"].pop("generated", None)
        ref["meta"].pop("generated", None)
        check("119/真数据聚合复现已提交报告", got == ref)


# ======================================================================
# 130：v2 barrier residency —— watchdog env 钩子框架 + limitation 声明
# ======================================================================
def _t130_watchdog_hook():
    try:
        import sparse_attn.indexer.greedy_triton as gt
    except Exception as e:
        check("130/greedy_triton 可导入", False, str(e))
        return
    check("130/greedy_triton 可导入", True)
    # env 钩子：TLI_E131_V2_TIMEOUT_MS 默认 120000ms
    has_fn = hasattr(gt, "_v2_timeout_s") and hasattr(gt, "v2_watchdog_overdue")
    check("130/watchdog env 钩子（_v2_timeout_s/v2_watchdog_overdue）", has_fn)
    if not has_fn:
        return
    old = os.environ.pop("TLI_E131_V2_TIMEOUT_MS", None)
    try:
        default_s = gt._v2_timeout_s()
        os.environ["TLI_E131_V2_TIMEOUT_MS"] = "60000"
        env_s = gt._v2_timeout_s()
        os.environ["TLI_E131_V2_TIMEOUT_MS"] = "0"
        dis_s = gt._v2_timeout_s()
    finally:
        if old is None:
            os.environ.pop("TLI_E131_V2_TIMEOUT_MS", None)
        else:
            os.environ["TLI_E131_V2_TIMEOUT_MS"] = old
    check("130/TLI_E131_V2_TIMEOUT_MS 默认 120s / env 覆盖 / ≤0 禁用",
          default_s == 120.0 and env_s == 60.0 and dis_s == 0.0,
          f"default={default_s} env={env_s} dis={dis_s}")
    # 带外 advisory 检查（无同步语义破坏：纯 host 时间戳比对）
    gt._V2_LAUNCH_TS = None
    check("130/无 launch 记录时 watchdog 返回未超时", gt.v2_watchdog_overdue() == (False, None))
    gt._V2_LAUNCH_TS = (time.monotonic(), 120.0)
    o1 = gt.v2_watchdog_overdue()
    gt._V2_LAUNCH_TS = (time.monotonic() - 9999.0, 120.0)
    o2 = gt.v2_watchdog_overdue()
    gt._V2_LAUNCH_TS = None
    check("130/watchdog 超时判定（新鲜未超时/过期超时）",
          o1 == (False, 0.0 if o1[1] == 0.0 else o1[1]) and o1[0] is False and o2[0] is True,
          f"fresh={o1} stale={o2}")
    # limitation 声明（residency clamp 未覆盖 shared memory/分配粒度；跨架构未证明）
    src = (REPO / "sparse_attn/indexer/greedy_triton.py").read_text()
    doc_ok = ("shared memory" in src and "分配粒度" in src
              and "跨架构" in src and "H20" in src)
    check("130/limitation 声明（shared memory/分配粒度/跨架构未证明/H20 实证）",
          doc_ok)


def main():
    print(f"== E132b 红绿测试（repo={REPO}）==")
    _t129_gate_kqat_stride()
    _t129_gate_bq_full_stride()
    _t129_gate_avg_bq_stride()
    _t129_fallback_value_equiv()
    _t131_replay_bitexact()
    _t131_meta_passthrough()
    _t119_reject_tampered()
    _t119_green_reproduce()
    _t130_watchdog_hook()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n== 总计 {len(_results)} 项，失败 {n_fail} ==")
    if n_fail:
        for name, ok, info in _results:
            if not ok:
                print(f"  FAIL: {name} {info}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
