# -*- coding: utf-8 -*-
"""GPT 审计 2026-10-12_0933（core 4c26679）缺陷 111/112/113/115/116/123
红绿验收套件（E132a 修复 agent）。

  111  TL-E096-CAP-CONSUMER-TOCTOU（P1）：eval.py 消费侧读字节→parse→
       SHA→行数→sidecar 校验须在与 producer 同一把结果文件级锁
       （runcap_gate_lock，7367367 同 key：out_path+".runcap_gate.lock"）
       的窗口内一次完成（同一不可变字节快照），锁外只做 scorer。
       红（修复前）：eval 源码不含 runcap_gate_lock；他进程持 producer 锁
       3s 时 eval 不等待即完成。绿：锁竞争串行 + 评分结果正常落盘。
  112  TL-E096-CAP-UNIFORMITY（P1 可用性阻断）：新增独立 cap-comparability
       key（只含跨任务可统一的 cap 语义：effective/protocol/treatment 级），
       多任务聚合单一性检查改用该 key；完整 run identity hash 仅用于单文
       件防覆盖。红：两个不同 dataset、同 effective_cap=16384 的 strict 任
       务被误拒「混挂 N 个不同 cap identity」、无 result.json。绿：正常聚
       出 result.json；异 cap（16K vs 31.5K）仍拒（不回归 102 混挂门）。
  113  TL-E096-CAP-LEGACY-MIX（P2）：strict 与 legacy 混合表 fail-closed
       拒绝发布；全 legacy 表须显式模式且 cap 字段全 null。红：strict +
       legacy（--legacy-no-runcap）修复前照写 result.json 且 _meta 同时含
       两计数 + 非空 cap。绿：拒绝且不写 result.json；全 legacy 显式模式
       仍放行且 cap 字段全 null。
  115  TL-E096-CAP-SCHEMA-INTEGRITY（P1）：validate_run_cap_manifest 重算
       canonical payload hash + payload↔外层逐字段闭包（effective_cap/
       model/dataset/method/timestamp 等）；consumer 校验 sidecar 的
       result_file == basename(jsonl_path)。红：合法 manifest 仅改外层
       effective_cap 16384→31500 / result_file→other.jsonl（保留原 payload/
       hash）validate 仍返回成功、eval 照常打分。绿：[CAP-SCHEMA-FAIL] /
       [RUNCAP-FAIL] 拒绝；合法 manifest 不受影响。
  116  TL-E127-KV-SHAPE-STRIDE（P1）：_validate_gqa_shape 增加 V 校验——
       V token 长度==K token 长度、V head 数/head_dim/device/dtype/stride
       与 K 一致，非法 fail-before-launch。红：审计例 q[1,1,32,128] /
       k[1,16384,8,128] / v[1,1,8,128] / mask[1,8,16384] 修复前返回 true。
       绿：ValueError（含「116」）；合法 Qwen 连续同形 K/V 不触发。
  123  TL-E131-CONFIG-IDENTITY（P2）：exp/bench/e131_arm_bench.py mavg_e127
       臂的 meta.config 记录实际 effective args（make_args 返回的
       Namespace 构造），不再无条件写 COMMON_ARGS 占位。红：模块无
       arm_config_for_meta、main() 无条件 ARM_CONFIGS.get 占位。绿：
       mavg_e127 effective_args 与 make_args 逐键相等、且与 COMMON_ARGS
       在 cmp_ratio/subspace/layer_skip 三键不同；冻结五臂不变。

口径：显式 check（非 assert），python / python -O 双跑语义一致；全部
CPU（116 的 device 不匹配例用 meta device 构造，不占 GPU）；eval 段依赖
jieba/rouge（/tmp/e102_deps 回退，同 101-106 套件口径）。
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

EVAL_SRC = os.path.join(REPO, "benchmark", "LongBench", "eval.py")
E131_SRC = os.path.join(REPO, "exp", "bench", "e131_arm_bench.py")

_PASS, _FAIL, _SKIP = [], [], []


def check(name, cond, detail=""):
    if cond:
        _PASS.append(name)
        print(f"[PASS] {name}")
    else:
        _FAIL.append(name)
        print(f"[FAIL] {name} —— {detail}")


def skip(name, reason):
    _SKIP.append(name)
    print(f"[SKIP] {name} —— {reason}")


def _sys_exit_code(fn, *a, **kw):
    """SystemExit 捕获（消息走 e.code，非 stdout——历史坑）。"""
    try:
        fn(*a, **kw)
        return None, None
    except SystemExit as e:
        return e.code, str(e.code)


def _eval_env():
    env = dict(os.environ, PYTHONPATH=REPO)
    try:
        import jieba  # noqa: F401
        import rouge  # noqa: F401
        return env
    except ImportError:
        if os.path.isdir("/tmp/e102_deps"):
            env["PYTHONPATH"] = REPO + ":/tmp/e102_deps"
            return env
    return None


def _make_runcap_product(td, fname, dataset="hotpotqa", fp="deadbeef123",
                         ts="1012024000", cap=16384, n_rows=3,
                         status="complete", sha_override=None,
                         result_file_override=None, method="none"):
    """构造 JSONL + 完整 inline run-cap sidecar（096 生产同款，可参数化
    dataset/指纹/时间戳/cap——112 需要不同 dataset 同 cap 的两个产物）。"""
    from benchmark.LongBench import run_cap_manifest as rcm
    jpath = os.path.join(td, fname)
    rows = [{"pred": "a", "answers": ["a"], "all_classes": None,
             "length": 100, "_id": f"{dataset}:{fp}:{i}"}
            for i in range(n_rows)]
    with open(jpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha, lines = rcm.compute_result_binding(jpath)
    core = rcm.build_run_cap_core(
        model_name="Qwen3-8B", model_path=None, method=method,
        dataset=dataset, dataset_fingerprint=fp, run_timestamp=ts,
        effective_cap=cap,
        cap_gate_info={"registry_protocol_cap": 31500, "capability": None,
                       "max_position_embeddings": None},
        max_new_tokens=128, producer={"script_sha256": "x",
                                       "git_commit": None,
                                       "script_path": "x"})
    manifest = rcm.finalize_run_cap_manifest(
        core, status=status, n_samples=n_rows,
        input_tokens=[10] * n_rows, truncated_flags=[False] * n_rows,
        result_file=(result_file_override or fname),
        result_sha256=(None if status == "generating"
                       else (sha_override or sha)),
        result_lines=lines)
    rcm.write_run_cap_manifest(jpath + rcm.RUN_CAP_SIDECAR_SUFFIX, manifest)
    return jpath, manifest


def _run_eval(td, extra=(), timeout=300):
    env = _eval_env()
    if env is None:
        return None
    return subprocess.run(
        [sys.executable, "-m", "benchmark.LongBench.eval",
         "--output-path", td, "--t", "1012024000"] + list(extra),
        capture_output=True, text=True, cwd=REPO, env=env, timeout=timeout)


# ===========================================================================
# 111：eval.py 消费侧 TOCTOU——与 producer 同一把结果文件级锁
# ===========================================================================
_LOCK_HOLDER = (
    "import sys, time\n"
    "sys.path.insert(0, {repo!r})\n"
    "from benchmark.LongBench.run_cap_manifest import runcap_gate_lock\n"
    "with runcap_gate_lock({path!r}):\n"
    "    print('LOCKED', flush=True)\n"
    "    time.sleep(3)\n"
)


def test_111_consumer_lock():
    # ---- 静态红/绿：eval 源码接线 runcap_gate_lock ----
    src = open(EVAL_SRC, encoding="utf-8").read()
    check("111 静态：eval.py 接线 runcap_gate_lock（读字节→parse→SHA→行数"
          "→sidecar 校验同一锁窗口）",
          "runcap_gate_lock" in src and "actual_sha=actual_sha" in src,
          "eval.py 源码不含 runcap_gate_lock / 快照 sha 传参")

    # ---- 功能红/绿：他进程持 producer 锁（同一 lock key）时 eval 必须等待
    with tempfile.TemporaryDirectory() as td:
        jpath, _ = _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl")
        holder = subprocess.Popen(
            [sys.executable, "-c", _LOCK_HOLDER.format(
                repo=REPO, path=jpath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = holder.stdout.readline()
            if "LOCKED" not in line:
                err = holder.stderr.read()
                skip("111 功能：锁竞争串行", f"holder 启动失败 {err[:200]}")
                return
            t0 = time.time()
            r = _run_eval(td)
            elapsed = time.time() - t0
            rp = os.path.join(td, "result.json")
            check("111 功能：eval 与 producer 同 key 锁互斥——持锁 3s 期间"
                  f"eval 阻塞等待（实际等待 {elapsed:.2f}s）且释放后正常"
                  "完成（rc=0 + result.json 落盘）",
                  r is not None and r.returncode == 0
                  and os.path.isfile(rp) and elapsed >= 1.5,
                  f"rc={getattr(r, 'returncode', None)} elapsed={elapsed:.2f}s"
                  f"（修复前 eval 不获锁即完成——TOCTOU 交错窗口敞开）")
        finally:
            holder.wait(timeout=30)


# ===========================================================================
# 112：cap-comparability key——多任务聚合单一性的正确口径
# ===========================================================================
def test_112_cap_comparability():
    from benchmark.LongBench import run_cap_manifest as rcm

    # ---- 单元：cap_comparability_key 存在且语义正确 ----
    has_fn = hasattr(rcm, "cap_comparability_key")
    check("112 单元：run_cap_manifest.cap_comparability_key 存在",
          has_fn, "模块无 cap_comparability_key 函数")
    if has_fn:
        k16a = rcm.cap_comparability_key(
            effective_cap=16384, protocol_cap=31500,
            cap_treatment_type="restricted_context_treatment",
            cap_treatment_divergence=True)
        k16b = rcm.cap_comparability_key(
            effective_cap=16384, protocol_cap=31500,
            cap_treatment_type="restricted_context_treatment",
            cap_treatment_divergence=True)
        k315 = rcm.cap_comparability_key(
            effective_cap=31500, protocol_cap=31500,
            cap_treatment_type="protocol", cap_treatment_divergence=False)
        check("112 单元：同 cap 语义 key 稳定且异 cap key 不同",
              k16a == k16b and k16a != k315 and len(k16a) == 10,
              f"a={k16a} b={k16b} c={k315}")

    # ---- 功能红/绿：两个不同 dataset、同 effective_cap=16384 的 strict
    #      任务必须正常聚出 result.json（修复前被误拒「混 cap」）----
    with tempfile.TemporaryDirectory() as td:
        _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                             dataset="hotpotqa", fp="fp_hotpot",
                             ts="1012024000", cap=16384)
        _make_runcap_product(td, "qasper-none-1012024001.jsonl",
                             dataset="qasper", fp="fp_qasper",
                             ts="1012024001", cap=16384)
        r = _run_eval(td)
        rp = os.path.join(td, "result.json")
        ok = r is not None and r.returncode == 0 and os.path.isfile(rp)
        meta = json.load(open(rp, encoding="utf-8"))["_meta"]["runcap"] \
            if ok else {}
        check("112 功能：不同 dataset 同 effective_cap=16384 的两个 strict "
              "任务正常聚出 result.json（跨任务只统一 cap 语义，不要求同一"
              "run identity）",
              ok and meta.get("strict_validated") == 2
              and meta.get("cap_effective") == 16384
              and meta.get("cap_comparability_key") is not None,
              f"rc={getattr(r, 'returncode', None)} "
              f"err={getattr(r, 'stderr', '')[-300:]!r}")

    # ---- 不回归 102：同目录异 cap（16K vs 31.5K）仍拒绝 ----
    with tempfile.TemporaryDirectory() as td:
        _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                             cap=16384)
        _make_runcap_product(td, "qasper-none-1012024001.jsonl",
                             dataset="qasper", fp="fp_qasper",
                             ts="1012024001", cap=31500)
        r = _run_eval(td)
        check("112 不回归：异 cap（16K vs 31.5K）混挂目录仍拒绝"
              "（[RUNCAP-FAIL] + 「cap identity」文案）且不写 result.json",
              r is not None and r.returncode != 0
              and "cap identity" in r.stdout + r.stderr
              and not os.path.exists(os.path.join(td, "result.json")),
              f"rc={getattr(r, 'returncode', None)}")


# ===========================================================================
# 113：strict 与 legacy 混合表 fail-closed
# ===========================================================================
def test_113_legacy_mix():
    # ---- 红/绿：strict + legacy（显式 --legacy-no-runcap）混合 → 拒绝发布
    with tempfile.TemporaryDirectory() as td:
        _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl", cap=16384)
        legacy = os.path.join(td, "qasper-none-1012024001.jsonl")
        with open(legacy, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps(
                    {"pred": "a", "answers": ["a"], "all_classes": None,
                     "length": 100, "_id": f"qasper:legacyfp:{i}"},
                    ensure_ascii=False) + "\n")
        r = _run_eval(td, ["--legacy-no-runcap"])
        rp = os.path.join(td, "result.json")
        check("113 红/绿：strict + legacy 混合目录（--legacy-no-runcap）→ "
              "fail-closed 拒绝发布（[RUNCAP-FAIL]，不写 result.json——修复"
              "前照写且把 strict 的 cap 标到整张表）",
              r is not None and r.returncode != 0
              and "[RUNCAP-FAIL]" in r.stdout + r.stderr
              and not os.path.exists(rp),
              f"rc={getattr(r, 'returncode', None)} "
              f"result.json={'存在（修复前行为）' if os.path.exists(rp) else '不存在'}")

    # ---- 绿：全 legacy + 显式模式 → 放行且 cap 字段全 null ----
    with tempfile.TemporaryDirectory() as td:
        for fn, ds in (("hotpotqa-none-1012024000.jsonl", "hotpotqa"),
                       ("qasper-none-1012024001.jsonl", "qasper")):
            with open(os.path.join(td, fn), "w", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps(
                        {"pred": "a", "answers": ["a"], "all_classes": None,
                         "length": 100, "_id": f"{ds}:legacyfp:{i}"},
                        ensure_ascii=False) + "\n")
        r = _run_eval(td, ["--legacy-no-runcap"])
        rp = os.path.join(td, "result.json")
        ok = r is not None and r.returncode == 0 and os.path.isfile(rp)
        meta = json.load(open(rp, encoding="utf-8"))["_meta"]["runcap"] \
            if ok else {}
        check("113 绿：全 legacy 显式模式放行且 cap 字段全 null"
              "（cap_effective/cap_protocol/cap_comparability_key 均如实"
              " null，不冒称）",
              ok and meta.get("legacy_no_sidecar") == 2
              and meta.get("strict_validated") == 0
              and meta.get("cap_effective") is None
              and meta.get("cap_protocol") is None
              and meta.get("cap_comparability_key") is None,
              f"rc={getattr(r, 'returncode', None)} meta={meta}")


# ===========================================================================
# 115：schema 完整性——payload hash 重算 + payload↔外层闭包 + result_file
# ===========================================================================
def _mk_good_manifest():
    from benchmark.LongBench import run_cap_manifest as rcm
    core = rcm.build_run_cap_core(
        model_name="Qwen3-8B", model_path=None, method="none",
        dataset="hotpotqa", dataset_fingerprint="deadbeef123",
        run_timestamp="1012024000", effective_cap=16384,
        cap_gate_info={"registry_protocol_cap": 31500, "capability": None,
                       "max_position_embeddings": None},
        max_new_tokens=128, producer={"script_sha256": "x",
                                       "git_commit": None,
                                       "script_path": "x"})
    return rcm.finalize_run_cap_manifest(
        core, status="complete", n_samples=3,
        input_tokens=[10, 10, 10], truncated_flags=[False] * 3,
        result_file="hotpotqa-none-1012024000.jsonl",
        result_sha256="0" * 64, result_lines=3)


def test_115_schema_integrity():
    from benchmark.LongBench import run_cap_manifest as rcm

    # ---- 绿：合法 manifest 仍通过（不误伤）----
    code, msg = _sys_exit_code(
        rcm.validate_run_cap_manifest, _mk_good_manifest())
    check("115 绿：合法 inline manifest 通过强化 schema 门（payload hash "
          "重算 + 闭包核对零误伤）",
          code is None, f"意外拒绝：{msg}")

    # ---- 红/绿 ①：外层 effective_cap 篡改（保留原 payload/hash）→ 拒绝
    m = _mk_good_manifest()
    m["effective_cap"] = 31500
    code, msg = _sys_exit_code(rcm.validate_run_cap_manifest, m)
    check("115 红/绿：外层 effective_cap 16384→31500（payload/hash 原样）→ "
          "[CAP-SCHEMA-FAIL] 拒绝（修复前 validate 返回成功——错误 cap "
          "sidecar 静默进入结果元数据）",
          code is not None and "[CAP-SCHEMA-FAIL]" in str(msg)
          and "effective_cap" in str(msg),
          f"code={code} msg={str(msg)[:150]}")

    # ---- 红/绿 ②：外层 dataset 篡改 → 拒绝 ----
    m = _mk_good_manifest()
    m["dataset"] = "qasper"
    code, msg = _sys_exit_code(rcm.validate_run_cap_manifest, m)
    check("115 红/绿：外层 dataset 篡改 → [CAP-SCHEMA-FAIL] 拒绝",
          code is not None and "[CAP-SCHEMA-FAIL]" in str(msg)
          and "dataset" in str(msg),
          f"code={code} msg={str(msg)[:150]}")

    # ---- 红/绿 ③：payload 字段篡改但不更新 hash → hash 重算不符拒绝 ----
    m = _mk_good_manifest()
    m["cap_identity"]["payload"]["effective_cap"] = 31500
    code, msg = _sys_exit_code(rcm.validate_run_cap_manifest, m)
    check("115 红/绿：payload.effective_cap 篡改（hash 未更新）→ hash 重算"
          "不符拒绝",
          code is not None and "[CAP-SCHEMA-FAIL]" in str(msg),
          f"code={code} msg={str(msg)[:150]}")

    # ---- 红/绿 ④：payload + hash 同步篡改 → payload↔外层闭包拒绝 ----
    m = _mk_good_manifest()
    m["cap_identity"]["payload"]["method"] = "tli"
    m["cap_identity"]["hash"] = __import__("hashlib").sha256(
        rcm._canonical(m["cap_identity"]["payload"]).encode("utf-8")
    ).hexdigest()[:10]
    code, msg = _sys_exit_code(rcm.validate_run_cap_manifest, m)
    check("115 红/绿：payload+hash 同步篡改（method none→tli）→ payload↔"
          "外层闭包核对拒绝（重算 hash 挡不住的自洽篡改）",
          code is not None and "[CAP-SCHEMA-FAIL]" in str(msg)
          and "method" in str(msg),
          f"code={code} msg={str(msg)[:150]}")

    # ---- 红/绿 ⑤：consumer 校验 sidecar.result_file == basename(jsonl) ----
    with tempfile.TemporaryDirectory() as td:
        _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                             result_file_override="other.jsonl")
        r = _run_eval(td)
        check("115 红/绿：sidecar result_file=other.jsonl ≠ 当前 JSONL "
              "basename → eval 拒绝打分且不写 result.json（误拷/错配 "
              "backfill 拦截；修复前照常打分）",
              r is not None and r.returncode != 0
              and "result_file" in r.stdout + r.stderr
              and not os.path.exists(os.path.join(td, "result.json")),
              f"rc={getattr(r, 'returncode', None)}")


# ===========================================================================
# 116：_validate_gqa_shape V 校验（fail-before-launch）
# ===========================================================================
def test_116_v_shape_gate():
    try:
        import torch
    except ImportError as e:
        skip("116 段全部", f"torch 不可用 {e}")
        return
    import sparse_attn.ops.eager_decoding as ed

    # 审计红证例：q[1,1,32,128] / k[1,16384,8,128] / v[1,1,8,128]
    # / mask[1,8,16384]——修复前门返回 true（V token 长度 1 被按 K 的
    # 16384 寻址，任何 tok>=1 都越界）
    q = torch.zeros(1, 1, 32, 128)
    k = torch.zeros(1, 16384, 8, 128)
    v_short = torch.zeros(1, 1, 8, 128)
    mask = torch.zeros(1, 8, 16384, dtype=torch.bool)
    mask[:, :, :16] = True

    def _expect_reject(name, tensors):
        q_, k_, v_, m_ = tensors
        try:
            ed._validate_gqa_shape(q_, k_, v_, m_, 1)
            check(f"116 {name} → ValueError", False,
                  "门返回 true（未拒绝——修复前行为）")
        except ValueError as e:
            check(f"116 {name} → ValueError（fail-before-launch，文案含 "
                  "「116」）",
                  "116" in str(e), str(e)[:140])

    _expect_reject("审计例 V token 长度 1 ≠ K 16384",
                   (q, k, v_short, mask))
    _expect_reject("V head_dim 64 ≠ K head_dim 128",
                   (q, k, torch.zeros(1, 16384, 8, 64), mask))
    _expect_reject("V dtype fp32 ≠ K bf16",
                   (q, k.to(torch.bfloat16),
                    torch.zeros(1, 16384, 8, 128, dtype=torch.float32),
                    mask))
    _expect_reject("V device meta ≠ K cpu",
                   (q, k, torch.zeros(1, 16384, 8, 128, device="meta"),
                    mask))
    v_strided = torch.zeros(1, 8, 16384, 128).transpose(1, 2)
    _expect_reject("V stride 与 K 不同（转置布局——kernel 以 K stride 寻址 "
                   "V 必错址）",
                   (q, k, v_strided, mask))

    # ---- 绿：合法 Qwen 连续同形 K/V 不触发（零误伤）----
    try:
        ed._validate_gqa_shape(q, k, torch.zeros(1, 16384, 8, 128), mask, 1)
        check("116 绿：合法 Qwen 连续同形 K/V（H=32/Hkv=8/D=128）正常放行",
              True)
    except ValueError as e:
        check("116 绿：合法 Qwen 连续同形 K/V（H=32/Hkv=8/D=128）正常放行",
              False, str(e)[:140])


# ===========================================================================
# 123：e131_arm_bench mavg_e127 臂 meta.config 记录实际 effective args
# ===========================================================================
def test_123_arm_config_identity():
    src = open(E131_SRC, encoding="utf-8").read()
    check("123 静态：e131_arm_bench 含 arm_config_for_meta（mavg_e127 记"
          "实际 effective args，非 COMMON_ARGS 占位）",
          "arm_config_for_meta" in src,
          "源码无 arm_config_for_meta——main() 无条件 "
          "ARM_CONFIGS.get(args.arm, {note 占位})")
    try:
        import torch  # noqa: F401  （模块顶层 import torch，CPU 导入即可）
        spec = importlib.util.spec_from_file_location(
            "e131_arm_bench_e132a", E131_SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["e131_arm_bench_e132a"] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        skip("123 功能段", f"e131_arm_bench 导入失败 {type(e).__name__}: {e}")
        return

    has = hasattr(mod, "arm_config_for_meta")
    check("123 功能：arm_config_for_meta 可调用", has, "模块无该函数")
    if not has:
        return
    cfg = mod.arm_config_for_meta("mavg_e127")
    eff = vars(mod.make_args("mavg_e127"))
    ok = (cfg.get("effective_args") == eff
          and eff["tia_level2_cmp_ratio"] == 2
          and eff["tli_subspace"] == "full"
          and eff["tli_enable_layer_skip"] is True
          # 与 COMMON_ARGS 的三键差异显式可检（修复前 meta 冒称 COMMON）
          and eff["tia_level2_cmp_ratio"]
          != mod.COMMON_ARGS["tia_level2_cmp_ratio"]
          and eff["tli_subspace"] != mod.COMMON_ARGS["tli_subspace"]
          and eff["tli_enable_layer_skip"]
          != mod.COMMON_ARGS["tli_enable_layer_skip"])
    check("123 绿：mavg_e127 meta.config.effective_args == make_args 实际"
          "值（cmp_ratio=2 / subspace=full / layer_skip=True，与 "
          "COMMON_ARGS 三键不同——机器校准锚点配置身份可复核）",
          ok, f"cfg={cfg}")
    check("123 绿：冻结五臂 meta.config 仍逐字段记 ARM_CONFIGS（E126 派单"
          "口径不变）",
          mod.arm_config_for_meta("mavg") == dict(mod.ARM_CONFIGS["mavg"])
          and mod.arm_config_for_meta("cclustersim")
          == dict(mod.ARM_CONFIGS["cclustersim"]))


def main():
    tests = [test_111_consumer_lock, test_112_cap_comparability,
             test_113_legacy_mix, test_115_schema_integrity,
             test_116_v_shape_gate, test_123_arm_config_identity]
    for t in tests:
        print(f"\n== {t.__name__} ==")
        t()
    print(f"\n=== {len(_PASS)} PASS / {len(_FAIL)} FAIL"
          f"{f' / {len(_SKIP)} SKIP' if _SKIP else ''} ===")
    if _FAIL:
        for f in _FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
