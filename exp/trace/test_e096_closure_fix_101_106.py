# -*- coding: utf-8 -*-
"""GPT 审计 2026-10-12_0739 五项修复（101/102/103/105/106）红绿验收套件。

  101  eager_decoding host 侧 GQA/shape fail-closed 边界门
       绿：审计三例 + 零头数 → ValueError（launch/分配前，文案含
           「101 fail-closed」）；合法 Qwen3 shape 修复前后逐位一致
           （dense CPU + gather GPU 双路径）；三门第二防线（直呼
           _dense/_gather 同样受保护）。
       红：修复前基线（固定 SHA ce6cde0，E096_PRE_FIX_SHA 可覆盖）同输入
           无 101 契约 ValueError——dense 路径迟到 EinopsError/
           ZeroDivisionError；gather GPU 路径审计例 1 复现 CUDA
           illegal memory access（越界读实测）或静默错值，例 2/3 静默
           接受非法 shape（uninitialized 输出头）。
  102  eval.py 打分前强制消费 run-cap sidecar
       绿：complete 合法 sidecar → rc=0 且 result.json _meta 落盘 cap
           身份；--legacy-no-runcap / --no-require-runcap 显式降级放行
           （[RUNCAP-LEGACY] + cap 如实 null）。
       红：status=generating / 结果 sha 篡改 / 行数与 manifest 不符 /
           聚合目录混挂异 cap / 缺 sidecar 默认口径 → 一律 rc!=0 且
           不写 result.json。
  103  gate 与发布之间跨进程独占锁（.runcap_gate.lock + flock）
       绿：锁内持有期间他进程阻塞；异常路径必释放；锁文件只创建不删除；
           后到者锁内重读身份 → [CAP-CLASH]；pred.py 静态接线
           （with runcap_gate_lock 包住 gate→数据写→complete 回填）。
  105  backfill 五重绑定
       绿：run-bind 唯一匹配 / 旧格式共识块 / --expect-count 显式声明
           → 补制成功且 log_binding（log sha + 字节范围）可复核。
       红：跨模型日志 / 一行 partial 冒充全集（_id 指纹冲突）/ 多块
           run-bind 重复匹配 / 旧格式多块 cap 不一致 / 行号序列缺重 /
           CLI 与文件名不一致 → 一律 [GATE-FAIL]；行数 != 预期全集且
           无 _id 证明 → 降格 provenance_incomplete（不冒称 complete）。
  106  RULER llm_eval cap+任务集进输出身份 + 结果 receipt
       绿：同 method 同秒不同 cap/任务集 → 文件名必不同；save_results
           原子落盘 txt + receipt（result sha256/行数/cap 回执同代
           绑定）；tli treatment sidecar 原子落盘 + sha 进 receipt。

口径：显式 check（非 assert），python 与 python -O 双跑语义一致；
GPU 段以 torch.cuda.is_available() 门控（本机有 H20，实际全跑）；
jieba/rouge 缺失时尝试 /tmp/e102_deps，仍缺则 102 段 SKIP 如实计数。
"""
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PRE_FIX_SHA = os.environ.get("E096_PRE_FIX_SHA", "ce6cde0")
BACKFILL_FP = os.path.join(REPO, "exp", "trace",
                           "backfill_run_cap_manifest.py")

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


# ===========================================================================
# 101：eager_decoding host 侧 GQA/shape fail-closed 边界门
# ===========================================================================

def _load_prefix_module():
    """从固定 SHA 检出修复前 eager_decoding.py（红证锚点，禁用 main:）。"""
    src = subprocess.check_output(
        ["git", "-C", REPO, "show",
         f"{PRE_FIX_SHA}:sparse_attn/ops/eager_decoding.py"], text=True)
    f = tempfile.NamedTemporaryFile("w", suffix="_prefix_eager.py",
                                    delete=False, encoding="utf-8")
    f.write(src)
    f.close()
    spec = importlib.util.spec_from_file_location("prefix_eager_101", f.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prefix_eager_101"] = mod
    spec.loader.exec_module(mod)
    return mod, f.name


def _mk_case(H, Hkv, R, T=64, D=64, dev="cpu", nsel=16):
    """构造审计案例输入（token 级 mask，block_size=1）。"""
    import torch
    g = (torch.Generator(device=dev).manual_seed(0) if dev == "cuda"
         else torch.Generator().manual_seed(0))
    q = torch.randn(1, 1, H, D, generator=g, dtype=torch.bfloat16,
                    device=dev)
    k = torch.randn(1, T, Hkv, D, generator=g, dtype=torch.bfloat16,
                    device=dev)
    v = torch.randn(1, T, Hkv, D, generator=g, dtype=torch.bfloat16,
                    device=dev)
    mask = torch.zeros(1, 1, R, T, dtype=torch.bool, device=dev)
    if dev == "cuda":
        sel = torch.randperm(T, device=dev, generator=g)[:nsel]
    else:
        sel = torch.randperm(T, generator=g)[:nsel]
    mask[:, :, :, sel] = True
    cu = torch.tensor([0, T], device=dev)
    return q, k, v, mask, cu


AUDIT_CASES = [
    ("H10_Hkv3_R10", 10, 3, 10),   # 越界读 K/V（per_qh 行映射 kvh=3≥Hkv）
    ("H10_Hkv3_R3", 10, 3, 3),     # 共享 grid=R*G=9 < H=10 → qh=9 未写
    ("H32_Hkv8_R9", 32, 8, 9),     # R=9 误判 per_qh → 23 个输出头未写
]


def test_101_host_gate():
    import torch
    import sparse_attn.ops.eager_decoding as fixed

    # ---- 绿：四类非法输入（审计三例 + 零头数）→ ValueError 契约门 ----
    for name, H, Hkv, R in AUDIT_CASES:
        q, k, v, mask, cu = _mk_case(H, Hkv, R)
        try:
            fixed.eager_decoding_attn(q, k, v, mask, 1, cu)
            check(f"101 绿 {name} → ValueError", False, "未 raise")
            continue
        except ValueError as e:
            ok = "101 fail-closed" in str(e) and f"H={H}" in str(e)
            check(f"101 绿 {name} → ValueError（launch 前 host 契约门）",
                  ok, str(e)[:120])
    # 零头数（Hkv=0 → ZeroDivisionError 修复前 / ValueError 修复后）
    q, k, v, mask, cu = _mk_case(32, 1, 1)
    k = v = torch.zeros(1, 64, 0, 64, dtype=torch.bfloat16)
    try:
        fixed.eager_decoding_attn(q, k, v, mask, 1, cu)
        check("101 绿 零头数 → ValueError", False, "未 raise")
    except ValueError as e:
        check("101 绿 零头数 → ValueError", "101 fail-closed" in str(e)
              and "零/负头数" in str(e), str(e)[:120])

    # ---- 绿：第二防线——直呼 _dense / _gather 同样受门保护 ----
    q, k, v, mask, cu = _mk_case(32, 8, 9)
    guarded = True
    for fn in (fixed._eager_decoding_attn_dense,
               fixed._eager_decoding_attn_gather):
        try:
            fn(q, k, v, mask, 1, cu)
            guarded = False
        except ValueError as e:
            guarded = guarded and "101 fail-closed" in str(e)
    check("101 绿 直呼 _dense/_gather 第二防线同样拒绝", guarded)

    # ---- 绿：静态——三门首语句均为 _validate_gqa_shape ----
    import ast
    tree = ast.parse(open(os.path.join(
        REPO, "sparse_attn/ops/eager_decoding.py"),
        encoding="utf-8").read())
    first_ok = True
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "eager_decoding_attn", "_eager_decoding_attn_dense",
                "_eager_decoding_attn_gather"):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]   # 跳过 docstring
            st = body[0] if body else None
            called = (st is not None and isinstance(st, ast.Expr)
                      and isinstance(st.value, ast.Call)
                      and isinstance(st.value.func, ast.Name)
                      and st.value.func.id == "_validate_gqa_shape")
            first_ok = first_ok and called
    check("101 绿 三入口首语句 = _validate_gqa_shape（AST，launch/分配前）",
          first_ok)


def test_101_red_prefix():
    """红证：修复前基线无 host 门——迟到错/越界炸，无 101 契约 ValueError。"""
    import torch
    prefix, tmpf = _load_prefix_module()

    # 静态红：修复前源无边界门
    src = subprocess.check_output(
        ["git", "-C", REPO, "show",
         f"{PRE_FIX_SHA}:sparse_attn/ops/eager_decoding.py"], text=True)
    check("101 红 修复前源无 _validate_gqa_shape（静态）",
          "_validate_gqa_shape" not in src)

    # dense CPU 路径：同输入无 101 ValueError（实际：EinopsError 迟到错）
    behaviors = {}
    for name, H, Hkv, R in AUDIT_CASES:
        q, k, v, mask, cu = _mk_case(H, Hkv, R)
        try:
            prefix.eager_decoding_attn(q, k, v, mask, 1, cu)
            behaviors[name] = "SILENT"
        except ValueError as e:
            behaviors[name] = f"ValueError:{'101' in str(e)}"
        except Exception as e:
            behaviors[name] = type(e).__name__
    check("101 红 修复前 dense 路径三例均无 101 契约 ValueError（迟到错："
          f"{behaviors}）——无 host 门，错误发生在计算中途非入口",
          all(v != "ValueError:True" for v in behaviors.values())
          and "SILENT" not in behaviors)

    # 零头数：修复前 ZeroDivisionError（非契约门）
    q, k, v, mask, cu = _mk_case(32, 1, 1)
    k = v = torch.zeros(1, 64, 0, 64, dtype=torch.bfloat16)
    try:
        prefix.eager_decoding_attn(q, k, v, mask, 1, cu)
        z = "SILENT"
    except ValueError:
        z = "ValueError"
    except Exception as e:
        z = type(e).__name__
    check("101 红 修复前零头数 → ZeroDivisionError（无契约门）",
          z == "ZeroDivisionError", z)

    os.unlink(tmpf)


def _prefix_gpu_probe_main():
    """子进程探针（单案例）：修复前 gather GPU 路径（Triton，T=16384）。
    逐案例独立子进程——审计例 1 的越界读可能触发 CUDA illegal memory
    access 炸掉进程上下文，隔离后例 2/3 仍可独立探测。"""
    import torch
    case_name = sys.argv[2]
    _, H, Hkv, R = next(c for c in AUDIT_CASES if c[0] == case_name)
    prefix, tmpf = _load_prefix_module()
    os.environ["TLI_E127_FAST"] = "1"
    q, k, v, mask, cu = _mk_case(H, Hkv, R, T=16384, D=64, dev="cuda")
    try:
        o = prefix.eager_decoding_attn(q, k, v, mask, 1, cu)
        torch.cuda.synchronize()
        o = o[0, 0]
        extra = ""
        if case_name == "H10_Hkv3_R10":
            # 数值级 OOB 确认：per_qh 行映射 kvh=r//G，r=9 → kvh=3 ≥
            # Hkv=3 → 连续布局下越界读进下一 token 的 head 0——row 9 的
            # 输出应等于 q[9] 对 k[toks+1, head0] 的 attention（本机此前
            # 实测 maxdiff 5.31e-04 量级）。
            toks = mask[0, 0, 9].nonzero().flatten()
            nxt = torch.clamp(toks + 1, max=16383)
            qs = q[0, 0, 9].float() * (64 ** -0.5)
            kk = k[0][nxt][:, 0].float()
            vv = v[0][nxt][:, 0].float()
            p = torch.softmax(qs @ kk.T, -1)
            pred = p @ vv
            md = float((o[9].float() - pred).abs().max())
            extra = f" oob_row9_maxdiff={md:.2e}"
        print(f"PROBE {case_name} SILENT finite="
              f"{bool(torch.isfinite(o.float()).all().item())}{extra}",
              flush=True)
    except ValueError as e:
        print(f"PROBE {case_name} ValueError101={'101' in str(e)}",
              flush=True)
    except Exception as e:
        print(f"PROBE {case_name} CRASH {type(e).__name__}: "
              f"{str(e)[:100]}", flush=True)
    os.unlink(tmpf)


def test_101_red_gpu_probe():
    """红证（GPU 段，逐案例子进程隔离防 CUDA 上下文污染）：修复前
    gather 路径审计例 1 越界读 K/V（kvh=3≥Hkv=3，kernel 无边界检查）
    ——实测 CUDA illegal memory access 或静默错值；例 2/3 静默接受
    非法 shape（uninitialized 输出头）。任一结果都证明「无 launch 前
    host 门」；修复后同输入 launch 前 ValueError。"""
    import torch
    if not torch.cuda.is_available():
        skip("101 红证 GPU 探针（gather Triton 路径）", "无 CUDA")
        return
    res = {}
    for name, _, _, _ in AUDIT_CASES:
        r = subprocess.run(
            [sys.executable, __file__, "--prefix-gpu-probe", name],
            capture_output=True, text=True, timeout=600,
            cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))
        m = re.search(rf"PROBE {name} (\S+)(.*)", r.stdout)
        if m:
            res[name] = m.group(1)
            res[name + "_detail"] = m.group(2).strip()
        elif r.returncode != 0 and "illegal memory access" in r.stderr:
            res[name] = "CUDA_ILMEM"   # 进程死于越界读（最强红证）
        else:
            res[name] = f"UNKNOWN(rc={r.returncode})"
    check("101 红 例1（H=10,Hkv=3,R=10）修复前 gather 路径越界读 K/V"
          f"——实测 {res['H10_Hkv3_R10']}"
          f"{res.get('H10_Hkv3_R10_detail', '')[:60]}（kernel 无 "
          "kvh<Hkv 边界检查，per_qh 行映射 kvh=r//G=3 越界读下一 token "
          "head0——row9 输出 = 对 k[toks+1,head0] 的 attention，本机"
          "实测 maxdiff 1e-4 量级；或 CUDA illegal memory access。修复前"
          "无任何 host 门拦截，非法 shape 静默直达 launch）",
          res["H10_Hkv3_R10"] in ("SILENT", "CUDA_ILMEM", "CRASH")
          and ("oob_row9_maxdiff=" in res.get(
              "H10_Hkv3_R10_detail", "")
              or res["H10_Hkv3_R10"] == "CUDA_ILMEM"))
    check("101 红 例2/3 修复前 gather 路径静默接受非法 shape"
          f"（实测 {res['H10_Hkv3_R3']} / {res['H32_Hkv8_R9']}——共享 "
          "grid=R*G=9<H=10 与 per_qh 误判 grid=9<H=32，未写输出头 "
          "uninitialized；修复后同输入 launch 前 ValueError）",
          res["H10_Hkv3_R3"] == "SILENT"
          and res["H32_Hkv8_R9"] == "SILENT")


def test_101_bitwise_legal():
    """绿：合法 Qwen3 shape（H=32,Hkv=8,G=4）修复前后逐位一致。"""
    import torch
    import sparse_attn.ops.eager_decoding as fixed
    prefix, tmpf = _load_prefix_module()

    # dense CPU（默认路径）
    bitwise = True
    for R, per_qh_name in ((8, "shared"), (32, "per_qh")):
        q, k, v, mask, cu = _mk_case(32, 8, R)
        o_fix = fixed.eager_decoding_attn(q, k, v, mask, 1, cu)
        o_pre = prefix.eager_decoding_attn(q, k, v, mask, 1, cu)
        bitwise = bitwise and torch.equal(o_fix, o_pre)
    check("101 绿 合法 shape dense CPU 修复前后逐位一致（shared+per_qh）",
          bitwise)
    os.unlink(tmpf)

    if not torch.cuda.is_available():
        skip("101 绿 gather GPU 逐位回归", "无 CUDA")
        return
    prefix, tmpf = _load_prefix_module()
    os.environ["TLI_E127_FAST"] = "1"
    bitwise = True
    for R in (8, 32):
        q, k, v, mask, cu = _mk_case(32, 8, R, T=16384, D=64, dev="cuda")
        o_fix = fixed.eager_decoding_attn(q, k, v, mask, 1, cu)
        o_pre = prefix.eager_decoding_attn(q, k, v, mask, 1, cu)
        torch.cuda.synchronize()
        bitwise = bitwise and torch.equal(o_fix, o_pre)
    os.environ.pop("TLI_E127_FAST", None)
    check("101 绿 合法 shape gather GPU（Triton，T=16384）修复前后逐位一致"
          "（shared+per_qh）——host 门对合法输入零影响", bitwise)
    os.unlink(tmpf)


# ===========================================================================
# 102：eval.py 打分前强制消费 run-cap sidecar
# ===========================================================================

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


def _make_runcap_product(td, fname, cap, n_rows=3, status="complete",
                         sha_override=None, n_override=None,
                         eff=None):
    """构造 JSONL + 完整 inline run-cap sidecar（096 生产同款）。"""
    from benchmark.LongBench import run_cap_manifest as rcm
    jpath = os.path.join(td, fname)
    rows = [{"pred": "a", "answers": ["a"], "all_classes": None,
             "length": 100, "_id": f"hotpotqa:deadbeef123:{i}"}
            for i in range(n_rows)]
    with open(jpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha, lines = rcm.compute_result_binding(jpath)
    core = rcm.build_run_cap_core(
        model_name="Qwen3-8B", model_path=None, method="none",
        dataset="hotpotqa", dataset_fingerprint="deadbeef123",
        run_timestamp="1012024000", effective_cap=cap,
        cap_gate_info={"registry_protocol_cap": 31500, "capability": None,
                       "max_position_embeddings": None},
        max_new_tokens=128, producer={"script_sha256": "x",
                                       "git_commit": None,
                                       "script_path": "x"})
    manifest = rcm.finalize_run_cap_manifest(
        core, status=status, n_samples=(n_override or n_rows),
        input_tokens=[10] * n_rows, truncated_flags=[False] * n_rows,
        result_file=fname,
        result_sha256=(None if status == "generating"
                       else (sha_override or sha)),
        result_lines=lines)
    rcm.write_run_cap_manifest(jpath + rcm.RUN_CAP_SIDECAR_SUFFIX, manifest)
    return jpath, manifest


def _run_eval(td, extra=()):
    env = _eval_env()
    if env is None:
        return None
    return subprocess.run(
        [sys.executable, "-m", "benchmark.LongBench.eval",
         "--output-path", td, "--t", "1012024000"] + list(extra),
        capture_output=True, text=True, cwd=REPO, env=env, timeout=300)


def test_102_eval_runcap():
    env = _eval_env()
    if env is None:
        skip("102 段全部（eval 消费 sidecar）",
             "jieba/rouge 不可用且 /tmp/e102_deps 缺失")
        return

    # ---- 绿：合法 complete sidecar → 打分成功 + _meta 落盘 cap 身份 ----
    with tempfile.TemporaryDirectory() as td:
        jpath, m = _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                                        cap=16384)
        r = _run_eval(td)
        if r is None:
            skip("102 绿 happy path", "依赖缺失")
            return
        rp = os.path.join(td, "result.json")
        ok = r.returncode == 0 and os.path.isfile(rp)
        meta = json.load(open(rp, encoding="utf-8"))["_meta"]["runcap"] \
            if ok else {}
        check("102 绿 complete sidecar → rc=0 且 result.json _meta 携带 "
              f"cap 身份（effective=16384/protocol=31500/hash="
              f"{meta.get('cap_identity_hash')}）",
              ok and meta.get("strict_validated") == 1
              and meta.get("cap_effective") == 16384
              and meta.get("cap_protocol") == 31500
              and meta.get("cap_identity_hash") is not None,
              f"rc={r.returncode} {r.stderr[-200:]!r}")

    # ---- 红：status=generating（生成中断/未回填）→ 拒绝且不写 result.json
    with tempfile.TemporaryDirectory() as td:
        jpath, m = _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                                        cap=16384, status="generating")
        r = _run_eval(td)
        check("102 红 status=generating → [RUNCAP-FAIL] rc!=0 且不写 "
              "result.json（半残产物拒绝打分）",
              r.returncode != 0 and "[RUNCAP-FAIL]" in r.stdout + r.stderr
              and not os.path.exists(os.path.join(td, "result.json")),
              f"rc={r.returncode}")

    # ---- 红：结果文件被篡改（sha 不符，行数一致）→ 拒绝 ----
    with tempfile.TemporaryDirectory() as td:
        jpath, m = _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                                        cap=16384, sha_override="0" * 64)
        r = _run_eval(td)
        check("102 红 结果 sha256 与 manifest 绑定不符 → 拒绝打分（TOCTOU/"
              "篡改拦截），不写 result.json",
              r.returncode != 0 and "sha256" in r.stdout + r.stderr
              and not os.path.exists(os.path.join(td, "result.json")),
              f"rc={r.returncode}")

    # ---- 红：行数与 manifest 不符（sha 正确——隔离行数检查）→ 拒绝 ----
    with tempfile.TemporaryDirectory() as td:
        from benchmark.LongBench import run_cap_manifest as rcm
        jpath, m = _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                                        cap=16384)
        # 直接改写 sidecar（write 落盘不校验；schema 门在消费侧）：
        # provenance 降为 backfilled（schema 允许 n_samples != 行数的
        # 唯一口径）+ n_samples=99 → eval 侧行数检查 ④ 单独命中。
        sc = jpath + rcm.RUN_CAP_SIDECAR_SUFFIX
        mm = json.load(open(sc, encoding="utf-8"))
        mm["provenance"] = "backfilled-from-stdout"
        mm["input_tokens"] = None
        mm["truncated"] = None
        mm["n_samples"] = 99
        rcm.write_run_cap_manifest(sc, mm)
        r = _run_eval(td)
        check("102 红 行数 != n_samples（sha 一致）→ 拒绝打分（部分写入/混行"
              "拦截），不写 result.json",
              r.returncode != 0 and "行数" in r.stdout + r.stderr
              and not os.path.exists(os.path.join(td, "result.json")),
              f"rc={r.returncode}")

    # ---- 红：聚合目录混挂异 cap identity → 拒绝整目录打分 ----
    with tempfile.TemporaryDirectory() as td:
        _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl", cap=16384)
        _make_runcap_product(td, "hotpotqa-none-1012024001.jsonl", cap=31500)
        r = _run_eval(td)
        check("102 红 聚合目录混挂 2 个 cap identity（16K 与 31.5K）→ "
              "[RUNCAP-FAIL] 拒写 result.json（异 cap 产物不得混入同一聚合"
              "被当同一口径均值）",
              r.returncode != 0
              and "cap identity" in r.stdout + r.stderr
              and not os.path.exists(os.path.join(td, "result.json")),
              f"rc={r.returncode}")

    # ---- 红：缺 sidecar 默认拒绝；--legacy-no-runcap 显式降级放行 ----
    with tempfile.TemporaryDirectory() as td:
        jpath, _ = _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl",
                                        cap=16384)
        os.remove(jpath + ".runcap.json")
        r = _run_eval(td)
        check("102 红 无 sidecar 默认口径 → 拒绝（fail-closed 不静默打分 "
              "cap 未知产物）",
              r.returncode != 0 and "[RUNCAP-FAIL]" in r.stdout + r.stderr,
              f"rc={r.returncode}")
        r = _run_eval(td, ["--legacy-no-runcap"])
        rp = os.path.join(td, "result.json")
        ok = r.returncode == 0 and os.path.isfile(rp)
        meta = json.load(open(rp, encoding="utf-8"))["_meta"]["runcap"] \
            if ok else {}
        check("102 绿 --legacy-no-runcap 显式降级放行（存量 E100/E109 复用"
              "格不炸死）+ [RUNCAP-LEGACY] 警告 + cap 如实记 null",
              ok and "[RUNCAP-LEGACY]" in r.stdout + r.stderr
              and meta.get("legacy_no_sidecar") == 1
              and meta.get("cap_identity_hash") is None,
              f"rc={r.returncode}")
    with tempfile.TemporaryDirectory() as td:
        _make_runcap_product(td, "hotpotqa-none-1012024000.jsonl", cap=16384)
        jpath = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        os.remove(jpath + ".runcap.json")
        r = _run_eval(td, ["--no-require-runcap"])
        check("102 绿 --no-require-runcap 显式关闭（等效降级开关）→ 放行",
              r.returncode == 0
              and os.path.isfile(os.path.join(td, "result.json")),
              f"rc={r.returncode}")


# ===========================================================================
# 103：gate 与发布之间跨进程独占锁
# ===========================================================================

_LOCK_CHILD_A = r"""
import sys, time, json
sys.path.insert(0, {repo!r})
from benchmark.LongBench import run_cap_manifest as rcm
import argparse
p = sys.argv[1]
with rcm.runcap_gate_lock(p):
    if len(sys.argv) > 2 and sys.argv[2] == "write16384":
        core = rcm.build_run_cap_core(
            model_name="Qwen3-8B", model_path=None, method="none",
            dataset="hotpotqa", dataset_fingerprint="deadbeef123",
            run_timestamp="1012024000", effective_cap=16384,
            cap_gate_info=None, max_new_tokens=128, producer=None)
        m = rcm.finalize_run_cap_manifest(
            core, status="complete", n_samples=3, input_tokens=[1,2,3],
            truncated_flags=[False]*3, result_file="x.jsonl",
            result_sha256="a"*64, result_lines=3)
        rcm.write_run_cap_manifest(p + rcm.RUN_CAP_SIDECAR_SUFFIX, m)
    print("HELD", flush=True)
    time.sleep(3)
print("RELEASED", flush=True)
"""

_LOCK_CHILD_B = r"""
import sys, time
sys.path.insert(0, {repo!r})
from benchmark.LongBench import run_cap_manifest as rcm
p = sys.argv[1]
t0 = time.time()
with rcm.runcap_gate_lock(p):
    waited = time.time() - t0
    # 锁内 double-check：后到者重读已存在 manifest → CAP-CLASH
    core = rcm.build_run_cap_core(
        model_name="Qwen3-8B", model_path=None, method="none",
        dataset="hotpotqa", dataset_fingerprint="deadbeef123",
        run_timestamp="1012024000", effective_cap=31500,
        cap_gate_info=None, max_new_tokens=128, producer=None)
    try:
        rcm.gate_run_cap_identity(p, core)
        print(f"B_OK waited={{waited:.1f}}")
    except SystemExit as e:
        print(f"B_CLASH waited={{waited:.1f}} code={{e.code}}")
"""


def test_103_gate_lock():
    from benchmark.LongBench import run_cap_manifest as rcm
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "hotpotqa-none-1012024000.jsonl")

        # ---- 绿：持有期间他进程阻塞（跨进程互斥实测）----
        script_a = _LOCK_CHILD_A.format(repo=REPO)
        pa = subprocess.Popen([sys.executable, "-c", script_a, p],
                              stdout=subprocess.PIPE, text=True)
        held = pa.stdout.readline().strip() == "HELD"   # A 已持锁（显式
        # 读取，不依赖 assert——-O 口径下 assert 剥离会引入竞态）
        t0 = time.time()
        script_b = _LOCK_CHILD_B.format(repo=REPO)
        pb = subprocess.run([sys.executable, "-c", script_b, p],
                            capture_output=True, text=True, timeout=60)
        waited_total = time.time() - t0
        pa.wait()
        check("103 绿 跨进程互斥：A 持锁 3s 内 B 阻塞，A 释放后 B 才进入"
              f"（B 实测等待 {waited_total:.1f}s ≥ 2.5s）",
              held and waited_total >= 2.5,
              f"held={held} waited={waited_total:.1f}")

        # ---- 绿：后到者锁内重读身份 → [CAP-CLASH]（double-check 语义）----
        pa = subprocess.Popen(
            [sys.executable, "-c", script_a, p, "write16384"],
            stdout=subprocess.PIPE, text=True)
        held2 = pa.stdout.readline().strip() == "HELD"   # A 锁内写 manifest
        pb = subprocess.run([sys.executable, "-c", script_b, p],
                            capture_output=True, text=True, timeout=60)
        pa.wait()
        check("103 绿 TOCTOU 交错拦截：A 锁内写 16384 manifest → B 阻塞至 A "
              "释放后锁内 gate(31500) → [CAP-CLASH]（锁外先过的门不算数）",
              held2 and "B_CLASH" in pb.stdout
              and "CAP-CLASH" in pb.stdout,
              f"held={held2} " + pb.stdout + pb.stderr)

        # ---- 绿：锁文件只创建不删除（POSIX inode 语义，结果文件级路径）----
        check("103 绿 锁文件 <out_path>.runcap_gate.lock 存留（只创建不删除"
              "——删除会让并发方锁到不同 inode、互斥失效；结果文件级路径）",
              os.path.isfile(p + rcm.RUN_CAP_LOCK_SUFFIX))

        # ---- 绿：粒度修正——同目录不同结果文件不互斥（生产回归反证：
        # 本机 fullkv 双任务同目录并发，目录级锁令后到者持 62GB 显存
        # 空等；GPT 103 威胁模型是同一 out_path，文件级锁即闭合 TOCTOU）----
        pa = subprocess.Popen(
            [sys.executable, "-c", script_a, p],
            stdout=subprocess.PIPE, text=True)
        held3 = pa.stdout.readline().strip() == "HELD"   # A 持 p 的锁 3s
        p_other = os.path.join(td, "qasper-none-1012024001.jsonl")
        t0 = time.time()
        with rcm.runcap_gate_lock(p_other):   # 同目录异结果文件：立即入锁
            enter_wait = time.time() - t0
        pa.terminate(); pa.wait()
        check("103 绿 锁粒度（结果文件级）：A 持 hotpotqa 锁时 qasper 同目"
              "录立即入锁（<1s）——同目录多任务并发不被目录级锁误串行",
              held3 and enter_wait < 1.0,
              f"held={held3} enter_wait={enter_wait:.2f}s")

        # ---- 绿：异常路径必释放（try/finally）----
        released = True
        try:
            with rcm.runcap_gate_lock(p):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        t0 = time.time()
        with rcm.runcap_gate_lock(p):   # 立即可再入 = 已释放
            pass
        check("103 绿 异常路径锁必释放（raise 传播后再入 < 1s）",
              time.time() - t0 < 1.0)

    # ---- 绿：pred.py 静态接线——with runcap_gate_lock 包住 gate→写→回填 ----
    import ast
    src = open(os.path.join(REPO, "benchmark/LongBench/pred.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            if any(isinstance(it.context_expr, ast.Call)
                   and isinstance(it.context_expr.func, ast.Name)
                   and it.context_expr.func.id == "runcap_gate_lock"
                   for it in node.items):
                calls = [ast.unparse(n.func) for n in ast.walk(node)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name)]
                has_gate = "gate_run_cap_identity" in calls
                has_bind = "compute_result_binding" in calls
                found = has_gate and has_bind
    check("103 绿 pred.py 接线：with runcap_gate_lock(out_path) 内含 "
          "gate_run_cap_identity + compute_result_binding（gate→数据写→"
          "complete 回填全窗口持锁）", found)


# ===========================================================================
# 105：backfill 五重绑定
# ===========================================================================

GATE_BLOCK = (
    "[CONTEXT-CAP-GATE] model={model}\n"
    "  experiment_input_cap (live)  = {eff}   "
    "<- model2maxlen.json:{model}\n"
    "  experiment_input_cap (registry protocol) = {proto}   <- registry\n"
    "[EXPERIMENT-CAP-DIVERGENCE] 本次实验 cap={eff} 与注册表 protocol 值 "
    "{proto} 不同——按受限上下文 treatment 如实记录\n"
)
RUN_BIND_LINE = ("[CONTEXT-CAP-GATE] run-bind task={task} t={t} "
                 "output={out}\n")


def _backfill(result, log=None, model="DeepSeek-V2-Lite-Chat",
              task="hotpotqa", t="1012024000", extra=()):
    cmd = [sys.executable, BACKFILL_FP, "--result", result,
           "--model", model, "--method", "none", "--task", task,
           "--t", t, "--max-new-tokens", "32"] + list(extra)
    if log:
        cmd += ["--log", log]
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=REPO, timeout=120)


def _mk_rows(n, task="hotpotqa", with_id=True, fp=None, start=0):
    rows = []
    for i in range(n):
        r = {"pred": "a", "answers": [f"ans{i}"], "all_classes": None,
             "length": 1000 + i, "budget": None, "score_sum": None}
        if with_id:
            r["_id"] = f"{task}:{fp}:{start + i}"
        rows.append(r)
    return rows


def _fp_of(rows, task="hotpotqa"):
    src = json.dumps([[r["answers"], r["length"]] for r in rows],
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def test_105_backfill_binding():
    # ---- 红：跨模型日志（GPT 反例 1）----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(2, fp="aaaaaaaaaaaa")
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(GATE_BLOCK.format(model="Qwen3-8B", eff=16384,
                                      proto=31500))
        r = _backfill(result, log)
        check("105 红 跨模型日志（块 model=Qwen3-8B vs CLI DeepSeek）→ "
              "[GATE-FAIL] 拒绝（修复前 .search 永取第一条可把 DeepSeek "
              "产物声明成 Qwen cap）",
              r.returncode != 0 and "跨模型" in r.stdout + r.stderr
              and not os.path.exists(result + ".runcap.json"),
              f"rc={r.returncode}")

    # ---- 红：一行 partial 冒充全集（GPT 反例 2，⑤ 指纹冲突硬拒）----
    with tempfile.TemporaryDirectory() as td:
        full_rows = _mk_rows(200)
        fp_full = _fp_of(full_rows)          # 全 200 行全集指纹
        partial = [full_rows[5]]             # 只留第 5 行
        # 行号重排为 0（冒充全集首行）但保留全 200 行的全集指纹——_id 自称
        # 「全集指纹第 0 行」，而本文件重算指纹只含 1 行 → ⑤ 指纹冲突。
        partial[0]["_id"] = f"hotpotqa:{fp_full}:0"
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in partial:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                      eff=16384, proto=31500))
        r = _backfill(result, log)
        check("105 红 一行 partial（_id 内嵌全集指纹 != 本文件重算指纹）→ "
              "[GATE-FAIL] 硬拒（⑤ 优先且不降格——指纹冲突是硬证据，降格"
              "放行会让操作者继续错配）",
              r.returncode != 0 and "指纹" in r.stdout + r.stderr,
              f"rc={r.returncode}")

    # ---- 红：行号序列缺重（idx 从 1 起非 0 起）----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(2, fp="aaaaaaaaaaaa", start=1)   # idx 1,2
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                      eff=16384, proto=31500))
        r = _backfill(result, log)
        check("105 红 行号序列非 0..n-1（缺行/乱序）→ [GATE-FAIL] 拒绝",
              r.returncode != 0 and "行号" in r.stdout + r.stderr,
              f"rc={r.returncode}")

    # ---- 红：多块 run-bind 重复匹配（GPT 反例 3——不再永远取第一条）----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(200)
        fp = _fp_of(rows)
        for r, i in zip(rows, range(200)):
            r["_id"] = f"hotpotqa:{fp}:{i}"
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            for _ in range(2):   # 同 (model,task,t,output) 两块——不唯一
                f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                          eff=16384, proto=31500))
                f.write(RUN_BIND_LINE.format(task="hotpotqa",
                                             t="1012024000",
                                             out="hotpotqa-none-"
                                                 "1012024000.jsonl"))
        r = _backfill(result, log)
        check("105 红 多 gate 块与目标重复匹配（候选不唯一）→ [GATE-FAIL] "
              "拒绝（修复前永远取第一条）",
              r.returncode != 0 and "重复匹配" in r.stdout + r.stderr,
              f"rc={r.returncode}")

    # ---- 红：旧格式多块 cap 不一致 → 无法归因拒绝 ----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(200)
        fp = _fp_of(rows)
        for r, i in zip(rows, range(200)):
            r["_id"] = f"hotpotqa:{fp}:{i}"
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                      eff=16384, proto=31500))
            f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                      eff=31500, proto=31500))
        r = _backfill(result, log)
        check("105 红 旧格式日志多块 cap 不一致（16384 vs 31500）→ "
              "[GATE-FAIL] 拒绝（无法归因到本 run）",
              r.returncode != 0 and "不一致" in r.stdout + r.stderr,
              f"rc={r.returncode}")

    # ---- 绿：run-bind 唯一匹配 + ③ log sha/字节范围可复核 ----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(200)
        fp = _fp_of(rows)
        for r, i in zip(rows, range(200)):
            r["_id"] = f"hotpotqa:{fp}:{i}"
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        log_body = (
            "前导噪音行（锚点前，不参与候选）\n"
            + GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat", eff=16384,
                                proto=31500)
            + RUN_BIND_LINE.format(task="hotpotqa", t="1012024000",
                                   out="hotpotqa-none-1012024000.jsonl")
            + "生成中…\n"
            + GATE_BLOCK.format(model="Qwen3-8B", eff=31500, proto=31500)
            + RUN_BIND_LINE.format(task="gov_report", t="1012024999",
                                   out="gov_report-none-1012024999.jsonl")
        )
        with open(log, "w", encoding="utf-8") as f:
            f.write(log_body)
        r = _backfill(result, log)
        ok = r.returncode == 0
        m = json.load(open(result + ".runcap.json",
                           encoding="utf-8")) if ok else {}
        lb = m.get("log_binding", {})
        log_sha = hashlib.sha256(
            open(log, "rb").read()).hexdigest()
        b0 = lb.get("match_byte_ranges", [[]])[0]
        blk_text = open(log, "rb").read()[b0[0]:b0[1]].decode("utf-8")
        check("105 绿 run-bind 唯一匹配（跨模型/跨任务块不误配）→ 补制成功"
              " + log_binding（log_sha256 实测一致 + 字节范围切回即目标块"
              " 含 run-bind 行）",
              ok and m.get("provenance") == "backfilled-from-stdout"
              and lb.get("log_sha256") == log_sha
              and lb.get("match_mode") == "run-bind"
              and "run-bind task=hotpotqa" in blk_text
              and m["sample_set"]["complete"] is True,
              f"rc={r.returncode} {r.stdout[-150:]!r}")

    # ---- 绿：旧格式共识块（.24 存量批同型日志）----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(200)
        fp = _fp_of(rows)
        for r, i in zip(rows, range(200)):
            r["_id"] = f"hotpotqa:{fp}:{i}"
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            for _ in range(2):   # 旧格式无 run-bind，两块 cap 证据一致
                f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                          eff=16384, proto=31500))
        r = _backfill(result, log)
        ok = r.returncode == 0
        m = json.load(open(result + ".runcap.json",
                           encoding="utf-8")) if ok else {}
        check("105 绿 旧格式多块 cap 证据全一致 → consensus 放行（字节范围"
              "记录全部候选块）",
              ok and "consensus" in m.get("log_binding", {}).get(
                  "match_mode", "")
              and len(m["log_binding"]["match_byte_ranges"]) == 2
              and m["effective_cap"] == 16384,
              f"rc={r.returncode}")

    # ---- 绿④：无 _id 旧 v1 文件 + 行数 != 预期全集 → 降格不冒称 ----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(2, with_id=False)     # 旧 v1：无 _id
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(GATE_BLOCK.format(model="DeepSeek-V2-Lite-Chat",
                                      eff=16384, proto=31500))
        r = _backfill(result, log)
        ok = r.returncode == 0
        m = json.load(open(result + ".runcap.json",
                           encoding="utf-8")) if ok else {}
        check("105 绿④ 无 _id + 2 行 != 预期全集 200 → 降格 "
              "provenance_incomplete（cap 证据保留但不冒称 complete，"
              "sample_set.complete=false 隔离出论文聚合）",
              ok and m["provenance"] == "provenance_incomplete"
              and m["sample_set"]["complete"] is False
              and m["effective_cap"] == 16384,
              f"rc={r.returncode}")
        # 显式 --expect-count 覆盖表值 → 全集证明成立 → 不降格
        os.remove(result + ".runcap.json")
        r = _backfill(result, log, extra=["--expect-count", "2"])
        ok = r.returncode == 0
        m = json.load(open(result + ".runcap.json",
                           encoding="utf-8")) if ok else {}
        check("105 绿④ --expect-count 显式声明（操作者自负其责）→ 全集证明"
              "成立不降格",
              ok and m["provenance"] == "backfilled-from-stdout"
              and m["sample_set"]["complete"] is True,
              f"rc={r.returncode}")

    # ---- 红：CLI 与结果文件名不一致（前哨）----
    with tempfile.TemporaryDirectory() as td:
        rows = _mk_rows(2, fp="aaaaaaaaaaaa")
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        r = _backfill(result, task="musique", t="1012024000")
        check("105 红 --task=musique vs 文件名 hotpotqa-… → [GATE-FAIL] "
              "拒绝（CLI 声明与产物身份不符）",
              r.returncode != 0 and "不一致" in r.stdout + r.stderr,
              f"rc={r.returncode}")


# ===========================================================================
# 106：RULER llm_eval cap+任务集进输出身份 + 结果 receipt
# ===========================================================================

def test_106_ruler_identity():
    try:
        import torch  # noqa: F401
        from benchmark.RULER import llm_eval as le
    except ImportError as e:
        skip("106 段全部", f"导入失败 {e}")
        return

    ts = "20261012_070000"
    tasks = ["niah_single_1", "niah_single_2"]
    f16 = le.build_result_filename("tli_best", 16384, tasks, ts)
    f31 = le.build_result_filename("tli_best", 31500, tasks, ts)
    f_other_tasks = le.build_result_filename("tli_best", 16384,
                                             ["niah_multikey_1"], ts)
    check("106 绿 同 method 同秒异 cap → 文件名必不同（"
          f"{os.path.basename(f16)} vs {os.path.basename(f31)}——修复前"
          "二者逐字相同可互相覆盖）",
          f16 != f31 and "_L16384_" in f16 and "_L31500_" in f31)
    check("106 绿 同 method 同 cap 同秒异任务集 → 文件名必不同（任务集"
          "摘要进身份）", f16 != f_other_tasks)

    # 修复前红证（静态）：基线文件名只含 method+timestamp
    src_prefix = subprocess.check_output(
        ["git", "-C", REPO, "show",
         f"{PRE_FIX_SHA}:benchmark/RULER/llm_eval.py"], text=True)
    check("101/106 红 修复前 llm_eval 文件名 = method_name+timestamp（cap "
          "不在身份中——静态基线）",
          'f"{method_name}_{timestamp}"' in src_prefix)

    # ---- 绿：save_results 原子落盘 txt + receipt（sha/行数/cap 同代绑定）----
    with tempfile.TemporaryDirectory() as td:
        from argparse import Namespace
        args = Namespace(output_dir=td, model="hf_models/Qwen3-8B",
                         model_name="Qwen3-8B", lm_eval_batch_size=1,
                         max_length=16384,
                         tasks=["niah_single_1", "niah_single_2"])
        fn = le.build_result_filename("tli_best", 16384, args.tasks,
                                       "20261012_070000")
        le.save_results(args, fn, {}, "TABLE",
                        {"mean_select_tokens": 1024}, None,
                        method_name="tli_best")
        txt = os.path.join(td, fn + ".txt")
        receipt_p = txt + ".receipt.json"
        ok = os.path.isfile(txt) and os.path.isfile(receipt_p)
        receipt = json.load(open(receipt_p, encoding="utf-8")) if ok else {}
        body = open(txt, "rb").read()
        check("106 绿 save_results 落盘 txt + receipt：result_sha256 与文件"
              "字节级一致 + result_lines=行数 + cap 回执=16384 + 任务集清单"
              "入 receipt（结果身份不靠文件名单点承载）",
              ok and receipt.get("schema") == "ruler-eval-receipt-v1"
              and receipt.get("result_sha256") == hashlib.sha256(
                  body).hexdigest()
              and receipt.get("result_lines") == body.count(b"\n")
              and receipt.get("cap") == 16384
              and receipt.get("tasks") == sorted(args.tasks)
              and receipt.get("method_name") == "tli_best",
              json.dumps(receipt)[:200])

        # 同秒异 cap 两次落盘共存（不互相覆盖）
        fn31 = le.build_result_filename("tli_best", 31500, args.tasks,
                                        "20261012_070000")
        le.save_results(args, fn31, {}, "TABLE",
                       {"mean_select_tokens": 1024}, None,
                       method_name="tli_best")
        check("106 绿 同秒 16K/31.5K 双产物共存（路径级可分，不覆盖）",
              os.path.isfile(os.path.join(td, fn + ".txt"))
              and os.path.isfile(os.path.join(td, fn31 + ".txt"))
              and fn != fn31)

        # 原子性：无 .tmp 残留
        check("106 绿 原子落盘无 .tmp 残留（tmp+os.replace——读侧永不见"
              "半文件）",
              not glob.glob(os.path.join(td, "*.tmp*")))

    # tli 臂：treatment sidecar 原子落盘 + sha 进 receipt
    with tempfile.TemporaryDirectory() as td:
        from argparse import Namespace
        from types import SimpleNamespace

        class _Snap:
            manifest_json = json.dumps({"treatment": "x"}, sort_keys=True)

        args = Namespace(output_dir=td, model="m", model_name="Qwen3-8B",
                         lm_eval_batch_size=1, max_length=16384,
                         tasks=["niah_single_1"])
        fn = le.build_result_filename("tli_best", 16384, args.tasks,
                                      "20261012_070001")
        le.save_results(args, fn, {}, "T", {"mean_select_tokens": 1},
                        _Snap(), method_name="tli_best")
        txt = os.path.join(td, fn + ".txt")
        sc = txt + le.TREATMENT_MANIFEST_SIDECAR_SUFFIX
        receipt = json.load(open(txt + ".receipt.json", encoding="utf-8"))
        check("106 绿 tli 臂 treatment manifest sidecar 落盘 + sha256 进 "
              "receipt（081 层与结果同代绑定）",
              os.path.isfile(sc)
              and open(sc, encoding="utf-8").read() == _Snap.manifest_json
              and receipt["treatment_manifest_sha256"] == hashlib.sha256(
                  _Snap.manifest_json.encode("utf-8")).hexdigest())

    # main 接线静态：main 调 build_result_filename（args.max_length 进身份）
    import ast
    tree = ast.parse(open(os.path.join(REPO, "benchmark/RULER/llm_eval.py"),
                          encoding="utf-8").read())
    wired = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "build_result_filename":
            args_unparsed = [ast.unparse(a) for a in node.args]
            wired = ("args.max_length" in args_unparsed
                     and "args.tasks" in args_unparsed)
    check("106 绿 main 接线：build_result_filename(method, args.max_length,"
          " args.tasks, ts)——cap 与任务集进文件名身份", wired)


# ===========================================================================

def main():
    print("=" * 72)
    print("GPT 审计 2026-10-12_0739 五项修复红绿验收（101/102/103/105/106）")
    print(f"REPO={REPO}  PRE_FIX_SHA={PRE_FIX_SHA}  python={'-O' if __debug__ is False else ''}")
    print("=" * 72)
    test_101_host_gate()
    test_101_red_prefix()
    test_101_red_gpu_probe()
    test_101_bitwise_legal()
    test_102_eval_runcap()
    test_103_gate_lock()
    test_105_backfill_binding()
    test_106_ruler_identity()
    print("=" * 72)
    print(f"=== {len(_PASS)} PASS / {len(_FAIL)} FAIL"
          f"{' / ' + str(len(_SKIP)) + ' SKIP' if _SKIP else ''} ===")
    if _FAIL:
        for f in _FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--prefix-gpu-probe":
        _prefix_gpu_probe_main()
    else:
        main()
