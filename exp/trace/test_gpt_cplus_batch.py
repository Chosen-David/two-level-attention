# -*- coding: utf-8 -*-
"""GPT 审计 C+ 批红绿测试（142 并发残余 + 149 同构缺口）。

覆盖审计项（修复 = 绿；红证 = 同一脚本复制进基线 SHA e37e5b5 树跑出
[TEST-FAIL]，本脚本按「红证可在基线树运行」设计——基线上 stage 仍写
仓库根共享固定路径，142 并发/旧路径/stale 断言以显式判定失败）：

  142 并发残余（TL-FASA-CALIBRATION-IDENTITY，GPT advice C+ 批）
    缺陷：stage 物化目标是仓库根共享固定路径
          dimension_rankings/<model>/<dataset>_agreement.pkl——两个不同
          内容的源并发首次 stage 同一目标时，双方都通过 isfile 不存在
          检查，A 先 os.replace、B 后 os.replace 覆盖 A → A 进程
          name/snapshot 绑 A、官方 loader 却读到 B（原子替换 ≠ 内容
          归属锁定；/tmp/repro_fasa_142.py 双进程调度复现）。
    绿：  物化目标 = 内容寻址进程私有目录
          dimension_rankings_staged/<full_sha256>/dimension_rankings/
          <model>/...
          ——双进程并发 stage 不同内容各自三方同源（name/snapshot/
          runtime_file 一致，旧共享路径下即错绑形态）；同内容并发
          幂等无害；旧仓库根 dimension_rankings/ 不删不动不读；
          私有目标被篡改仍 fail-closed（强度不降）。
          （P2 追修：目录段由 sha8 前缀升级为完整 SHA256——sha8 前 8
          hex 碰撞时两个不同内容共享同一目录、并发首次 stage 后写
          覆盖先写；sha8 只作 treatment 名显示标识。旧 sha8 段 staged
          残骸不删不动不读。）
  149 同构残余（TL-MINFERENCE-PATTERN-IDENTITY，GPT advice C+ 批）
    缺陷：freeze 只把 sha8 挂 args，adapter（enable_minference）在
          enable 时重新 open config_path 载入 JSON——窗口内源被替换
          则 name 绑 A、runtime 读 B（与 142 同构）。
    绿：  freeze 升级为完整冻结对象（sha8 + raw + 解析后 JSON）挂
          args._minference_pattern_frozen；adapter 消费冻结对象构建
          best_pattern（源文件删除后 enable 仍成功 = 证明不再 open）；
          降级路径 config 缺失显式 SystemExit（-O 下同型）；畸形
          JSON 在 freeze 期 fail-closed（C 批套件覆盖）。
  152 SKIP 汇总（TL-CPLUS-SKIP-SUCCESS-152，P2）
    缺陷：test_149_adapter_frozen_consumption 依赖导入失败时 print
          SKIP + return（无状态返回），main() 无条件打印
          RESULT: ALL PASS 且 return 0——required adapter 未执行也
          宣称全项通过。
    绿：  各测试函数显式返回 "PASS"/"SKIP"；main 汇总——全 PASS 才
          ALL PASS rc=0；有 SKIP 打 RESULT: PARTIAL (SKIP: <项>)
          rc=3。测试内 _check 失败保持 SystemExit [TEST-FAIL]
          （异常即 FAIL，rc=1 由解释器/SystemExit 消息字符串透传）。

用法（本仓根，普通与 -O 双跑，两者结果一致才算绿）：
  PYTHONPATH=<repo> python3 exp/trace/test_gpt_cplus_batch.py
  PYTHONPATH=<repo> python3 -O exp/trace/test_gpt_cplus_batch.py

不依赖 GPU（142 用 AST 提取官方 load_pickle 规避 fasa_utils 模块级
.to("cuda")；149 adapter 测试用 Qwen3Attention.__new__ 空壳 + 冻结
对象，不触 CUDA kernel）。149 adapter 段需 vendored minference 可导入
（缺编译产物 .so 时如实 SKIP，同 B 批套件纪律——但 152 起 SKIP 必须
以 PARTIAL rc=3 汇总，不得宣称 ALL PASS）。
"""
import ast
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import tempfile
import types
from pathlib import Path

# 仓库根 = 脚本位于 <repo>/exp/trace/ 上两级（字面绝对路径，不用 $PWD）
REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

from sparse_attn.info import (  # noqa: E402
    get_method_name_with_info, stage_fasa_agreement,
    freeze_minference_pattern_sha8,
)


def _check(cond, msg=""):
    """与 A/B/C 批套件同纪律：显式判定（非 assert），-O 下不失效。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _extract_vendor_fn(relative, names, globals_extra):
    """AST 提取 vendor 源内指定函数（规避 fasa_utils 模块级 .to("cuda")，
    与 /tmp/repro_fasa_142.py 同手法——不执行模块级副作用）。"""
    path = Path(REPO) / relative
    nodes = [x for x in ast.parse(path.read_text()).body
             if isinstance(x, ast.FunctionDef) and x.name in names]
    if {x.name for x in nodes} != set(names):
        raise RuntimeError(f"missing source function: {names}")
    mod = types.ModuleType("_probe_" + hashlib.sha256(
        relative.encode()).hexdigest()[:8])
    mod.__file__ = str(path)
    mod.__dict__.update(globals_extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(path), "exec"), mod.__dict__)
    return mod


def _fa_args(agr_dir, dataset="qasper", model="Qwen3-8B"):
    """最小 fasa args namespace（142 消费字段全集，与 C 批套件同款）。"""
    import argparse
    ns = argparse.Namespace()
    ns.method = "fasa_vendor"
    ns.fasa_agreement_dir = agr_dir
    ns.fasa_dataset = dataset
    ns.model = model
    ns.fasa_budget = 1024
    ns.fasa_threshold_ratio = 0.8
    ns.fasa_min_k = 16
    ns.fasa_max_k = 16
    ns.fasa_mode = "topk"
    return ns


# ================================================================ 142 并发残余
def _cp_worker(payload, expected_obj, repo, pipe):
    """C+ 批 142 并发绿证 worker：stage → adapter 语义读官方副本 → 回报。

    完整复刻 qwen3_fasa_patch.enable_fasa 的消费方式：chdir 到
    stage_fasa_agreement 返回的 staged 根，官方 load_pickle 用相对路径
    dimension_rankings/<model>/<dataset>_agreement.pkl 读内容寻址副本。
    任何一步异常都以 error 行回报（不静默）。"""
    role, src_dir = payload
    try:
        # 官方 loader：AST 提取（模块级 .to("cuda") 副作用不可整 import）
        loader = _extract_vendor_fn(
            "exp/vendor_baselines/fasa_vendor/fasa_utils.py",
            ["load_pickle"], {"pickle": pickle})
        args = _fa_args(src_dir)
        staged_root = stage_fasa_agreement(args, repo_root=repo)
        snap = getattr(args, "_fasa_staged_snapshot")
        name = get_method_name_with_info(args)
        # adapter 语义：CWD 切到 staged 根（try/finally 恢复）
        cwd_bak = os.getcwd()
        os.chdir(staged_root)
        try:
            runtime_obj = loader.load_pickle(
                "dimension_rankings/Qwen3-8B/qasper_agreement.pkl")
        finally:
            os.chdir(cwd_bak)
        runtime_bytes = Path(snap.dst_path).read_bytes()
        pipe.send({
            "role": role,
            "staged_root": staged_root,
            "dst_path": snap.dst_path,
            "name": name,
            "snap_sha256": snap.sha256,
            "snap_sha8": snap.sha8,
            "runtime_file_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
            "runtime_obj": runtime_obj,
            "expected_obj": expected_obj,
        })
    except BaseException as exc:  # noqa: BLE001——异常如实回传
        pipe.send({"role": role, "error": type(exc).__name__,
                   "message": str(exc)})
    finally:
        pipe.close()


def _run_two_workers(td, content_a, content_b, repo_name):
    """起 A/B 两个进程并发 stage（真实并发，无交错控制——内容寻址下
    时序无关），返回两个结果行。"""
    ctx = multiprocessing.get_context("fork")
    base = Path(td)
    repo = base / repo_name
    repo.mkdir(parents=True, exist_ok=True)
    hashes, payloads, expected = {}, {}, {}
    for role, content in (("A", content_a), ("B", content_b)):
        src_dir = base / f"{repo_name}_src_{role}"
        (src_dir / "Qwen3-8B").mkdir(parents=True, exist_ok=True)
        (src_dir / "Qwen3-8B" / "qasper_agreement.pkl").write_bytes(content)
        hashes[role] = hashlib.sha256(content).hexdigest()
        payloads[role] = (role, str(src_dir))
        # pickle round-trip 对象（worker 内官方 load_pickle 的期望值）
        expected[role] = pickle.loads(content)
    processes, pipes, rows = [], [], []
    try:
        for role in ("A", "B"):
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_cp_worker,
                args=(payloads[role], expected[role], str(repo), child))
            proc.start()
            child.close()
            processes.append(proc)
            pipes.append(parent)
        for pipe in pipes:
            if not pipe.poll(30):
                raise RuntimeError("C+ worker deadline exceeded")
            rows.append(pipe.recv())
        for proc in processes:
            proc.join(5)
            if proc.is_alive() or proc.exitcode != 0:
                raise RuntimeError("C+ worker failed")
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(2)
        for pipe in pipes:
            pipe.close()
    return rows, hashes, repo


def _assert_three_way(row, own_sha, own_expected_obj, tag):
    """三方同源断言：name 尾段 / 快照 SHA / runtime 文件 SHA / 官方
    loader 读回对象 全部等于本进程自己的内容。"""
    _check("error" not in row, f"{tag}: worker 异常: {row}")
    _check(row["snap_sha256"] == own_sha and
           row["runtime_file_sha256"] == own_sha,
           f"{tag}: 快照/runtime 文件 SHA 必须等于本方内容: {row}")
    _check(row["snap_sha8"] == own_sha[:8] and
           row["name"].endswith("_agr" + own_sha[:8]),
           f"{tag}: 名字尾段必须等于本方 sha8: {row['name']}")
    _check(row["runtime_obj"] == own_expected_obj,
           f"{tag}: 官方 loader 读回对象必须等于本方内容（读错即错绑）")


def test_142_concurrent_content_addressed():
    """双进程并发 stage 不同内容到同一 repo：各自三方同源（红 = 旧共享
    固定路径下 A 的 runtime_file 读到 B 的内容）。"""
    content_a = pickle.dumps(
        {"owner": "A", "vals": list(range(64))}, protocol=4)
    content_b = pickle.dumps(
        {"owner": "B", "vals": list(range(63, -1, -1))}, protocol=4)
    _check(content_a != content_b, "前置：A/B 内容必须不同")
    with tempfile.TemporaryDirectory() as td:
        rows, hashes, repo = _run_two_workers(
            td, content_a, content_b, "repo_diff")
        _check(len(rows) == 2, f"前置：双 worker 结果应齐全: {rows}")
        row_by_role = {r["role"]: r for r in rows}
        _assert_three_way(row_by_role["A"], hashes["A"],
                          pickle.loads(content_a), "142A")
        _assert_three_way(row_by_role["B"], hashes["B"],
                          pickle.loads(content_b), "142B")
        # 内容寻址隔离：A/B staged 根不同（full_sha256 目录段），互不覆盖
        _check(row_by_role["A"]["staged_root"] !=
               row_by_role["B"]["staged_root"],
               "142: 不同内容必须隔离到不同 full_sha256 staged 目录")
        _check(os.path.basename(row_by_role["A"]["staged_root"]) ==
               hashes["A"] and
               os.path.basename(row_by_role["B"]["staged_root"]) ==
               hashes["B"],
               "142: staged 根尾段必须等于各自内容完整 sha256"
               f"（P2 追修：sha8 前缀碰撞残余——目录段不得用 sha8）")
        # 旧仓库根共享固定路径绝不存在（不再被写）
        _check(not os.path.exists(os.path.join(str(repo),
                                               "dimension_rankings")),
               "142: 并发 stage 后仓库根共享固定路径必须不存在（不写共享"
               "路径即错绑窗口关闭）")
    print("142 并发不同内容 PASS  双进程并发 stage 不同内容 → 各自"
          "name/快照/runtime 文件/官方 loader 读回 四方同源；staged 根按"
          "完整 sha256 隔离互不覆盖（sha8 只作名字显示标识）；仓库根共享"
          "固定路径不被写（错绑窗口结构性关闭）")
    return "PASS"


def test_142_concurrent_same_content():
    """双进程并发 stage 同一内容：同一路径幂等（内容逐字节相同，
    os.replace 无害），双方三方同源。"""
    content = pickle.dumps({"owner": "SAME", "vals": list(range(32))},
                          protocol=4)
    with tempfile.TemporaryDirectory() as td:
        rows, hashes, repo = _run_two_workers(
            td, content, content, "repo_same")
        row_by_role = {r["role"]: r for r in rows}
        _assert_three_way(row_by_role["A"], hashes["A"],
                          pickle.loads(content), "142 同内容 A")
        _assert_three_way(row_by_role["B"], hashes["B"],
                          pickle.loads(content), "142 同内容 B")
        _check(row_by_role["A"]["staged_root"] ==
               row_by_role["B"]["staged_root"],
               "142 同内容: 同内容双进程应命中同一 staged 路径（幂等）")
        _check(row_by_role["A"]["dst_path"] == row_by_role["B"]["dst_path"],
               "142 同内容: 物化目标应同路径")
    print("142 并发同内容 PASS  双进程并发 stage 同一内容 → 同一路径"
          "幂等（os.replace 逐字节相同无害），双方三方同源")
    return "PASS"


def test_142_legacy_shared_path_untouched():
    """旧仓库根 dimension_rankings/ 历史产物：不删不动不读，stage 正常
    成功且物化在私有目录。"""
    with tempfile.TemporaryDirectory() as td:
        legacy_bytes = b"LEGACY-SHARED-PATH-ARTIFACT" * 16
        repo = Path(td) / "repo_legacy"
        legacy = repo / "dimension_rankings" / "Qwen3-8B" / \
            "qasper_agreement.pkl"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(legacy_bytes)
        agr_dir = Path(td) / "src"
        (agr_dir / "Qwen3-8B").mkdir(parents=True)
        content = b"NEW-CALIBRATION" * 32
        (agr_dir / "Qwen3-8B" / "qasper_agreement.pkl").write_bytes(content)

        args = _fa_args(str(agr_dir))
        staged_root = stage_fasa_agreement(args, str(repo))
        # 旧路径内容逐字节不变（不删不动不读——历史产物原地保留）
        _check(legacy.read_bytes() == legacy_bytes,
               "142: 旧仓库根共享路径历史产物必须原样保留")
        # 新物化在私有目录，内容 = 源
        dst = getattr(args, "_fasa_staged_snapshot").dst_path
        with open(dst, "rb") as f:
            _check(f.read() == content,
                   "142: 物化目标内容应等于源（私有目录）")
        _check(dst.startswith(os.path.join(str(staged_root),
                                            "dimension_rankings") + os.sep),
               f"142: 物化目标应位于 staged 根下 dimension_rankings/ 内: {dst}")
    print("142 旧共享路径 PASS  仓库根 dimension_rankings/ 历史产物不删"
          "不动不读；新 stage 物化到 dimension_rankings_staged/<full_sha256>/"
          " 私有目录（官方 loader 的 CWD 由 adapter 切走，旧路径不再"
          "可达；旧 sha8 段残骸同样不删不动不读）")
    return "PASS"


def test_142_stale_private_target():
    """私有目标被篡改/损坏：SHA 全量比对 fail-closed（B 批 stale 语义
    在新路径下强度不降）。"""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo_stale"
        repo.mkdir()
        agr_dir = Path(td) / "src"
        (agr_dir / "Qwen3-8B").mkdir(parents=True)
        content = b"STALE-TEST-CONTENT" * 8
        (agr_dir / "Qwen3-8B" / "qasper_agreement.pkl").write_bytes(content)
        args = _fa_args(str(agr_dir))
        stage_fasa_agreement(args, str(repo))
        dst = getattr(args, "_fasa_staged_snapshot").dst_path
        # 篡改私有目标（模拟磁盘损坏/外部改动）
        with open(dst, "wb") as f:
            f.write(b"TAMPERED" * 8)
        try:
            stage_fasa_agreement(_fa_args(str(agr_dir)), str(repo))
            raise AssertionError(
                "142: 篡改后的私有目标未被拒绝（stale 强度下降）")
        except SystemExit as e:
            _check("142" in str(e),
                   f"142: 篡改目标报错应带编号: {e}")
    print("142 stale 强度 PASS  私有目标被篡改仍 SystemExit fail-closed"
          "（SHA 全量比对语义在新路径下保持）")
    return "PASS"


# ================================================================ 149 同构残余
def test_149_adapter_frozen_consumption():
    """adapter 消费冻结对象：freeze 后源文件删除，enable_minference 仍
    成功且 runtime best_pattern = 冻结内容（红 = 旧实现二次 open 已删除
    的 config_path 必 FileNotFoundError / 读到被替换的 B）。"""
    vendor_dir = os.path.join(
        REPO, "exp", "vendor_baselines", "minference_vendor")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    try:
        import qwen3_minference_patch as minf_patch  # noqa: F401
        from qwen3_minference_patch import enable_minference
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    except Exception as e:  # noqa: BLE001——缺 vendored 依赖时如实 SKIP
        # 152：SKIP 必须显式返回状态给 main 汇总——旧版 print 后裸
        # return，main 无条件 ALL PASS rc=0，required adapter 未执行
        # 也宣称全项通过。SKIP ≠ PASS，汇总层打 PARTIAL rc=3。
        print(f"[CPLUS-149] SKIP: minference_vendor 依赖不可导入（{e}）")
        return "SKIP"

    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "best_pattern.json")
        pattern_a = {"0": {"0": ["vertical_and_slash", 1, 10000],
                           "1": ["vertical_and_slash", 2, 20000]}}
        with open(cfg, "wb") as f:
            f.write(json.dumps(pattern_a).encode())
        args = types.SimpleNamespace(
            method="minference_vendor",
            minference_config_path=cfg,
            minference_starting_layer=-1)
        freeze_minference_pattern_sha8(args)
        frozen = getattr(args, "_minference_pattern_frozen", None)
        _check(frozen is not None,
               "149: freeze 未挂完整冻结对象（_minference_pattern_frozen）")
        _check(frozen.pattern == pattern_a,
               "149: 冻结对象 pattern 应等于源 JSON 解析结果")

        # 红证形态核心：freeze 后删除源——adapter 若二次 open 必炸
        os.remove(cfg)
        # 空壳 Qwen3Attention（不触 CUDA kernel，只要 isinstance 可挂载）
        attn = Qwen3Attention.__new__(Qwen3Attention)
        attn.config = types.SimpleNamespace(num_attention_heads=32,
                                            num_key_value_heads=8)
        attn.forward = lambda *a, **k: None
        model = types.SimpleNamespace(
            _modules={"attn": attn},
            named_modules=lambda: [("attn", attn)])

        n_patched = enable_minference(
            model, is_search=False, config_path=cfg,
            starting_layer=-1, frozen_pattern=frozen.pattern)
        _check(n_patched == 1, f"149: 空 model 应 patch 1 个 attn: {n_patched}")
        runtime_cfg = getattr(attn, "_minference_attn_config", None)
        _check(runtime_cfg is not None,
               "149: enable 后 attn 应挂 _minference_attn_config")
        _check(runtime_cfg["best_pattern"] == pattern_a,
               "149: runtime best_pattern 必须等于冻结内容 A（错绑即红）")
        _check(runtime_cfg["is_search"] is False,
               "149: is_search=False 口径")

        # 降级路径（未经入口 freeze 直接调 adapter）：config 缺失显式
        # SystemExit（非 assert——python -O 下同型）
        try:
            enable_minference(model, is_search=False, config_path=cfg,
                              starting_layer=-1)
            raise AssertionError("149: 降级路径 config 缺失未 fail-closed")
        except SystemExit as e:
            _check("149" in str(e),
                   f"149: 降级路径报错应带编号: {e}")

    print("149 adapter 冻结消费 PASS  freeze 完整冻结对象（sha8 + raw + "
          "解析后 JSON）；源文件删除后 enable_minference 仍成功且 runtime "
          "best_pattern = 冻结内容（不再二次 open 可变 config_path）；降级"
          "路径 config 缺失显式 SystemExit（-O 下同型）")
    return "PASS"


def main():
    # 152（TL-CPLUS-SKIP-SUCCESS-152）：汇总必须区分 PASS/SKIP——
    # required adapter（149）未执行时不得宣称全项通过。各测试显式返回
    # "PASS"/"SKIP"；_check 失败仍走 SystemExit [TEST-FAIL]（异常即
    # FAIL，不进入汇总层）。全 PASS 才 ALL PASS rc=0；有 SKIP 打
    # PARTIAL + SKIP 项列表、rc=3。
    tests = [
        ("142_concurrent_content_addressed",
         test_142_concurrent_content_addressed),
        ("142_concurrent_same_content",
         test_142_concurrent_same_content),
        ("142_legacy_shared_path_untouched",
         test_142_legacy_shared_path_untouched),
        ("142_stale_private_target",
         test_142_stale_private_target),
        ("149_adapter_frozen_consumption",
         test_149_adapter_frozen_consumption),
    ]
    statuses = {name: fn() for name, fn in tests}
    skips = [name for name, st in statuses.items() if st == "SKIP"]
    if skips:
        print(f"\nGPT-CPLUS-BATCH(142-concurrent/149-adapter) "
              f"RESULT: PARTIAL (SKIP: {', '.join(skips)})")
        return 3
    print("\nGPT-CPLUS-BATCH(142-concurrent/149-adapter) RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
