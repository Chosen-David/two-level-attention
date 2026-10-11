# -*- coding: utf-8 -*-
"""TL-E125-ANCHOR-BACKEND-001 修复门禁黑盒测试（GPT 2026-10-11 审计 001）。

复现矩阵（对照 GPT 审计三态表 + 来源冲突）：
  T1 真实锚点（meta 缺 scorer_backend）+ raw 四臂 difflib:stdlib 一致 +
     本轮 difflib:stdlib → 显式迁移放行，exit 0 发布 verdict（迁移字段入 meta）
  T2 真实锚点 + 本轮 levenshtein:fixture → ABORT（修复前此态放行，红证核心）
  T3 真实锚点 + raw 一臂 backend 不同 → 来源冲突 ABORT（迁移不放行）
  T4 真实锚点 + raw 缺一臂 result.json → ABORT fail closed

口径：黑盒调用原 main()，mock run_eval 返回受控 backend（不执行真实评分，
合成分数不代表真实结果）；合成 pred 文件行数达 EXPECT。python 与 -O 双跑。
"""
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
TARGET = os.path.join(REPO, "exp", "trace", "analyze_e125_screen_verdict.py")
REAL_ANCHOR = os.path.join(REPO, "exp", "trace", "results",
                           "e123_trial_verdict.json")
RAW_TEMPLATE = os.path.join(REPO, "exp", "trace", "results",
                            "e123_trial_raw")


def _check(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    print(f"[PASS] {msg}")


def load_module(tmp_root, anchor_path):
    spec = importlib.util.spec_from_file_location("e125_gate_audit", TARGET)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    g.OUT_ROOT = str(tmp_root)
    g.OUT_JSON = str(tmp_root / "verdict.json")
    g.E123_VERDICT = anchor_path
    g.B = 50
    return g


def build_arms(g, tmp_root):
    """合成四臂 pred 文件（行数达 EXPECT），mock run_eval 受控返回。"""
    for arm in g.ARMS:
        p = tmp_root / f"pred_{arm}"
        p.mkdir(parents=True, exist_ok=True)
        for task in g.TASKS:
            (p / f"{task}-synthetic.jsonl").write_text("{}\n" * g.EXPECT[task])


def run_main(g, backend_id):
    g.run_eval = lambda *a, **kw: ({t: 50.0 for t in g.TASKS}, backend_id)
    log = io.StringIO()
    exit_code = 0
    # sys.exit(字符串) 消息走 stderr：合并捕获后从总输出判定门禁文案
    with contextlib.redirect_stdout(log), \
            contextlib.redirect_stderr(log):
        try:
            g.main()
        except SystemExit as e:
            exit_code = str(e.code)
    return exit_code, log.getvalue()


def stage_raw(tmp_root, arms, backend_map):
    """在 tmp 树内复制 E123 raw 四臂 result.json，按 backend_map 改写。"""
    raw_dir = tmp_root / "e123_trial_raw"
    for arm in arms:
        src = pathlib.Path(RAW_TEMPLATE) / f"pred_{arm}" / "result.json"
        d = json.loads(src.read_text())
        if arm in backend_map:
            (d.setdefault("_meta", {}))["scorer_backend"] = backend_map[arm]
        dst = raw_dir / f"pred_{arm}"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "result.json").write_text(json.dumps(d))
    return raw_dir


def main():
    n_pass = 0
    with tempfile.TemporaryDirectory(prefix="e125-gate-test-") as tmp:
        tmp_root = pathlib.Path(tmp)

        # ---- T1：真实锚点 + raw 一致 → 显式迁移放行 ----
        g = load_module(tmp_root / "t1", REAL_ANCHOR)
        build_arms(g, tmp_root / "t1")
        # 迁移源用的是 REPO 相对路径 exp/trace/results/e123_trial_raw，
        # 真实仓库内即真 raw（四臂 difflib:stdlib，已独立核验）。
        code, log = run_main(g, "difflib:stdlib")
        _check(code == 0, "T1 真实锚点 raw 一致：迁移放行 exit 0")
        _check("[MIGRATE]" in log, "T1 迁移日志显式打印 backend 来源")
        v = json.loads((tmp_root / "t1" / "verdict.json").read_text())
        _check(v["meta"]["anchor_provenance"]["anchor_scorer_backend"]
               == "difflib:stdlib", "T1 verdict 登记迁移后 backend")
        _check(v["meta"]["anchor_provenance"]["anchor_backend_migration"]
               is not None, "T1 verdict 登记迁移来源闭包")
        n_pass += 4

        # ---- T2：真实锚点 + 本轮 levenshtein → ABORT（红证核心） ----
        g = load_module(tmp_root / "t2", REAL_ANCHOR)
        build_arms(g, tmp_root / "t2")
        code, log = run_main(g, "levenshtein:fixture")
        # sys.exit(msg) 的 msg 在 e.code 里（被捕获时解释器不打印）
        _check(code != 0 and "不一致" in str(code) and "fail closed" in code,
               "T2 缺字段锚点 vs 异后端：ABORT（修复前此态放行）")
        _check(not (tmp_root / "t2" / "verdict.json").exists(),
               "T2 无 verdict 发布")
        n_pass += 2

        # ---- T3：raw 来源冲突（一臂异后端）→ ABORT ----
        t3 = tmp_root / "t3"
        g = load_module(t3, REAL_ANCHOR)
        build_arms(g, t3)
        # 把模块内 REPO 指向 tmp 树，使迁移读合成 raw（冲突注入）
        raw = stage_raw(t3, ["mavg", "cavg_g", "cavg_off", "fullkv"],
                        {"fullkv": "levenshtein:fixture"})
        g.REPO = str(t3)  # 迁移读 {REPO}/exp/trace/results/e123_trial_raw
        (t3 / "exp/trace/results").mkdir(parents=True, exist_ok=True)
        os.symlink(raw, t3 / "exp/trace/results/e123_trial_raw")
        code, log = run_main(g, "levenshtein:fixture")
        _check(code != 0 and "不一致" in code and "fail closed" in code,
               "T3 raw 四臂 backend 冲突：迁移拒绝 ABORT")
        _check(not (t3 / "verdict.json").exists(), "T3 无 verdict 发布")
        n_pass += 2

        # ---- T4：raw 缺一臂 result.json → ABORT ----
        t4 = tmp_root / "t4"
        g = load_module(t4, REAL_ANCHOR)
        build_arms(g, t4)
        raw = stage_raw(t4, ["mavg", "cavg_g", "cavg_off"], {})
        (t4 / "exp/trace/results").mkdir(parents=True, exist_ok=True)
        os.symlink(raw, t4 / "exp/trace/results/e123_trial_raw")
        g.REPO = str(t4)
        code, log = run_main(g, "difflib:stdlib")
        _check(code != 0 and "fail closed" in code,
               "T4 raw 缺一臂：ABORT fail closed（不猜默认值）")
        _check(not (t4 / "verdict.json").exists(), "T4 无 verdict 发布")
        n_pass += 2

    print(f"ALL PASS ({n_pass}/{n_pass})")


if __name__ == "__main__":
    main()
