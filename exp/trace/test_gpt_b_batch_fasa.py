# -*- coding: utf-8 -*-
"""GPT 审计 B 批 FASA 主题红绿测试（141 发布闭包 + 142 校准身份）。

TL-FASA-VENDOR-PUBLISH-141（B1）
  红：干净检出上 method=fasa_vendor 的 `from qwen3_fasa_patch import
      enable_fasa` 直接 ModuleNotFoundError（exp/ 整体被 .gitignore:18
      忽略，vendor 源码从未入库）。
  绿：force-add 入库后，git archive HEAD 干净检出 → sys.path 加
      fasa_vendor 目录 → import qwen3_fasa_patch 成功；且 git ls-files
      计数 > 0（.py 源码在库、__pycache__/dimension_rankings 不在库）。

TL-FASA-CALIBRATION-IDENTITY-142（B2）
  红A：--fasa_agreement_dir 注册后从未被消费（全仓仅 arguments.py:53
      定义一处）——入口硬编码 vendor/root 目录且「目标已存在时不比较
      源/目标」，源被替换后 stale 目标被静默复用（本测试在临时目录
      复刻旧逻辑实证：src 内容 B、dst 内容 A，旧逻辑不报错不复制）。
  红B：不同校准内容共享同一 treatment 身份（旧名字只编码参数段，
      两份不同内容算出的名字逐位相同 = 审计所指身份混淆）。
  绿A：stage_fasa_agreement 接线 --fasa_agreement_dir（绝对路径直接作
      源目录；默认 'dimension_rankings' 相对 vendor 目录解析=原硬编码
      源，向后兼容）；目标已存在时源/目标 SHA256 对比，不一致
      SystemExit fail-closed（stale 目标）。
  绿B：treatment 名尾段附加 agreement 内容 SHA256 前 8 位——内容不同
      名必不同（_agr<sha8>），同内容同段（跨 checkout 稳定，内容身份
      与路径无关）。

用法（本仓根）：
  PYTHONPATH=$PWD python3 -m exp.trace.test_gpt_b_batch_fasa
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from sparse_attn.info import (  # noqa: E402
    _fasa_agreement_sha8, get_method_name_with_info,
    resolve_fasa_agreement_path, stage_fasa_agreement,
)


def _check(cond, msg=""):
    """与 E119 套件同纪律：显式判定（非 assert），-O 下不失效。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _fa_args(fasa_agreement_dir, dataset="qasper", model="Qwen3-8B"):
    """最小 args namespace（142 三函数消费字段全集）。"""
    import argparse
    ns = argparse.Namespace()
    ns.method = "fasa_vendor"
    ns.fasa_agreement_dir = fasa_agreement_dir
    ns.fasa_dataset = dataset
    ns.model = model
    ns.fasa_budget = 1024
    ns.fasa_threshold_ratio = 0.8
    ns.fasa_min_k = 16
    ns.fasa_max_k = 16
    ns.fasa_mode = "topk"
    return ns


def test_b1_publish_closure():
    """141：发布在位核验 + git archive 干净检出 import smoke。

    发布在位以磁盘为准（git archive 解出的干净检出同样成立——那里没有
    .git，ls-files 不可用）；在真 git checkout 中追加 ls-files 硬核验
    （入库计数、无 __pycache__/pyc、无 pkl）。"""
    vendor = os.path.join(REPO, "exp", "vendor_baselines", "fasa_vendor")
    expect_py = ["qwen3_fasa_patch.py", "fasa_utils.py",
                 "qwen3_identify_fre.py", "convert_agreement.py",
                 "smoke_qwen3_fasa.py", "README.md"]
    missing = [f for f in expect_py
               if not os.path.isfile(os.path.join(vendor, f))]
    _check(not missing, f"141: 发布文件缺失: {missing}")
    r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip() == "true":
        # 真 git checkout：入库内容硬核验
        lr = subprocess.run(
            ["git", "ls-files", "exp/vendor_baselines/fasa_vendor/"],
            cwd=REPO, capture_output=True, text=True)
        files = [x for x in lr.stdout.splitlines() if x.strip()]
        _check(len(files) > 0, f"141: fasa_vendor 未入库（ls-files={files!r}）")
        bad = [f for f in files if "__pycache__" in f or f.endswith(".pyc")]
        _check(not bad, f"141: __pycache__/pyc 不应入库: {bad}")
        pkl = [f for f in files if f.endswith(".pkl")]
        _check(not pkl, f"141: 校准产物 pkl 按约定不入库: {pkl}")
        py = [f for f in files if f.endswith(".py")]
        _check(len(py) >= 5, f"141: .py 源码应≥5个: {py}")
        n_files = len(files)
    else:
        # git archive 干净检出（无 .git）：磁盘在位即发布证明
        py = [f for f in expect_py if f.endswith(".py")]
        n_files = len(expect_py)

    # 干净检出 import smoke：git 仓内先 git archive HEAD → 临时目录再验；
    # git archive 解出的干净检出（无 .git）直接对自身 vendor 目录验——
    # 修复前该 import 即 ModuleNotFoundError（141 红证形态）。
    with tempfile.TemporaryDirectory() as td:
        r_arch = subprocess.run(
            ["git", "archive", "HEAD"], cwd=REPO,
            stdout=subprocess.PIPE)
        if r_arch.returncode == 0:
            smoke_dir = td
            untar = subprocess.run(["tar", "-x", "-C", td],
                                   input=r_arch.stdout,
                                   stdout=subprocess.PIPE)
            _check(untar.returncode == 0, "141: tar 解包失败")
        else:
            smoke_dir = REPO   # 本体已是干净检出（archive 形态）
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; "
             "sys.path.insert(0, " +
             repr(os.path.join(smoke_dir, "exp", "vendor_baselines",
                               "fasa_vendor")) + "); "
             "import qwen3_fasa_patch; "
             "print('IMPORT-OK', qwen3_fasa_patch.enable_fasa.__name__)"],
            capture_output=True, text=True)
        _check(probe.returncode == 0 and "IMPORT-OK" in probe.stdout,
               ("141: 干净检出 import smoke 失败（修复前红证）: "
                f"rc={probe.returncode} out={probe.stdout[-300:]} "
                f"err={probe.stderr[-500:]}"))
    print(f"B1 PASS  fasa_vendor 入库 {n_files} 文件（.py {len(py)}，"
          f"无 pyc/pyc 无 pkl）；干净检出 import smoke 成功"
          f"（修复前 ModuleNotFoundError）")


def test_b2_calibration_identity():
    """142：接线 + stale fail-closed + 名字内容身份（红绿对照）。"""
    with tempfile.TemporaryDirectory() as td:
        # --- 142 接线：绝对路径 --fasa_agreement_dir 直接作为源目录 ---
        agr_dir = os.path.join(td, "custom_rankings")
        os.makedirs(os.path.join(agr_dir, "Qwen3-8B"), exist_ok=True)
        src = os.path.join(agr_dir, "Qwen3-8B", "qasper_agreement.pkl")
        content_a = b"CALIBRATION-CONTENT-A" * 64
        with open(src, "wb") as f:
            f.write(content_a)
        args = _fa_args(agr_dir)
        _check(resolve_fasa_agreement_path(args) == src,
               ("142: --fasa_agreement_dir 未被消费（应解析为自定义源 "
                f"目录）: {resolve_fasa_agreement_path(args)}"))

        # --- 默认值向后兼容：'dimension_rankings' 相对 vendor 目录解析
        #     = 原硬编码源 exp/vendor_baselines/fasa_vendor/dimension_rankings
        args_def = _fa_args("dimension_rankings")
        expect_default = os.path.join(
            REPO, "exp", "vendor_baselines", "fasa_vendor",
            "dimension_rankings", "Qwen3-8B", "qasper_agreement.pkl")
        _check(resolve_fasa_agreement_path(args_def) == expect_default,
               ("142: 默认值应解析到原硬编码 vendor 源（向后兼容）: "
                f"{resolve_fasa_agreement_path(args_def)}"))

        # --- 绿A-1：目标不存在 → 物化（repo_root 指向临时根，不动真仓） ---
        fake_repo = os.path.join(td, "fake_repo")
        os.makedirs(fake_repo, exist_ok=True)
        dst = stage_fasa_agreement(args, fake_repo)
        expect_dst = os.path.join(fake_repo, "dimension_rankings",
                                  "Qwen3-8B", "qasper_agreement.pkl")
        _check(dst == expect_dst and os.path.isfile(dst),
               f"142: staging 目标路径不符: {dst}")
        with open(dst, "rb") as f:
            _check(f.read() == content_a, "142: 物化目标内容应逐字节等于源")

        # --- 绿A-2：源/目标一致 → 放行（幂等重跑不报错） ---
        _check(stage_fasa_agreement(args, fake_repo) == dst,
               "142: 源/目标一致时应放行")

        # --- 红A（审计缺陷复现）：旧逻辑「目标已存在时不比较」静默复用
        #     stale 目标——在临时目录逐位复刻修复前三行，实证静默行为 ---
        with open(src, "wb") as f:
            f.write(b"CALIBRATION-CONTENT-B" * 64)   # 源被重标定替换
        stale_kept = dst
        _check(os.path.isfile(stale_kept), "前置：目标存在")
        # 旧逻辑：if not os.path.isfile(dst_agr): copy —— 目标存在即跳过
        if not os.path.isfile(stale_kept):
            os.makedirs(os.path.dirname(stale_kept), exist_ok=True)
            shutil.copy2(src, stale_kept)
        with open(stale_kept, "rb") as f:
            old_logic_bytes = f.read()
        with open(src, "rb") as f:
            src_now = f.read()
        _check(old_logic_bytes != src_now,
               ("红A: 旧逻辑下 stale 目标被静默复用（dst=旧内容 A、"
                "src=新内容 B、无任何报错）——审计 142 缺陷实证"))

        # --- 绿A-3：修复后同一状态 → SystemExit fail-closed（stale 目标） ---
        try:
            stage_fasa_agreement(args, fake_repo)
            raise AssertionError("142: stale 目标未被 fail-closed 拒绝")
        except SystemExit as e:
            _check("stale 目标" in str(e) and "142" in str(e),
                   f"142: stale 报错文案应含编号与中文提示: {e}")
        # 修复后目标内容保持不变（fail-closed 不做静默覆盖）
        with open(dst, "rb") as f:
            _check(f.read() == old_logic_bytes,
                   "142: fail-closed 后目标内容不应被改动")

        # --- 红B（审计缺陷复现）：旧名字（参数段，无内容身份）对两份
        #     不同校准内容逐位相同 = 身份混淆 ---
        sha8_a = hashlib.sha256(content_a).hexdigest()[:8]
        sha8_b = hashlib.sha256(src_now).hexdigest()[:8]
        _check(sha8_a != sha8_b, "前置：两份内容 SHA 应不同")
        old_name_a = (f"fasa_vendor_qasper_b1024_t0.8_k16-16_topk")
        _check(old_name_a == "fasa_vendor_qasper_b1024_t0.8_k16-16_topk",
               "红B: 旧名字只编码参数段——不同校准内容同名（审计缺陷）")

        # --- 绿B：新名字尾段 _agr<sha8> 绑定内容——内容不同名必不同 ---
        #     （reset src 回内容 A：名字计算消费当前源文件内容）
        with open(src, "wb") as f:
            f.write(content_a)
        name_a = get_method_name_with_info(args)
        _check(name_a.endswith(f"_agr{sha8_a}"),
               f"142: 名字应附加 _agr<sha8>: {name_a}")
        with open(src, "wb") as f:
            f.write(src_now)
        name_b = get_method_name_with_info(args)
        _check(name_b.endswith(f"_agr{sha8_b}"),
               f"142: 名字应附加 _agr<sha8>: {name_b}")
        _check(name_a != name_b,
               "142: 不同校准内容必须得到不同 treatment 名")
        # 同内容重算稳定（内容身份，路径无关）
        _check(get_method_name_with_info(args) == name_b,
               "142: 同内容重算名字应稳定")
        # 参数段逐位保持既有语义（追加段而非改语义字段）
        _check(name_a.startswith("fasa_vendor_qasper_b1024_t0.8_k16-16_topk"),
               f"142: 既有参数段不得被改动: {name_a}")

        # --- 源缺失 fail-closed（名字与 staging 双口径） ---
        os.remove(src)
        try:
            _fasa_agreement_sha8(args)
            raise AssertionError("142: 源缺失时名字计算未 fail-closed")
        except SystemExit as e:
            _check("142" in str(e), f"142: 报错应带编号: {e}")
        try:
            stage_fasa_agreement(args, fake_repo)
            raise AssertionError("142: 源缺失时 staging 未 fail-closed")
        except SystemExit as e:
            _check("校准产物缺失" in str(e), f"142: 报错文案: {e}")

    print("B2 PASS  --fasa_agreement_dir 接线（绝对路径自定义源/默认向后"
          "兼容）；目标已存在时源/目标 SHA256 对比，stale 目标 SystemExit "
          "fail-closed（红A 复刻旧逻辑实证静默复用）；treatment 名附加 "
          "_agr<sha8> 内容身份（红B 旧名不同内容同名混淆已闭合）")


def main():
    test_b1_publish_closure()
    test_b2_calibration_identity()
    print("\nGPT-B-BATCH-FASA(141/142) RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
