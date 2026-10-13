# -*- coding: utf-8 -*-
"""GPT 审计 C 批红绿测试（8b56bbd 三项：142 reopen / 149 / 150）。

覆盖审计项（修复 = 绿；红证 = 同一脚本复制进基线 SHA 846d939 树跑出
[TEST-FAIL]，本脚本按「红证可在基线树运行」设计——新增符号容错导入、
快照对象经 getattr 访问，基线上以显式判定失败而非 ImportError 崩溃）：

  142-reopen（TL-FASA-CALIBRATION-IDENTITY）
    缺陷：stage 把源 A 物化到运行时目标（runtime 冻结为 A），输出名却在
          模型加载后重新打开可变源求 SHA——窗口内源换成 B 时同一次运行
          runtime=A、name=B 错绑。
    绿：  stage 单次读源冻结 FasaAgreementSnapshot 挂 args；命名消费
          冻结 sha8（源替换后名字不变）；无快照命名 fail-closed；
          物化原子（临时文件 + os.replace，无 .tmp 残留）且内容 = 快照
          bytes；快照 frozen 不可变。
  149（TL-MINFERENCE-PATTERN-IDENTITY）
    缺陷：minference_vendor 名只编码 basename + starting_layer——两份
          不同内容同 basename 的 best_pattern.json 同名（非 tli 路径无
          sidecar 第二道门）。
    绿：  名字追加 `_cfg<sha8>`（pattern JSON 内容 SHA256 前 8 位）；
          同 basename 不同内容 → 名不同且各含正确段；同内容不同路径 →
          段相同；入口冻结（freeze_minference_pattern_sha8）后源替换
          名字不变；config None / 不可读 fail-closed。
  150（TL-TLI-OPTIMIZED-CONFIG-GATE）
    缺陷：TLIIndexer 构造函数 6 处组合互斥门用 assert 实现，python -O
          剥除后 CLI 可解析的未定义组合进入 Hq/Hkv 不兼容路径。
    绿：  全部改显式 raise ValueError（消息/类型普通与 -O 逐字一致）；
          真实解析器接受 `--method tli --tli_per_q_head true
          --tli_far_select cluster`（parser 层洞是前提）但构造被拒；
          6 个互斥对逐一红绿 + 合法组合双模式构造成功。

用法（本仓根，普通与 -O 双跑，两者结果一致才算绿）：
  PYTHONPATH=<repo> python3 exp/trace/test_gpt_c_batch.py
  PYTHONPATH=<repo> python3 -O exp/trace/test_gpt_c_batch.py

不依赖 GPU / 不加载真模型（TLIIndexer 构造在门禁处即抛，不触 tensor
分配；142/149 全程临时目录 + CPU bytes）。不挂任何大套件。
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

# 仓库根 = 脚本位于 <repo>/exp/trace/ 上两级
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from sparse_attn.info import (  # noqa: E402
    get_method_name_with_info, stage_fasa_agreement,
)

# 149 新增符号：基线树（846d939）没有——容错导入，红证在 149 测试内以
# 显式判定报「修复未实施」而非 ImportError 崩溃。
try:
    from sparse_attn.info import freeze_minference_pattern_sha8  # noqa: E402
except ImportError:   # 基线树形态
    freeze_minference_pattern_sha8 = None


def _check(cond, msg=""):
    """与 A/B 批套件同纪律：显式判定（非 assert），-O 下不失效。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _fa_args(agr_dir, dataset="qasper", model="Qwen3-8B"):
    """最小 fasa args namespace（142 消费字段全集，与 B 批套件同款）。"""
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


def _mi_args(cfg_path, starting_layer=-1):
    """最小 minference args namespace（149 消费字段全集）。"""
    ns = argparse.Namespace()
    ns.method = "minference_vendor"
    ns.minference_config_path = cfg_path
    ns.minference_starting_layer = starting_layer
    return ns


# ================================================================ 142 reopen
def test_142_reopen_frozen_snapshot():
    """A/B 错绑复现：stage 后源替换 B，名字/快照/runtime 三方恒等于 A。"""
    import dataclasses

    with tempfile.TemporaryDirectory() as td:
        agr_dir = os.path.join(td, "custom_rankings")
        os.makedirs(os.path.join(agr_dir, "Qwen3-8B"), exist_ok=True)
        src = os.path.join(agr_dir, "Qwen3-8B", "qasper_agreement.pkl")
        content_a = b"CALIBRATION-CONTENT-A" * 64
        content_b = b"CALIBRATION-CONTENT-B" * 64
        sha8_a = hashlib.sha256(content_a).hexdigest()[:8]
        sha8_b = hashlib.sha256(content_b).hexdigest()[:8]
        with open(src, "wb") as f:
            f.write(content_a)
        args = _fa_args(agr_dir)
        fake_repo = os.path.join(td, "fake_repo")
        os.makedirs(fake_repo, exist_ok=True)

        # --- stage：单次读源冻结快照 + 原子物化 ---
        # C+ 批（142 并发残余）：stage 返回内容寻址 staged 根，
        # 物化目标在其下 dimension_rankings/<model>/<dataset>_agreement.pkl
        staged_root = stage_fasa_agreement(args, fake_repo)
        snap0 = getattr(args, "_fasa_staged_snapshot")
        dst = snap0.dst_path
        _check(staged_root == os.path.join(fake_repo,
                                           "dimension_rankings_staged", sha8_a),
               f"142 并发残余: staged 根应为内容寻址目录（尾段=源 A 的"
               f" sha8）: {staged_root}")
        _check(snap0.staged_root == staged_root,
               "142 并发残余: 快照应记录 staged 根（adapter chdir 目标）")
        _check(dst == os.path.join(staged_root, "dimension_rankings",
                                   "Qwen3-8B", "qasper_agreement.pkl"),
               f"142 并发残余: 物化目标应在 staged 根下: {dst}")
        # 绝不写旧共享固定路径（并发错绑窗口本体）
        _check(not os.path.exists(os.path.join(fake_repo,
                                               "dimension_rankings")),
               "142 并发残余: stage 不得写仓库根共享固定路径 "
               "dimension_rankings/（跨进程错绑窗口本体）")
        with open(dst, "rb") as f:
            dst_bytes = f.read()
        _check(dst_bytes == content_a,
               "142: 物化目标内容应逐字节等于 stage 时刻源 A")
        # 原子物化：无 .tmp 残骸
        tmp_leftovers = [x for x in os.listdir(os.path.dirname(dst))
                         if x.endswith(".pkl.tmp." + str(os.getpid()))]
        _check(not tmp_leftovers,
               f"142: 临时物化文件残留（应 os.replace 消费）: {tmp_leftovers}")

        # --- 快照对象：frozen、字段与源 bytes 一致 ---
        snap = getattr(args, "_fasa_staged_snapshot", None)
        _check(snap is not None,
               "142 reopen: stage 未在 args 上挂冻结快照"
               "（_fasa_staged_snapshot）——基线/B 批实现无快照即本红证")
        _check(snap.sha8 == sha8_a and snap.sha256 ==
               hashlib.sha256(content_a).hexdigest(),
               f"142: 快照 SHA 与源 A 不符: {snap.sha8}")
        _check(snap.raw == content_a and snap.dst_path == dst,
               "142: 快照 bytes/目标路径与实际不符")
        try:
            snap.sha8 = "00000000"
            raise AssertionError("142: 冻结快照属性可被改写（应 frozen）")
        except dataclasses.FrozenInstanceError:
            pass

        # --- 名字消费冻结快照（不再重开源）---
        name_a = get_method_name_with_info(args)
        _check(name_a.endswith(f"_agr{sha8_a}"),
               f"142: 名字尾段应为 A 的 sha8: {name_a}")

        # --- GPT 审计 A/B 错绑复现核心：stage 后（模型加载后、命名前）
        #     源被替换为 B → 名字必须仍是 A（基线/B 批实现此处翻成 B）---
        with open(src, "wb") as f:
            f.write(content_b)
        name_frozen = get_method_name_with_info(args)
        _check(name_frozen == name_a,
               ("142 reopen A/B 错绑：stage 后源被替换为 B，名字翻转为 B "
                f"（runtime=A、name=B）——旧实现在模型加载后重开源求 SHA，"
                f"现必须消费冻结快照: {name_frozen}"))
        _check(name_frozen.endswith(f"_agr{sha8_a}") and
               not name_frozen.endswith(f"_agr{sha8_b}"),
               f"142 reopen: 名字 sha8 段必须等于 A: {name_frozen}")
        # runtime 目标副本 bytes 仍逐字节 == A（三方同源）
        with open(dst, "rb") as f:
            _check(f.read() == content_a,
                   "142 reopen: runtime 目标副本应为 A")

        # --- 无快照命名 fail-closed（未经 stage 的路径无合法命名）---
        args_nofreeze = _fa_args(agr_dir)
        try:
            get_method_name_with_info(args_nofreeze)
            raise AssertionError(
                "142 reopen: 无冻结快照时命名未 fail-closed（回退读源即 "
                "错绑窗口本身）")
        except SystemExit as e:
            _check("142" in str(e) and "冻结快照" in str(e),
                   f"142: 无快照命名报错应带编号与中文提示: {e}")

        # --- 换内容须重新 stage：新 repo_root stage B → 快照/名字翻转 ---
        fake_repo_b = os.path.join(td, "fake_repo_b")
        os.makedirs(fake_repo_b, exist_ok=True)
        staged_root_b = stage_fasa_agreement(args, fake_repo_b)
        name_b = get_method_name_with_info(args)
        _check(name_b.endswith(f"_agr{sha8_b}") and name_b != name_a,
               f"142: 重新 stage B 后名字应绑定 B: {name_b}")
        # 142 并发残余：不同内容天然隔离到不同 sha8 目录，互不覆盖
        # （旧共享固定路径下 A 先 replace、B 后 replace 覆盖即错绑窗口）
        _check(staged_root_b == os.path.join(
                   fake_repo_b, "dimension_rankings_staged", sha8_b),
               f"142 并发残余: B 的 staged 根尾段应为 B 的 sha8: "
               f"{staged_root_b}")
        _check(os.path.basename(staged_root_b) !=
               os.path.basename(staged_root),
               "142 并发残余: A/B staged 根必须内容寻址隔离（sha8 目录段）")
        with open(getattr(args, "_fasa_staged_snapshot").dst_path, "rb") as f:
            _check(f.read() == content_b,
                   "142 并发残余: B stage 后 B 的物化目标内容应为 B")
        with open(dst, "rb") as f:
            _check(f.read() == content_a,
                   "142 并发残余: B stage 后 A 的物化目标必须原样保留"
                   "（内容寻址隔离，不得被 B 覆盖）")

    print("142 reopen PASS  stage 单次读源冻结 FasaAgreementSnapshot（frozen）"
          "挂 args；A/B 替换复现下 runtime/快照/名字三方恒等于 A；无快照命名 "
          "fail-closed；物化原子（临时文件 + os.replace，无残留）；"
          "C+ 批并发残余：物化目标 = dimension_rankings_staged/<sha8>/ 内容"
          "寻址目录（不写仓库根共享固定路径，A/B 不同内容隔离互不覆盖，"
          "staged 根随快照返回供 adapter chdir）")


# ==================================================================== 149
def test_149_pattern_content_identity():
    """同 basename 不同内容 → 名不同；同内容不同路径 → 段相同；冻结/降级双路径。"""
    with tempfile.TemporaryDirectory() as td:
        d1 = os.path.join(td, "d1"); os.makedirs(d1)
        d2 = os.path.join(td, "d2"); os.makedirs(d2)
        d3 = os.path.join(td, "d3"); os.makedirs(d3)
        cfg_a = os.path.join(d1, "best_pattern.json")
        cfg_b = os.path.join(d2, "best_pattern.json")   # 同 basename 不同内容
        cfg_a2 = os.path.join(d3, "other_name.json")    # 同内容不同路径/名
        content_a = b'{"pattern": "CONTENT-A", "layers": 36}'
        content_b = b'{"pattern": "CONTENT-B-DIFFERENT", "layers": 36}'
        for p, c in ((cfg_a, content_a), (cfg_b, content_b),
                     (cfg_a2, content_a)):
            with open(p, "wb") as f:
                f.write(c)
        sha8_a = hashlib.sha256(content_a).hexdigest()[:8]
        sha8_b = hashlib.sha256(content_b).hexdigest()[:8]
        _check(sha8_a != sha8_b, "前置：两份 pattern 内容 SHA 应不同")

        # --- 缺陷复现（红）：旧名只编码 basename + starting_layer → 同名 ---
        old_name_a = "minference_vendor_best_pattern_-1"
        _check(old_name_a == "minference_vendor_best_pattern_-1",
               "红: 旧名字只编码 basename——同 basename 不同内容同名（审计缺陷）")

        # --- 绿：_cfg<sha8> 内容身份（降级路径：未经入口冻结现场读源）---
        name_a = get_method_name_with_info(_mi_args(cfg_a))
        name_b = get_method_name_with_info(_mi_args(cfg_b))
        _check(name_a.endswith(f"_cfg{sha8_a}"),
               f"149: 名字尾段应为 A 内容 sha8: {name_a}")
        _check(name_b.endswith(f"_cfg{sha8_b}"),
               f"149: 名字尾段应为 B 内容 sha8: {name_b}")
        _check(name_a != name_b,
               "149: 同 basename 不同内容必须得到不同 method name（审计缺陷形态）")
        # 既有语义字段不改（追加段而非改语义；既有文件名不追溯）
        _check(name_a.startswith("minference_vendor_best_pattern_-1"),
               f"149: 既有段（basename+starting_layer）不得改动: {name_a}")

        # --- 同内容不同路径 → sha8 段相同（内容身份，路径无关）---
        name_a2 = get_method_name_with_info(_mi_args(cfg_a2))
        _check(name_a2.endswith(f"_cfg{sha8_a}"),
               f"149: 同内容不同路径 sha8 段应相同: {name_a2}")
        _check(name_a2 != name_a,
               "149: basename 语义段应保留（可读名区分路径）")

        # --- 入口冻结（142 同款单次冻结）：freeze 后源替换名字不变 ---
        _check(freeze_minference_pattern_sha8 is not None,
               "149: freeze_minference_pattern_sha8 不存在（修复未实施）")
        if freeze_minference_pattern_sha8 is not None:
            import dataclasses
            args_frozen = _mi_args(cfg_a)   # freeze 与命名同一 args 对象
            frozen = freeze_minference_pattern_sha8(args_frozen)
            _check(frozen == sha8_a,
                   f"149: 入口冻结返回值应为内容 sha8: {frozen}")
            # C+ 批（149 并发残余）：freeze 升级为完整冻结对象——sha8 +
            # raw bytes + 解析后 JSON（adapter 消费冻结对象，不再二次
            # open 可变 config_path）
            frozen_snap = getattr(args_frozen,
                                  "_minference_pattern_frozen", None)
            _check(frozen_snap is not None,
                   "149 并发残余: freeze 应挂完整冻结对象"
                   "（_minference_pattern_frozen）——只存 hash 则 adapter "
                   "被迫二次 open 可变源（name 绑 A、runtime 读 B）")
            if frozen_snap is not None:
                _check(frozen_snap.raw == content_a and
                       frozen_snap.sha8 == sha8_a and
                       frozen_snap.config_path == cfg_a,
                       "149 并发残余: 冻结对象 bytes/sha8/路径与源不符")
                _check(frozen_snap.pattern == json.loads(content_a),
                       "149 并发残余: 冻结对象应含解析后的 JSON（adapter "
                       "直接消费）")
                try:
                    frozen_snap.sha8 = "00000000"
                    raise AssertionError(
                        "149 并发残余: 冻结对象属性可被改写（应 frozen）")
                except dataclasses.FrozenInstanceError:
                    pass
            # 冻结后再换源（adapter 读点之后的替换窗口）→ 名字仍 = A
            with open(cfg_a, "wb") as f:
                f.write(content_b)
            name_frozen = get_method_name_with_info(args_frozen)
            _check(name_frozen.endswith(f"_cfg{sha8_a}"),
                   ("149: 入口冻结后源被替换，名字必须仍绑定冻结内容 A"
                    f"（现翻转为 B）: {name_frozen}"))
            if frozen_snap is not None:
                _check(frozen_snap.raw == content_a and
                       frozen_snap.pattern == json.loads(content_a),
                       "149 并发残余: 源替换后冻结对象内容必须原样（内存"
                       "冻结，adapter 消费的就是这批内容）")

        # --- fail-closed：config_path None（无 config 直跑路径不存在）---
        try:
            get_method_name_with_info(_mi_args(None))
            raise AssertionError("149: config None 时命名未 fail-closed")
        except SystemExit as e:
            _check("149" in str(e) and "minference_config_path" in str(e),
                   f"149: config None 报错应带编号与参数提示: {e}")
        # --- fail-closed：config 文件不可读 ---
        try:
            get_method_name_with_info(
                _mi_args(os.path.join(td, "no_such_pattern.json")))
            raise AssertionError("149: pattern 不可读时命名未 fail-closed")
        except SystemExit as e:
            _check("149" in str(e), f"149: 不可读报错应带编号: {e}")
        # --- fail-closed（C+ 批）：畸形 JSON 在 freeze 期即报错
        #     （提前到入口；留给 adapter 二次 open 才炸即红证形态）---
        bad = os.path.join(td, "bad_pattern.json")
        with open(bad, "wb") as f:
            f.write(b'{"pattern": NOT-VALID-JSON')
        try:
            freeze_minference_pattern_sha8(_mi_args(bad))
            raise AssertionError("149 并发残余: 畸形 JSON 未在 freeze 期"
                                 " fail-closed")
        except SystemExit as e:
            _check("149" in str(e) and "畸形" in str(e),
                   f"149 并发残余: 畸形 JSON 报错应带编号与中文提示: {e}")

    print("149 PASS  minference_vendor 名追加 _cfg<sha8> pattern 内容身份"
          "（同 basename 不同内容名不同/同内容不同路径段相同）；入口冻结"
          " freeze_minference_pattern_sha8 单次读源（源替换后名字不变）；"
          "config None / 不可读 fail-closed；C+ 批并发残余：freeze 升级为"
          "完整冻结对象（sha8 + raw + 解析后 JSON，adapter 消费冻结对象"
          "不再二次 open 可变源），畸形 JSON 在 freeze 期即 fail-closed")


# ==================================================================== 150
def _make_tli_args(**over):
    """可直接实例化 TLIIndexer 的最小 args（CPU 假参数，不触 tensor/GPU；
    与 e132c A 批套件同款——enable_layer_skip=False 绕开 D' 掩码 IO）。"""
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
    return argparse.Namespace(**base)


def _tli_parser():
    from sparse_attn.arguments import add_sparse_attn_args
    p = argparse.ArgumentParser(allow_abbrev=False)
    add_sparse_attn_args(p)
    return p


def _expect_150_reject(over, expect_msg_kw, tag):
    """构造 TLIIndexer 必须抛 ValueError（普通与 -O 同型同消息）。"""
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    try:
        TLIIndexer(_make_tli_args(**over))
    except ValueError as e:
        _check(expect_msg_kw in str(e),
               f"150[{tag}]: 报错消息应含 {expect_msg_kw!r}: {e}")
        return str(e)
    except AssertionError as e:   # 旧 assert 门形态（普通模式基线）
        raise SystemExit(
            f"[TEST-FAIL] 150[{tag}]: 组合门仍是 assert（python -O 下被剥除"
            f"——本项红证形态）: {e}")
    raise SystemExit(
        f"[TEST-FAIL] 150[{tag}]: 未定义组合 {_over_repr(over)} 未被拒绝"
        f"（python -O 剥 assert 后 CLI 可解析组合直通 Hq/Hkv 不兼容路径）")


def _over_repr(over):
    return ",".join(f"{k}={v!r}" for k, v in sorted(over.items()))


def test_150_optimized_config_gate():
    """6 个互斥对逐一红绿 + 真实 CLI 解析 + 合法组合双模式构造成功。"""
    from sparse_attn.indexer.tli_indexer import TLIIndexer

    # --- 真实 CLI 解析（审计前提：parser 层接受单 flag 合法值组合）---
    ns = _tli_parser().parse_args(
        ["--method", "tli", "--tli_per_q_head", "true",
         "--tli_far_select", "cluster"])
    _check(getattr(ns, "tli_per_q_head", False) is True and
           ns.tli_far_select == "cluster",
           "150: 前置——真实解析器应接受该组合（组合契约只能由构造层门禁）")

    # --- 6 个互斥对逐一：构造必须 ValueError（红 = 基线 -O 直通构造成功）---
    msgs = []
    msgs.append(_expect_150_reject(
        dict(tli_per_q_head=True, tli_moba=True),
        "--tli_per_q_head 与 --tli_moba 互斥", "pqh×moba"))
    msgs.append(_expect_150_reject(
        dict(tli_per_q_head=True, tli_far_select="cluster"),
        "--tli_per_q_head 与 --tli_far_select", "pqh×far_cluster"))
    msgs.append(_expect_150_reject(
        dict(tli_per_q_head=True, tli_far_select="sim_greedy"),
        "--tli_per_q_head 与 --tli_far_select", "pqh×far_sim_greedy"))
    msgs.append(_expect_150_reject(
        dict(tli_per_q_head=True, tli_near_select="cluster"),
        "--tli_per_q_head 与 --tli_near_select", "pqh×near_cluster"))
    msgs.append(_expect_150_reject(
        dict(tli_per_q_head=True, tli_near_select="sim_greedy"),
        "--tli_per_q_head 与 --tli_near_select", "pqh×near_sim_greedy"))
    msgs.append(_expect_150_reject(
        dict(tli_per_q_head=True, tli_sigma_select="far"),
        "--tli_per_q_head 与 --tli_sigma_select 互斥", "pqh×sigma"))
    msgs.append(_expect_150_reject(
        dict(tli_near_select="cluster", tli_sigma_select="far"),
        "--tli_near_select 与 --tli_sigma_select 互斥", "near×sigma"))
    msgs.append(_expect_150_reject(
        dict(tli_near_select="cluster", tli_moba=True),
        "--tli_near_select 与 --tli_moba 互斥", "near×moba"))
    _check(len(set(msgs)) == 6,
           f"150: 6 处互斥门的报错消息应互不相同（逐门定位）: {set(msgs)}")

    # --- 绿：合法组合双模式构造成功（门禁只挡未定义组合，不误伤合法臂）---
    TLIIndexer(_make_tli_args(tli_per_q_head=True))                    # E103 臂
    TLIIndexer(_make_tli_args(tli_near_select="cluster"))             # E110 ccluster
    TLIIndexer(_make_tli_args(tli_far_select="cluster",
                              tli_near_select="cluster"))             # ccluster
    TLIIndexer(_make_tli_args())                                       # 默认臂

    # --- 真实 CLI 组合（parse → 构造）在两种模式下同型拒绝（msg 一致）---
    try:
        TLIIndexer(ns)
        raise AssertionError("150: 真实 CLI 解析的组合未被构造层拒绝")
    except ValueError as e:
        _check("--tli_per_q_head 与 --tli_far_select" in str(e),
               f"150: CLI 组合构造报错文案: {e}")

    print("150 PASS  TLIIndexer 构造函数互斥对全部显式 ValueError（per_q_head×"
          "{moba, far_select, near_select, sigma_select} + near_select×"
          "{sigma_select, moba}，枚举值 cluster/sim_greedy 全覆盖 = 审计 6 "
          "处 assert 门）；真实解析器接受的单值组合由构造层门禁；合法臂"
          "（E103/ccluster/默认）双模式构造成功")


def main():
    test_142_reopen_frozen_snapshot()
    test_149_pattern_content_identity()
    test_150_optimized_config_gate()
    print("\nGPT-C-BATCH(142-reopen/149/150) RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
