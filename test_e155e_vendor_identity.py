#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_e155e_vendor_identity.py —— vendor treatment 身份补齐门禁（CPU only，0 SKIP）。

【缺陷（用户 2026-10-20 问题 ①「R3 manifest 登记缺口」）】46 个 vendor CLI
参数不进任何 treatment 身份：076 注释「非 tli 方法可读段已编码其全部
旋钮」对 vendor 方法不成立——
  * clusterkv_vendor：window/fit_iter/head_sel/balance/dist_t/gqa_policy/
    topk_stat 7 参数完全不进可读名——异配置同名互覆（B09 族）可达；
  * asynctls_vendor/jssa_vendor：channel/basis 文件只编 basename（同
    basename 异内容 = 同名）——142/149 已修 minference/fasa 同构缺陷，
    此两家漏网；
  * 其余 vendor 旋钮虽全编码进可读名，但 manifest/hash 零登记。

【修复（本门禁守护的行为）】E155e：
  1. vendor 参数按 method 条件入 manifest（_VENDOR_TREATMENT_FIELDS；
     文件型字段 → 内容身份 {basename, sha256}，fail closed）；
  2. vendor 方法名统一追加 _h<hash10> 尾段（_vendor_name_with_hash，
     与 tli 076 同构）；
  3. tli/none/tia/twi/quest 的 manifest 逐位不变（method 条件登记，
     E155d 在飞网格与全部 tli hash 锚点稳定——本门禁 T1 硬锚）。

【红绿口径】修复前 T2-T7 必须红（同名/无 hash 段）；修复后全绿。
用法：python3 test_e155e_vendor_identity.py（two-level-attention/ 下，
python 与 python -O 双跑；显式 _check 不依赖 assert 语句）。
"""
import os
import sys
import tempfile
import types

sys.dont_write_bytecode = True

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from sparse_attn import info as INFO  # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def _check(cond, msg):
    # 065 口径：显式 raise（裸 assert 在 python -O 下被剥掉 → 假绿）。
    if not cond:
        raise AssertionError(msg)


def _ns(method, **kw):
    a = types.SimpleNamespace(method=method)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _tmp_file(tag, content):
    d = tempfile.mkdtemp(prefix=f"e155e_{tag}_")
    p = os.path.join(d, f"{tag}.bin")
    with open(p, "wb") as f:
        f.write(content)
    return p


def t1_tli_manifest_unchanged():
    """tli/无 method namespace 的 manifest 字段集 == _TREATMENT_FIELD_
    DEFAULTS 逐键相同（E155d 在飞网格 + 全部 tli hash 锚点稳定的硬锚）。"""
    name = "T1 tli manifest 逐位不变（无 method/vendor 键混入）"
    try:
        ns = _ns("tli")
        m = INFO.resolve_treatment_snapshot(ns).manifest
        _check(set(m) == set(INFO._TREATMENT_FIELD_DEFAULTS),
               f"tli manifest 字段集漂移：多出 "
               f"{sorted(set(m) - set(INFO._TREATMENT_FIELD_DEFAULTS))} / 缺 "
               f"{sorted(set(INFO._TREATMENT_FIELD_DEFAULTS) - set(m))}")
        # e119 锚点配置的 tli hash 不移位（跨构造路径单射的既有锚）。
        # 锚值 = 当前主线实测 h68aa19d58f——e119 测试文件自身的 h175d1bcba6
        # 在主线已陈旧（E135+ 字段扩集后未同步重算，与本门禁无关）；
        # 本门禁断言的是 E155e 改动前后该 hash 逐位不变（git stash 对照已核）。
        ns2 = _ns("tli", tia_block_size=64, tia_level1_topk=128,
                  tia_level2_topk=2048, tia_level2_cmp_ratio=4,
                  tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
                  tli_enable_kmeans=False, tli_enable_layer_skip=False)
        _check(INFO.get_method_name_with_info(ns2)
               == "tli_64_128_2048_c4_a0.25_b0.125_g0.625_h68aa19d58f",
               "e119 tli 锚点 hash 移位（tli manifest 被污染）")
        # 无 method 属性（程序化 namespace）也不触发 vendor 登记
        ns3 = types.SimpleNamespace()
        m3 = INFO.resolve_treatment_snapshot(ns3).manifest
        _check("method" not in m3, "无 method namespace 混入 method 键")
        report(name, True, "字段集/锚点 hash 双稳定")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t2_clusterkv_injective():
    """clusterkv 7 个未编码旋钮（如 window）改变 → 可读段相同但 hash/
    全名必须互异（修复前同名互覆 = B09 族直接复现位）。"""
    name = "T2 clusterkv window 旋钮身份单射（可读段同、hash 异）"
    try:
        n1 = INFO.get_method_name_with_info(
            _ns("clusterkv_vendor", clusterkv_token_budget=1024,
                clusterkv_sink=16, clusterkv_nlist=400, clusterkv_window=320))
        n2 = INFO.get_method_name_with_info(
            _ns("clusterkv_vendor", clusterkv_token_budget=1024,
                clusterkv_sink=16, clusterkv_nlist=400, clusterkv_window=256))
        m = INFO.resolve_treatment_snapshot(
            _ns("clusterkv_vendor", clusterkv_window=256)).manifest
        _check("method" in m and m["method"] == "clusterkv_vendor",
               "manifest 缺 method 键")
        _check("clusterkv_window" in m and m["clusterkv_window"] == 256,
               "manifest 缺 clusterkv_window 旋钮")
        for k in ("clusterkv_fit_iter", "clusterkv_head_sel",
                  "clusterkv_balance", "clusterkv_dist_t",
                  "clusterkv_gqa_policy"):
            _check(k in m, f"manifest 缺 {k}")
        _check(n1 != n2, f"window 320 vs 256 同名互覆：{n1!r}")
        _check(INFO._TREATMENT_HASH_RE.match(n1) is not None
               and INFO._TREATMENT_HASH_RE.match(n2) is not None,
               f"clusterkv 名缺 _h<hash10> 尾段：{n1!r}")
        report(name, True, f"window 320/256 全名互异（{n1} ≠ {n2}）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t3_asynctls_channel_content_identity():
    """asynctls channel 文件同 basename 异内容 → 可读段相同（basename
    tag 相同）但内容身份/hash 互异（142/149 同构缺陷的漏网两家之一）。"""
    name = "T3 asynctls channel 内容身份（同 basename 异内容 → hash 异）"
    try:
        c1 = _tmp_file("channels", b"channel-content-A")
        c2 = _tmp_file("channels", b"channel-content-B-different")
        n1 = INFO.get_method_name_with_info(
            _ns("asynctls_vendor", asynctls_channel_file=c1))
        n2 = INFO.get_method_name_with_info(
            _ns("asynctls_vendor", asynctls_channel_file=c2))
        m1 = INFO.resolve_treatment_snapshot(
            _ns("asynctls_vendor", asynctls_channel_file=c1)).manifest
        m2 = INFO.resolve_treatment_snapshot(
            _ns("asynctls_vendor", asynctls_channel_file=c2)).manifest
        _check(m1["asynctls_channel_file"]["basename"]
               == m2["asynctls_channel_file"]["basename"],
               "同 basename 前置不成立（测试夹具错）")
        _check(m1["asynctls_channel_file"]["sha256"]
               != m2["asynctls_channel_file"]["sha256"],
               "内容身份未绑定 sha256")
        _check(n1 != n2, f"同 basename 异内容同名互覆：{n1!r}")
        report(name, True, "channel 内容身份入 manifest/hash")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t4_jssa_basis_content_identity():
    """jssa basis 文件同 basename 异内容 → hash 互异（漏网两家之二）。"""
    name = "T4 jssa basis 内容身份（同 basename 异内容 → hash 异）"
    try:
        b1 = _tmp_file("basis", b"basis-content-A")
        b2 = _tmp_file("basis", b"basis-content-B-different")
        n1 = INFO.get_method_name_with_info(
            _ns("jssa_vendor", jssa_basis_path=b1, jssa_budget=1024,
                jssa_rank=32))
        n2 = INFO.get_method_name_with_info(
            _ns("jssa_vendor", jssa_basis_path=b2, jssa_budget=1024,
                jssa_rank=32))
        _check(n1 != n2, f"同 basename 异内容同名互覆：{n1!r}")
        # rank 旋钮（可读段已编码）也须入 manifest
        m = INFO.resolve_treatment_snapshot(
            _ns("jssa_vendor", jssa_basis_path=b1, jssa_rank=16)).manifest
        _check(m["jssa_rank"] == 16 and m["jssa_budget"] == 1024,
               "jssa rank/budget 未入 manifest")
        report(name, True, "basis 内容身份入 manifest/hash")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t5_vendor_name_hash_format():
    """全部 vendor 方法名带 _h<hash10> 尾段且与 _treatment_hash 单源。"""
    name = "T5 vendor 名 _h<hash10> 尾段格式 + hash 单源"
    try:
        cfgs = [
            _ns("quest_vendor", quest_token_budget=512, quest_chunk_size=16),
            _ns("moba_vendor", moba_chunk_size=256, moba_budget=1024),
            _ns("hisa_vendor", hisa_budget=1024),
            _ns("sparq_vendor", sparq_k=1024),
        ]
        for ns in cfgs:
            n = INFO.get_method_name_with_info(ns)
            mm = INFO._TREATMENT_HASH_RE.match(n)
            _check(mm is not None,
                   f"{ns.method} 名缺 _h<hash10> 尾段：{n!r}")
            _check(mm.group("hash") == "h" + INFO._treatment_hash(ns),
                   f"{ns.method} 名 hash 段与 _treatment_hash 不同源：{n!r}")
            m = INFO.resolve_treatment_snapshot(ns).manifest
            _check(m.get("method") == ns.method,
                   f"{ns.method} manifest 缺 method 键")
        report(name, True, "4 方法尾段格式 + hash 单源全过")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t6_minference_fasa_manifest():
    """minference config 文件内容身份 + fasa agreement 目录归一 pkl 内容
    身份入 manifest（142/149 的文件身份从名字段升级为 hash 承载）。"""
    name = "T6 minference/fasa 文件身份入 manifest"
    try:
        cfg1 = _tmp_file("pattern", b'{"a": 1}')
        cfg2 = _tmp_file("pattern", b'{"a": 2}')
        m1 = INFO.resolve_treatment_snapshot(
            _ns("minference_vendor", minference_config_path=cfg1)).manifest
        m2 = INFO.resolve_treatment_snapshot(
            _ns("minference_vendor", minference_config_path=cfg2)).manifest
        _check(m1["minference_config_path"]["sha256"]
               != m2["minference_config_path"]["sha256"],
               "minference config 内容身份未入 manifest")
        # fasa agreement：自定义绝对目录 → 归一到 <dir>/<model>/<dataset>_agreement.pkl
        # （resolve_fasa_agreement_path 需 args.model 拼 <model>/ 子目录）
        agr_dir = tempfile.mkdtemp(prefix="e155e_fasa_")
        pkl = os.path.join(agr_dir, "Qwen3-8B", "qasper_agreement.pkl")
        os.makedirs(os.path.dirname(pkl), exist_ok=True)
        with open(pkl, "wb") as f:
            f.write(b"agreement-v1")
        mf = INFO.resolve_treatment_snapshot(
            _ns("fasa_vendor", model="Qwen3-8B",
                fasa_agreement_dir=agr_dir)).manifest
        _check(mf.get("fasa_agreement", {}).get("basename")
               == "qasper_agreement.pkl",
               f"fasa agreement 未归一到 pkl：{mf.get('fasa_agreement')}")
        _check(len(mf["fasa_agreement"]["sha256"]) == 64,
               "fasa agreement sha256 长度错")
        report(name, True, "两文件身份均入 manifest")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


def t7_file_identity_fail_closed():
    """vendor 文件型配置缺失 → [GATE-FAIL] E155e SystemExit（不留
    「能算名不能跑」半配置——079② 同哲学）。"""
    name = "T7 vendor 文件缺失 fail closed"
    try:
        try:
            INFO.resolve_treatment_snapshot(_ns(
                "jssa_vendor", jssa_basis_path="/nonexistent/e155e.bin"))
        except SystemExit as e:
            _check("E155e" in str(e), f"错误文案缺 E155e 标识：{e}")
            report(name, True, "SystemExit fail loudly")
            return
        _check(False, "缺失文件未 fail closed（静默通过）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    t1_tli_manifest_unchanged()
    t2_clusterkv_injective()
    t3_asynctls_channel_content_identity()
    t4_jssa_basis_content_identity()
    t5_vendor_name_hash_format()
    t6_minference_fasa_manifest()
    t7_file_identity_fail_closed()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _check(n_pass + n_fail == len(RESULTS) == 7, "门禁不允许 SKIP/漏项")
    print(f"\n===== E155e vendor 身份门禁：executed {n_pass}/{n_pass + n_fail} "
          f"PASS（0 SKIP）=====")
    sys.exit(1 if n_fail else 0)
