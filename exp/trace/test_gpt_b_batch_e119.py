# -*- coding: utf-8 -*-
"""GPT 审计 B 批 E119 卫生主题验收（138 sidecar 归档 + 140 generation 发布）。

TL-E119-SIDECAR-REBASE-138（B5）
  缺陷：e119_ruler128k_formal_fullkv.json.manifest.yarn_correction.json
  绑定旧 manifest SHA（0b7f9a70...）而 fullkv manifest 已被
  e116i-generation-v3 合法重发布（现值 440b2b45...，run 20261013114511）
  → yarn_receipt resolver 按 061 纪律 fail-closed，R1 用例 FAIL。
  修复（GPT 方案「不改绑」）：过期 sidecar 从固定别名旁移到版本化
  归档路径（git mv，内容字节不动），R1 用例同步更新为双形态断言
  （普通与 -O 双跑见 E119_ONLY=R1，本文件做归档完整性独立验收）。

TL-E119-GENERATION-PUBLISH-140（B6）
  缺陷：e119_ruler128k_formal_fullkv.json.receipt.json 声明
  e116i-generation-v3 协议与不可变 derived_dir，但该 generation 目录
  （*.run-20261013114511-4190523-3022/）未入库——干净检出上 receipt
  的发布声明不可兑现。修复：derived_dir 全量 git add -f 入库；本文件
  按 receipt 单指针路径逐一读取四个规范文件（result.json / result.md /
  manifest.json / receipt.json）并与 receipt 内记录的 SHA256 逐值对齐
  （绝对路径→当前 checkout 根的重定向规则，使干净检出同口径可验）。

用法（本仓根，普通与 -O 双跑）：
  PYTHONPATH=$PWD python3 -m exp.trace.test_gpt_b_batch_e119
"""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "exp", "trace", "results")
sys.path.insert(0, REPO)

# receipt 里记录的是发布时刻的绝对路径（canonical checkout 根）；
# 验证时按「去掉 canonical 前缀 → 拼当前 checkout 根」重定向，
# 使同一断言在主仓/worktree/干净检出（git archive）下都成立。
CANONICAL_PREFIX = "/home/wangyuanshuo02/two-level-attention/"


def _check(cond, msg=""):
    """与 E119 套件同纪律：显式判定（非 assert），-O 下不失效。"""
    if not cond:
        raise SystemExit(f"[TEST-FAIL] {msg}")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(absolute_path):
    """发布时刻绝对路径 → 当前 checkout 下的对应路径。"""
    _check(absolute_path.startswith(CANONICAL_PREFIX),
           f"140: receipt 路径非 canonical 根: {absolute_path}")
    return os.path.join(REPO,
                        absolute_path[len(CANONICAL_PREFIX):].lstrip("/"))


def test_138_sidecar_archive():
    """138：过期 sidecar 归档完整性（字节不动 + 旧绑定保留 + 别名已空）。"""
    manifest = os.path.join(
        RESULTS, "e119_ruler128k_formal_fullkv.json.manifest.json")
    alias = manifest[:-len(".json")] + ".yarn_correction.json"
    archive = os.path.join(
        RESULTS, "e119_yarn_correction_archive_20261010_fullkv.json")
    _check(not os.path.isfile(alias),
           f"138: 固定别名应已旁移（仍存在: {alias}）")
    _check(os.path.isfile(archive),
           f"138: 版本化归档应在位（缺失: {archive}）")
    doc = json.load(open(archive))
    # 归档内容 = 原 sidecar 原样：旧 manifest 绑定（0b7f9a70...）原样
    # 保留——「不改绑、不删证据」，历史事实与重发布事实各自留档。
    _check(doc["target_manifest_sha256"].startswith("0b7f9a70"),
           ("138: 归档应保留旧 manifest 绑定 SHA",
            doc.get("target_manifest_sha256")))
    _check(doc["correction_version"] == "yarn-identity-057-v1", doc)
    # 现值 manifest 与旧绑定不同（正是归档动因，防未来误回挂）
    now = _sha(manifest)
    _check(now.startswith("440b2b45") and
           doc["target_manifest_sha256"] != now,
           ("138: 现 manifest SHA 应为重发布值且与归档绑定不同", now))
    # resolver 对 fullkv 透传 producer_receipt/4.0（与 R1 双形态断言同源）
    from benchmark.RULER.yarn_receipt import resolve_manifest_yarn_identity
    res = resolve_manifest_yarn_identity(manifest)
    _check(res["correction"] is None and
           res["effective_yarn_factor"] == 4.0 and
           res["yarn_factor_provenance"] == "producer_receipt", res)
    print("B5 PASS  138 归档完整性：固定别名已空、版本化归档在位且字节"
          "保留旧绑定 0b7f9a70...、现 manifest 440b2b45... 透传 "
          "producer_receipt/4.0（R1 普通/-O 双跑见 E119 套件）")


def test_140_generation_publish_alignment():
    """140：receipt 单指针 derived_dir → 四规范文件逐一 SHA 对齐。"""
    receipt_path = os.path.join(
        RESULTS, "e119_ruler128k_formal_fullkv.json.receipt.json")
    r = json.load(open(receipt_path))
    _check(r["publish_protocol"] == "e116i-generation-v3" and
           r["run_id"] == "20261013114511-4190523-3022",
           ("140: receipt 协议/run 校验", r.get("publish_protocol"),
            r.get("run_id")))
    derived = _repo_rel(r["outputs"]["derived_dir"])
    _check(os.path.isdir(derived),
           f"140: derived_dir 未入库或路径不可达: {derived}")
    gf = r["outputs"]["generation_files"]
    # 四个规范文件：receipt.generation_files 声明 + 顶层 SHA 键对齐
    checks = [
        ("result.json", gf["json"], r["result_sha256"]),
        ("result.md", gf["md"], r["gen_md_sha256"]),
        ("manifest.json", gf["manifest"], r["manifest_sha256"]),
        ("receipt.json", gf["receipt"], r["gen_receipt_sha256"]),
    ]
    for fname, declared, want in checks:
        path = os.path.join(derived, declared)
        _check(os.path.isfile(path),
               f"140: 规范文件缺失（按 receipt 单指针路径读不到）: {path}")
        got = _sha(path)
        _check(got == want,
               (f"140: {fname} SHA 与 receipt 记录不一致",
                f"got {got} want {want}"))
        print(f"  [140] {declared:14s} sha256 {got[:16]} == receipt "
              f"{want[:16]}  对齐")
    # derived manifest 与发布侧 manifest 同字节（同一身份双向闭包）
    pub_manifest = _repo_rel(r["outputs"]["manifest"])
    _check(_sha(pub_manifest) == r["manifest_sha256"],
           "140: 发布侧 manifest 也应与 receipt manifest_sha256 一致")
    # 预测物完整性：pred_root 下 11 任务 jsonl 在位（receipt cells 键集）
    n_pred = 0
    pred_root = os.path.join(derived, "pred_root")
    for dirpath, _dirs, files in os.walk(pred_root):
        n_pred += sum(1 for f in files if f.endswith(".jsonl"))
    # receipt cells 键为 "L131072/none-20261013"（格身份复合键）
    _check(len(r["cells"]) == 1 and "L131072" in list(r["cells"])[0],
           ("140: cells 应为单一 131072 档格", list(r["cells"])))
    cells = list(r["cells"].values())[0]["tasks"]
    _check(len(cells) == 11 and n_pred == 11,
           (f"140: 预测物应 11 份（receipt cells={len(cells)}、"
            f"入库 pred_root={n_pred}）"))
    print("B6 PASS  140 generation 发布闭包：derived_dir 已入库，四规范"
          "文件按 receipt 单指针路径可读且 SHA 全部与 receipt 记录对齐，"
          "发布侧 manifest 同 SHA，pred_root 11 任务预测物在位")


def main():
    test_138_sidecar_archive()
    test_140_generation_publish_alignment()
    print("\nGPT-B-BATCH-E119(138/140) RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
