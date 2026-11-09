#!/usr/bin/env python3
"""GLM RULER 64K niah_multikey_2 诊断臂独立评分（native fullKV / NFA full256）。

口径与 /tmp/audit_glm_subspace_full_20261106.py 的 audit_ruler 完全一致：
每行得分 = 命中的 ref 数 / ref 总数（小写包含匹配），总分 = 行均值 * 100。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path("/tmp/glm_ruler_diag_20261108")


def score(arm_dir):
    jsonls = [p for p in arm_dir.glob("*.jsonl")]
    assert len(jsonls) == 1, jsonls
    jsonl = jsonls[0]
    receipt_path = jsonl.with_name(jsonl.with_suffix("").name + "-yarn_receipt.json")
    raw = jsonl.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    receipt = json.loads(receipt_path.read_text())
    scores = []
    for row in rows:
        refs = row.get("answers", [])
        pred = row.get("pred", "")
        scores.append(sum(1.0 if r.lower() in pred.lower() else 0.0
                          for r in refs) / len(refs) if refs else 0.0)
    treatment = json.loads(receipt.get("treatment_manifest", {}).get("json", "{}"))
    out = {
        "arm": arm_dir.name,
        "rows": len(rows),
        "unique_ids": len({r.get("_id") for r in rows}),
        "ruler_string_match_all": sum(scores) / len(scores) * 100 if scores else None,
        "prediction_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_status": receipt.get("status"),
        "receipt_prediction_lines": receipt.get("prediction_lines"),
        "receipt_sha_ok": receipt.get("prediction_sha256") == hashlib.sha256(raw).hexdigest(),
        "task": receipt.get("task"),
        "context_length": receipt.get("context_length"),
        "generation_params": receipt.get("generation_params"),
        "treatment_sha256": receipt.get("treatment_manifest", {}).get("sha256"),
        "treatment": {k: v for k, v in treatment.items()
                      if k.startswith("nfa_") or k in ("method", "model")},
        "producer_script": receipt.get("producer_script"),
        "execution_modes": sorted({r.get("execution", {}).get("actual") for r in rows}),
    }
    # full-region 校验（仅 NFA 臂携带该字段）
    region_rows = [r for r in rows if r.get("_frozen_region")]
    if region_rows:
        out["full_region_ok"] = all(
            len(r["_frozen_region"].get("decisions", {})) == 47
            and all(not d.get("skip_far") and not d.get("skip_near")
                    for d in r["_frozen_region"]["decisions"].values())
            for r in region_rows)
        out["region_rows"] = len(region_rows)
    return out


results = [score(ROOT / "native_fullkv"), score(ROOT / "nfa_full256")]
for r in results:
    print(json.dumps(r, ensure_ascii=False, indent=2))
Path("/tmp/glm_ruler_diag_20261108/diag_scores.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + "\n")
