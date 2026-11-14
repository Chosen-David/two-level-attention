#!/usr/bin/env python3
"""r2 六请求回归：复用正式 audit_record 逐条只读重算。"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/wangyuanshuo02/ak_verify_results/glm_nfa_eps010_graph_smoke_20261112_r2")
REPO = Path("/tmp/tli-glm-nfa-eps010-graph-20261112-r2")
IDENTITY = "812122c414c4ed0c87ed54bf4da4cdf9398a2d4acb416ff1133c9c4665b8a66a"
sys.path.insert(0, "/tmp")
sys.path.insert(0, str(REPO))
from audit_glm_nfa_eps010_graph_20261112 import audit_record, write_json
from benchmark.LongBench.eval import sample_score
from benchmark.LongBench.prefill_frozen_skip import prepare_ids
from transformers import AutoTokenizer

manifest = json.loads((ROOT / "manifest_rank0.json").read_text())
assert manifest["identity"] == IDENTITY
assert json.loads((ROOT / "worker_done_rank0.json").read_text())["identity"] == IDENTITY
assert json.loads((ROOT / "heartbeat_rank0.json").read_text())["status"] == "complete"
model_root = Path(manifest["protocol"]["model"])
tokenizer = AutoTokenizer.from_pretrained(model_root)
prompt = json.loads((REPO / "benchmark/LongBench/config/dataset2prompt.json").read_text())["qasper"]
source_rows = [json.loads(line) for line in Path(
    "/home/wangyuanshuo02/datasets/LongBench/data/qasper.jsonl").read_text().splitlines()]
failures = []
audited = []
for path in sorted((ROOT / "requests/qasper").glob("*.json")):
    record = json.loads(path.read_text())
    audited.append(audit_record(
        record, source_rows[record["row_index"]], "qasper", prompt, tokenizer,
        sample_score, prepare_ids, IDENTITY, failures))
result = {
    "status": "PASS" if not failures and len(audited) == 6 else "FAIL",
    "identity": IDENTITY,
    "requests": len(audited),
    "rows": [row["row_index"] for row in audited],
    "scores": {str(row["row_index"]): row["score"] for row in audited},
    "gates": {str(row["row_index"]): row["gates"] for row in audited},
    "cuda_graph": {str(row["row_index"]): row["graph"] for row in audited},
    "failures": failures,
}
write_json(ROOT / "smoke_audit.json", result)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if result["status"] == "PASS" else 1)
