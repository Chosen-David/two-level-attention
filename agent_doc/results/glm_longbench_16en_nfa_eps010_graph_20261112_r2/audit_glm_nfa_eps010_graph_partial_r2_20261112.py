#!/usr/bin/env python3
"""r2 正式运行中快照：对单机当前全部已落盘记录只读重算。"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/wangyuanshuo02/ak_verify_results/glm_longbench_16en_nfa_eps010_graph_20261112_r2")
REPO = Path("/tmp/tli-glm-nfa-eps010-graph-20261112-r2")
DATA = Path("/home/wangyuanshuo02/datasets/LongBench/data")
IDENTITY = "6e4c432c18e3acc202a6ae27ad8673425840fc7769bce8f323b5d476666ab019"
sys.path.insert(0, "/tmp")
sys.path.insert(0, str(REPO))
from audit_glm_nfa_eps010_graph_20261112 import DATASETS, audit_record, write_json
from benchmark.LongBench.eval import sample_score
from benchmark.LongBench.prefill_frozen_skip import prepare_ids
from transformers import AutoTokenizer

ranks = [int(value) for value in sys.argv[1:]]
manifest = json.loads((ROOT / f"manifest_rank{ranks[0]}.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(manifest["protocol"]["model"])
prompts = json.loads((REPO / "benchmark/LongBench/config/dataset2prompt.json").read_text())
source_rows = {
    dataset: [json.loads(line) for line in (DATA / f"{dataset}.jsonl").read_text().splitlines()]
    for dataset in DATASETS if (ROOT / "requests" / dataset).is_dir()
}
failures = []
audited = []
seen = set()
for path in sorted(ROOT.glob("requests/*/*.json")):
    try:
        record = json.loads(path.read_text())
        key = (record["dataset"], record["row_index"])
        if key in seen:
            failures.append(f"duplicate {key}")
        seen.add(key)
        audited.append(audit_record(
            record, source_rows[record["dataset"]][record["row_index"]], record["dataset"],
            prompts[record["dataset"]], tokenizer, sample_score, prepare_ids, IDENTITY, failures))
    except Exception as exc:
        failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
result = {
    "status": "PASS" if not failures else "FAIL",
    "scope": "producer-side partial running snapshot; not the 3750 closed-set audit",
    "identity": IDENTITY,
    "ranks_on_host": ranks,
    "requests": len(audited),
    "datasets": sorted(seen),
    "failures": failures,
}
write_json(f"/tmp/glm_nfa_eps010_graph_partial_r2_rank{'_'.join(map(str, ranks))}_audit.json", result)
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if not failures else 1)
