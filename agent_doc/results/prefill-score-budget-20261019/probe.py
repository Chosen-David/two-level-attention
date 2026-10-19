"""Source-bound score-shape diagnostic, NOT a tensor/GPU memory benchmark."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "sparse_attn/ops/eager_prefill.py"
EXPECTED_SHA256 = "bd05914d4e7f6471f2287956b665ba4138d140663e4a83654eafc98be9752352"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-compliance", action="store_true")
    args = parser.parse_args()
    source = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != EXPECTED_SHA256:
        raise SystemExit("source changed: review shape assumptions before rerunning")
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_row_block_size")
    budget = 1 << 27
    namespace = {"_ROW_BLOCK_MAX_ELEM": budget}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), namespace)
    config = dict(chunk=2048, pos_lo=2048, K2=1024, block=64,
                  sink=128, swa=128, D=128, heads=[[32, 8], [64, 8], [32, 32]])
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cap = config["K2"] + 2 * config["block"] + config["sink"] + config["swa"]
    cases = []
    for heads, kv_heads in config["heads"]:
        # Shared-KV, non-group sparse call at source lines 174-177.
        limit = namespace["_row_block_size"](kv_heads, cap, config["swa"], config["sink"])
        rows = min(limit, config["chunk"])
        # Non-group W at lines 207-209, first row block of second chunk.
        width = min(cap + config["sink"] + rows + max(config["swa"], 1),
                    config["pos_lo"] + rows)
        elements = rows * heads * width  # Actual einsum result: [R,Hkv,G,W].
        cases.append(dict(Hq=heads, Hkv=kv_heads, G=heads // kv_heads,
                          helper_R=limit, actual_rows=rows, W=width,
                          score_elements=elements, score_payload_MiB=elements * 4 / 2**20,
                          exceeds_score_budget=elements > budget))
    print(json.dumps(dict(evidence="source-derived shape arithmetic; no tensor allocation",
                          source_sha256=source_hash, config=config, config_sha256=config_hash,
                          budget_elements=budget, budget_MiB=512, cases=cases), indent=2))
    if args.require_compliance and any(c["exceeds_score_budget"] for c in cases):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
