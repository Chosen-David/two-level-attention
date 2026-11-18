#!/usr/bin/env bash
# Qwen3-8B NFA K2 敏感性：一个 K2×LongBench 数据集质量单元独占一张卡。
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 K2 TASK GPU" >&2
  exit 2
fi
K2=$1
TASK=$2
GPU=$3
EXPECTED_HEAD=${EXPECTED_HEAD:-ebcae298297c18f68ec224de611dc8b922ab9e95}
REPO=${REPO:-/tmp/nfa-k2-sensitivity-20261117}
OUTROOT=${OUTROOT:-$HOME/ak_verify_results/nfa_k2_sensitivity_ebcae298_20261117}
DATA_ROOT=${DATA_ROOT:-$HOME/datasets/LongBench/data}
PYTHON=${PYTHON:-/home/fluentllmenv/bin/python}

case "$K2" in
  256|512) ARM="tail_gate_on_k2${K2}" ;;
  1024) ARM=tail_gate_on ;;
  2048) ARM=tail_gate_on_k22048 ;;
  *) echo "unsupported K2: $K2" >&2; exit 2 ;;
esac
case "$TASK" in
  hotpotqa|gov_report) ;;
  *) echo "unsupported task: $TASK" >&2; exit 2 ;;
esac
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
[[ -z "$(git -C "$REPO" diff --binary HEAD)" ]]
[[ -f "$HOME/Qwen3-8B/config.json" ]]
[[ -f "$DATA_ROOT/$TASK.jsonl" ]]

OUT=$OUTROOT/cells/$TASK/$ARM
mkdir -p "$OUT" "$OUTROOT/logs" "$OUTROOT/locks"
exec 9>"$OUTROOT/locks/quality_${TASK}_k2${K2}.lock"
if ! flock -n 9; then
  echo "CELL_OWNER_CONFLICT mode=quality task=$TASK k2=$K2" >&2
  exit 3
fi

validate_cell() {
  K2="$K2" TASK="$TASK" OUT="$OUT" EXPECTED_HEAD="$EXPECTED_HEAD" "$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUT"])
task = os.environ["TASK"]
k2 = int(os.environ["K2"])
manifest = json.loads((root / "quality_manifest_s0of1.json").read_text())
runtime = manifest["runtime"]
config = runtime["nfa_selector_config"]
entry = manifest["tasks"][task]
empty_diff = hashlib.sha256(b"").hexdigest()
assert manifest["schema"] == "longbench-official-quality-manifest-v1"
assert manifest["verification"] == "pending-independent-verification"
assert manifest["method"] == "nfa"
assert list(manifest["tasks"]) == [task]
assert entry["rows"] == entry["receipt_rows"] == 200
assert runtime["source"]["head"] == os.environ["EXPECTED_HEAD"]
assert runtime["source"]["source_diff_sha256"] == empty_diff
assert runtime["source"]["quality_source_diff_sha256"] == empty_diff
assert runtime["execution"] == "eager" and runtime["num_layers"] == 36
assert config["level1_topk"] == 128 and config["level2_topk"] == k2
assert config["dynamic_gate_epsilon"] == 0.1
assert config["dynamic_gate_rows"] == 256
assert config["near_tokens"] == "off"
print(f"CELL_DONE mode=quality task={task} k2={k2} rows=200 output={root}", flush=True)
PY
}

if [[ -f "$OUT/quality_manifest_s0of1.json" ]]; then
  validate_cell
  echo "CELL_ALREADY_COMPLETE mode=quality task=$TASK k2=$K2"
  exit 0
fi

export PYTHONPATH="$REPO:$HOME/local_pylibs/venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES=$GPU
cd "$REPO"
"$PYTHON" -u benchmark/efficiency/benchmark_longbench_quality.py \
  --method nfa \
  --execution eager \
  --nfa-subspace tail \
  --nfa-subspace-dim 32 \
  --nfa-gate on \
  --nfa-gate-epsilon 0.1 \
  --nfa-level2-topk "$K2" \
  --nfa-far-method minmax \
  --model-path "$HOME/Qwen3-8B" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --tasks "$TASK" \
  --device cuda:0
validate_cell
