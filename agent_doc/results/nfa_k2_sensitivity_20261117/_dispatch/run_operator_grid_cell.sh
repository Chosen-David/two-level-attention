#!/usr/bin/env bash
# Qwen3-8B NFA gate epsilon × K2：独立 nsys 进程测量 L1+L2 与主 attention。
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 EPS K2 TASK GPU" >&2
  exit 2
fi
EPS=$1
K2=$2
TASK=$3
GPU=$4
EXPECTED_HEAD=${EXPECTED_HEAD:-ebcae298297c18f68ec224de611dc8b922ab9e95}
REPO=${REPO:-/tmp/nfa-k2-sensitivity-20261117}
OUTROOT=${OUTROOT:-$HOME/ak_verify_results/nfa_k2_sensitivity_ebcae298_20261117}
DATA_ROOT=${DATA_ROOT:-$HOME/datasets/LongBench/data}
PYTHON=${PYTHON:-/home/fluentllmenv/bin/python}
case "$EPS" in
  0.01|0.05|0.1|0.2|0.5) ;;
  *) echo "unsupported epsilon: $EPS" >&2; exit 2 ;;
esac
case "$K2" in
  256|512|1024|2048) ;;
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
command -v nsys >/dev/null
if [[ "$EPS" == 0.1 ]]; then
  OUT=$OUTROOT/operators/$TASK/k2$K2
  LOCK=operator_${TASK}_k2${K2}.lock
else
  OUT=$OUTROOT/operators/$TASK/eps$EPS/k2$K2
  LOCK=operator_${TASK}_eps${EPS}_k2${K2}.lock
fi
mkdir -p "$OUT" "$OUTROOT/logs" "$OUTROOT/locks"
exec 9>"$OUTROOT/locks/$LOCK"
if ! flock -n 9; then
  echo "CELL_OWNER_CONFLICT mode=operator task=$TASK epsilon=$EPS k2=$K2" >&2
  exit 3
fi
export PYTHONPATH="$REPO:$HOME/local_pylibs/venv/lib/python3.11/site-packages"
cd "$REPO"
"$PYTHON" -u benchmark/efficiency/run_nfa_perf_matrix.py \
  --repo "$REPO" \
  --model-path "$HOME/Qwen3-8B" \
  --data-root "$OUTROOT/data" \
  --task "$TASK" \
  --sample-index 0 \
  --input-provenance deterministic-longbench-context-repetition \
  --longbench-data "$DATA_ROOT" \
  --output-dir "$OUT" \
  --gpu "$GPU" \
  --arms nfa-operator-profile \
  --lengths 32768 \
  --batches 1 \
  --execution eager \
  --warmup-steps 3 \
  --profile-repeat 5 \
  --decode-steps 32 \
  --visible-len 8192 \
  --subspace tail \
  --l2-subspace same \
  --level2-topk "$K2" \
  --gate-epsilon "$EPS" \
  --force-skip-mode natural \
  --max-attempts 2
"$PYTHON" -u benchmark/efficiency/run_nfa_perf_matrix.py \
  --repo "$REPO" \
  --model-path "$HOME/Qwen3-8B" \
  --data-root "$OUTROOT/data" \
  --task "$TASK" \
  --sample-index 0 \
  --input-provenance deterministic-longbench-context-repetition \
  --longbench-data "$DATA_ROOT" \
  --output-dir "$OUT" \
  --gpu "$GPU" \
  --arms nfa-operator-profile \
  --lengths 32768 \
  --batches 1 \
  --execution eager \
  --warmup-steps 3 \
  --profile-repeat 5 \
  --decode-steps 32 \
  --visible-len 8192 \
  --subspace tail \
  --l2-subspace same \
  --level2-topk "$K2" \
  --gate-epsilon "$EPS" \
  --force-skip-mode natural \
  --max-attempts 2 \
  --summarize-only
EPS="$EPS" K2="$K2" TASK="$TASK" OUT="$OUT" DATA_ROOT="$DATA_ROOT" \
EXPECTED_HEAD="$EXPECTED_HEAD" "$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ["OUT"])
task = os.environ["TASK"]
k2 = int(os.environ["K2"])
epsilon = float(os.environ["EPS"])
path = root / "nfa-operator-profile/nfa-operator-profile_l32768_b1_eager.json"
row = json.loads(path.read_text())
summary = json.loads((root / "summary.json").read_text())
empty_diff = hashlib.sha256(b"").hexdigest()
assert row["schema"] == "nfa-selector-and-e2e-v2"
assert row["source"]["head"] == os.environ["EXPECTED_HEAD"]
assert row["source"]["diff_sha256"] == empty_diff
assert row["gpu_hygiene"] == "clean"
assert row["level2_topk"] == k2
assert row["task"] == task and row["sample_index"] == 0
assert row["context_length"] == 32768 and row["batch_size"] == 1
assert row["execution"] == "eager" and row["measurement_mode"] == "operator-profile"
assert row["subspace"] == "tail" and row["l2_subspace"] == "same"
assert row["gate_epsilon"] == epsilon and row["force_skip_mode"] == "natural"
assert row["operator_mode"] == "off" and row["visible_len"] == 8192
assert row["warmup_steps"] == 3 and row["profile_repeat"] == 5
assert row["decode_steps"] == 32 and row["profiled_decode_steps"] == 5
assert row["input_provenance"] == "deterministic-longbench-context-repetition"
assert row["input"]["sha256"] == hashlib.sha256(
    (Path(os.environ["DATA_ROOT"]) / f"{task}.jsonl").read_bytes()).hexdigest()
assert row["graph_capture_count"] == 0
assert row["nsys_profile_finalized"] is True
assert row["region_kernel_audit"]["passed"] is True
for key in ("report", "sqlite"):
    artifact = Path(row["nsys_receipt"][f"{key}_path"])
    assert artifact.is_file()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == row["nsys_receipt"][f"{key}_sha256"]
for key in ("l1_ms", "l2_ms", "main_attention_ms", "nfa_operator_total_ms"):
    assert row["operator_summary"][key]["mean_ms_per_step"] > 0, key
assert summary["validation_status"] == "pending-independent-verification"
assert summary["arguments"]["level2_topk"] == k2
assert summary["arguments"]["gate_epsilon"] == epsilon
assert len(summary["records"]) == 1
print(f"CELL_DONE mode=operator task={task} epsilon={epsilon} k2={k2} output={root}", flush=True)
print(json.dumps({key: row["operator_summary"][key]["mean_ms_per_step"]
                  for key in ("l1_ms", "l2_ms", "main_attention_ms")}), flush=True)
PY
