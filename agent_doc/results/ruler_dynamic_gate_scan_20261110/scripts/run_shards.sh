#!/usr/bin/env bash
set -euo pipefail
GPU="$1"
REPO=/tmp/ruler-gate-scan-7798ddb6-20261110
HEAD=7798ddb6569071d1238ca21e7ddf925a18d8dcd5
OUT=/tmp/ruler_dynamic_gate_decision_scan_20261110
SCRIPT=/tmp/ruler_dynamic_gate_scan_20261110.py
DATA_ROOT=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS=(cwe fwe niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue niah_single_1 niah_single_2 niah_single_3 vt)
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
for LENGTH in 131072 65536 32768; do
  for index in "${!TASKS[@]}"; do
    if (( index % 4 != GPU )); then
      continue
    fi
    TASK="${TASKS[$index]}"
    OUTPUT="$OUT/ruler_${LENGTH}_task_${TASK}.json"
    LOG="$OUT/ruler_${LENGTH}_task_${TASK}.log"
    if [[ -f "$OUTPUT" ]] && python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("complete") and p.get("realized_cells")==1 and p.get("freeze_mode")=="decision-only-no-reclaim" else 1)' "$OUTPUT"; then
      printf '%s\n' "{\"status\":\"skip-complete\",\"gpu\":$GPU,\"length\":$LENGTH,\"task\":\"$TASK\"}" >> "$LOG"
      continue
    fi
    printf '%s\n' "{\"status\":\"start-decision-only\",\"gpu\":$GPU,\"length\":$LENGTH,\"task\":\"$TASK\"}" >> "$LOG"
    DATA="$DATA_ROOT/$LENGTH" CONTEXT_LENGTH="$LENGTH" OUTPUT="$OUTPUT" \
      TASKS="$TASK" SAMPLE_INDICES=0 NFA_REPO="$REPO" \
      EXPECTED_REPO_HEAD="$HEAD" MODEL=/home/wangyuanshuo02/Qwen3-8B \
      python3 "$SCRIPT" >> "$LOG" 2>&1
    printf '%s\n' "{\"status\":\"done\",\"gpu\":$GPU,\"length\":$LENGTH,\"task\":\"$TASK\"}" >> "$LOG"
  done
done
printf '%s\n' "{\"status\":\"decision-only-shard-complete\",\"gpu\":$GPU}" > "$OUT/shard${GPU}.done"
