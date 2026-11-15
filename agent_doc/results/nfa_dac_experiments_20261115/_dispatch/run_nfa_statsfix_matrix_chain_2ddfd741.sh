#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <SHARD_INDEX> <CHAIN_ID>}"
SHARD="${2:?usage: $0 <GPU> <SHARD_INDEX> <CHAIN_ID>}"
CHAIN="${3:?usage: $0 <GPU> <SHARD_INDEX> <CHAIN_ID>}"
WT=/tmp/nfa-dac-statsfix-formal-2ddfd741-20261116
EXPECTED_HEAD=2ddfd7416aedc03f0e43c6ea115c25c7212fa974
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
ROOT="$HOME/ak_verify_results/nfa_dac_statsfix_matrix_2ddfd741_20261116"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER"
case "$CHAIN" in
  pc02a) JOBS=("32768 full off" "65536 full off" "131072 full off") ;;
  pc02b) JOBS=("32768 full on" "65536 full on" "131072 full on") ;;
  pc31) JOBS=("32768 tail on" "65536 tail on" "131072 tail off") ;;
  pc32) JOBS=("65536 tail off" "131072 tail on") ;;
  *) printf 'unknown chain %s\n' "$CHAIN" >&2; exit 3 ;;
esac
if [[ "$SHARD" != 0 && "$SHARD" != 1 ]]; then
  printf 'shard must be 0 or 1\n' >&2
  exit 3
fi
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/${CHAIN}_gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s chain already running\n' "$GPU" >&2
  exit 3
fi
if [[ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  printf 'unexpected source HEAD\n' >&2
  exit 3
fi
if [[ -n "$(git -C "$WT" status --short)" ]]; then
  printf 'source worktree is not clean\n' >&2
  exit 3
fi
for JOB in "${JOBS[@]}"; do
  read -r LENGTH SUBSPACE GATE <<<"$JOB"
  ARM="nfa_${SUBSPACE}_gate_${GATE}"
  OUTPUT="$ROOT/$ARM"
  LOG="$ROOT/_logs/${ARM}_L${LENGTH}_s${SHARD}of2.log"
  MANIFEST="$OUTPUT/quality_manifest_L${LENGTH}_s${SHARD}of2.json"
  if [[ -e "$MANIFEST" ]]; then
    printf 'REFUSE existing formal manifest %s\n' "$MANIFEST" >&2
    exit 5
  fi
  apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
  if [[ -n "$apps" ]]; then
    printf 'GPU%s occupied by compute PID=%s\n' "$GPU" "$apps" >&2
    exit 4
  fi
  printf 'START host=%s gpu=%s chain=%s shard=%s rows=50 head=%s shell_pid=%s time=%s\n' \
    "$(hostname)" "$GPU" "$CHAIN" "$SHARD" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
  env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" -m benchmark.efficiency.benchmark_ruler_quality \
    --method nfa --execution eager --nfa-subspace "$SUBSPACE" --nfa-gate "$GATE" \
    --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
    --task niah_multikey_2 --context-length "$LENGTH" \
    --data-root "$DATA" --output-dir "$OUTPUT" \
    --shard-index "$SHARD" --num-shards 2 \
    --device cuda:0 --visible-len 8192 >>"$LOG" 2>&1
  if [[ ! -f "$MANIFEST" ]]; then
    printf 'formal shard exited without complete manifest %s\n' "$MANIFEST" >&2
    exit 6
  fi
  printf 'DONE host=%s gpu=%s chain=%s shard=%s rows=50 time=%s\n' \
    "$(hostname)" "$GPU" "$CHAIN" "$SHARD" "$(date -Is)" >>"$LOG"
done
printf 'host=%s gpu=%s chain=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$CHAIN" "$(date -Is)" \
  >"$ROOT/_status/${CHAIN}_gpu${GPU}.done"
