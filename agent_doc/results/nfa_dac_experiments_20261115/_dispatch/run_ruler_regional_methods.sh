#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <CELL> <SHARD>; CELL=c2_32768|c2_65536|c2_131072|c3_32768|c3_65536|c3_131072}"
CELL="${2:?usage: $0 <GPU> <CELL> <SHARD>}"
SHARD="${3:?usage: $0 <GPU> <CELL> <SHARD>}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD to the clean producer commit}"
WT="${WT:-/tmp/nfa-dac-ruler-regional-${EXPECTED_HEAD:0:8}}"
PYTHON_BIN="${PYTHON_BIN:-/home/fluentllmenv/bin/python}"
ROOT="${ROOT:-$HOME/ak_verify_results/nfa_dac_ruler_regional_${EXPECTED_HEAD:0:8}_20261116}"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER"
case "$CELL" in
  c2_32768)  LENGTH=32768;  NEAR=minmax; FAR=minmax; ARM=nfa_tail_gate_off_near_minmax ;;
  c2_65536)  LENGTH=65536;  NEAR=minmax; FAR=minmax; ARM=nfa_tail_gate_off_near_minmax ;;
  c2_131072) LENGTH=131072; NEAR=minmax; FAR=minmax; ARM=nfa_tail_gate_off_near_minmax ;;
  c3_32768)  LENGTH=32768;  NEAR=avg;    FAR=avg;    ARM=nfa_tail_gate_off_far_avg ;;
  c3_65536)  LENGTH=65536;  NEAR=avg;    FAR=avg;    ARM=nfa_tail_gate_off_far_avg ;;
  c3_131072) LENGTH=131072; NEAR=avg;    FAR=avg;    ARM=nfa_tail_gate_off_far_avg ;;
  *) printf 'unknown cell %s\n' "$CELL" >&2; exit 3 ;;
esac
if [[ "$SHARD" != 0 && "$SHARD" != 1 ]]; then
  printf 'shard must be 0 or 1\n' >&2
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
OUTPUT="$ROOT/$ARM"
MANIFEST="$OUTPUT/quality_manifest_L${LENGTH}_s${SHARD}of2.json"
LOG="$ROOT/_logs/${ARM}_L${LENGTH}_s${SHARD}of2.log"
if [[ -e "$MANIFEST" ]]; then
  printf 'REFUSE existing formal manifest %s\n' "$MANIFEST" >&2
  exit 5
fi
apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [[ -n "$apps" ]]; then
  printf 'GPU%s occupied by compute PID=%s\n' "$GPU" "$apps" >&2
  exit 4
fi
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/${CELL}_gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s cell already running\n' "$GPU" >&2
  exit 3
fi
printf 'START host=%s gpu=%s cell=%s shard=%s rows=50 near=%s far=%s head=%s shell_pid=%s time=%s\n' \
  "$(hostname)" "$GPU" "$CELL" "$SHARD" "$NEAR" "$FAR" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" -m benchmark.efficiency.benchmark_ruler_quality \
  --method nfa --execution eager --nfa-subspace tail --nfa-gate off \
  --nfa-near-method "$NEAR" --nfa-far-method "$FAR" \
  --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
  --task niah_multikey_2 --context-length "$LENGTH" \
  --data-root "$DATA" --output-dir "$OUTPUT" \
  --shard-index "$SHARD" --num-shards 2 \
  --device cuda:0 --visible-len 8192 >>"$LOG" 2>&1
if [[ ! -f "$MANIFEST" ]]; then
  printf 'formal shard exited without complete manifest %s\n' "$MANIFEST" >&2
  exit 6
fi
printf 'DONE host=%s gpu=%s cell=%s shard=%s rows=50 time=%s\n' \
  "$(hostname)" "$GPU" "$CELL" "$SHARD" "$(date -Is)" >>"$LOG"
printf 'host=%s gpu=%s cell=%s shard=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$CELL" "$SHARD" "$(date -Is)" \
  >"$ROOT/_status/${CELL}_s${SHARD}of2.done"
