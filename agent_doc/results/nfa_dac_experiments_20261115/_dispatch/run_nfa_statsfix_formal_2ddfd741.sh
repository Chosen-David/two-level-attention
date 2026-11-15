#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <SHARD_INDEX>}"
SHARD="${2:?usage: $0 <GPU> <SHARD_INDEX>}"
WT=/tmp/nfa-dac-statsfix-formal-2ddfd741-20261116
EXPECTED_HEAD=2ddfd7416aedc03f0e43c6ea115c25c7212fa974
ROOT="$HOME/ak_verify_results/nfa_dac_statsfix_2ddfd741_20261116"
OUTPUT="$ROOT/nfa_tail_gate_off"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER"
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s formal shard already running\n' "$GPU" >&2
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
apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [[ -n "$apps" ]]; then
  printf 'GPU%s occupied by compute PID=%s\n' "$GPU" "$apps" >&2
  exit 4
fi
LOG="$ROOT/_logs/nfa_tail_gate_off_L32768_s${SHARD}of4.log"
MANIFEST="$OUTPUT/quality_manifest_L32768_s${SHARD}of4.json"
if [[ -e "$MANIFEST" ]]; then
  printf 'REFUSE existing formal manifest %s\n' "$MANIFEST" >&2
  exit 5
fi
printf 'START host=%s gpu=%s shard=%s rows=25 head=%s shell_pid=%s time=%s\n' \
  "$(hostname)" "$GPU" "$SHARD" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  /usr/bin/python -m benchmark.efficiency.benchmark_ruler_quality \
  --method nfa --execution eager --nfa-subspace tail --nfa-gate off \
  --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
  --task niah_multikey_2 --context-length 32768 \
  --data-root "$DATA" --output-dir "$OUTPUT" \
  --shard-index "$SHARD" --num-shards 4 \
  --device cuda:0 --visible-len 8192 >>"$LOG" 2>&1
if [[ ! -f "$MANIFEST" ]]; then
  printf 'formal shard exited without complete manifest %s\n' "$MANIFEST" >&2
  exit 6
fi
printf 'DONE host=%s gpu=%s shard=%s rows=25 time=%s\n' \
  "$(hostname)" "$GPU" "$SHARD" "$(date -Is)" >>"$LOG"
printf 'host=%s gpu=%s shard=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$SHARD" "$(date -Is)" \
  >"$ROOT/_status/gpu${GPU}.done"
