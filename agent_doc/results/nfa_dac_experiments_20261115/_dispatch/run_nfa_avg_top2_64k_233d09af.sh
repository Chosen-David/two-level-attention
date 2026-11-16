#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <SUBSPACE> <GATE> <SHARD_INDEX>}"
SUBSPACE="${2:?usage: $0 <GPU> <SUBSPACE> <GATE> <SHARD_INDEX>}"
GATE="${3:?usage: $0 <GPU> <SUBSPACE> <GATE> <SHARD_INDEX>}"
SHARD="${4:?usage: $0 <GPU> <SUBSPACE> <GATE> <SHARD_INDEX>}"
WT=/tmp/nfa-dac-avg-top2-formal-233d09af-20261116
EXPECTED_HEAD=233d09afe535a697dc30038b3fe59f06b6304e00
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$HOME/ak_verify_results/nfa_dac_avg_top2_64k_233d09af_20261116"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER"
LENGTH=65536
if [[ "$SUBSPACE" != full && "$SUBSPACE" != tail ]]; then
  printf 'subspace must be full or tail\n' >&2
  exit 3
fi
if [[ "$GATE" != off && "$GATE" != on ]]; then
  printf 'gate must be off or on\n' >&2
  exit 3
fi
if [[ "$SHARD" != 0 && "$SHARD" != 1 ]]; then
  printf 'shard must be 0 or 1\n' >&2
  exit 3
fi
ARM="nfa_${SUBSPACE}_gate_${GATE}_far_avg_top2"
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/${ARM}_s${SHARD}_gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'unit already running\n' >&2
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
printf 'START host=%s gpu=%s arm=%s shard=%s rows=50 head=%s shell_pid=%s time=%s\n' \
  "$(hostname)" "$GPU" "$ARM" "$SHARD" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" -m benchmark.efficiency.benchmark_ruler_quality \
  --method nfa --execution eager --nfa-subspace "$SUBSPACE" --nfa-gate "$GATE" \
  --nfa-far-method avg_top2 \
  --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
  --task niah_multikey_2 --context-length "$LENGTH" \
  --data-root "$DATA" --output-dir "$OUTPUT" \
  --shard-index "$SHARD" --num-shards 2 \
  --device cuda:0 --visible-len 8192 >>"$LOG" 2>&1
if [[ ! -f "$MANIFEST" ]]; then
  printf 'formal shard exited without complete manifest %s\n' "$MANIFEST" >&2
  exit 6
fi
printf 'DONE host=%s gpu=%s arm=%s shard=%s rows=50 time=%s\n' \
  "$(hostname)" "$GPU" "$ARM" "$SHARD" "$(date -Is)" >>"$LOG"
printf 'host=%s gpu=%s arm=%s shard=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$ARM" "$SHARD" "$(date -Is)" \
  >"$ROOT/_status/${ARM}_s${SHARD}_gpu${GPU}.done"
