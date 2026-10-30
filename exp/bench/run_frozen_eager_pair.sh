#!/usr/bin/env bash
# 每个长度独立进程；起跑前硬查空卡，全程 30 秒旁路监控。
set -euo pipefail

MODEL=${1:?model path}
DATA_ROOT=${2:?RULER data root}
OUT=${3:?output root}
GPU=${4:-0}
LENGTHS=${5:-8192,32768,65536,131072}
MAX_NEW=${6:-16}
REPS=${7:-1}
PYTHON=${PYTHON:-/home/fluentllmenv/bin/python3}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if (( used >= 500 || apps > 0 )); then
  printf 'GPU%s 不符合空卡门：used=%sMiB compute_apps=%s\n' "$GPU" "$used" "$apps" >&2
  exit 2
fi
mkdir -p "$OUT"
monitor_log="$OUT/gpu_monitor_$(date +%Y%m%d_%H%M%S)_$$.log"
(
  while true; do
    nvidia-smi -i "$GPU" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
    sleep 30
  done
) >"$monitor_log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
printf '%s\n' "$monitor_log" >"$OUT/latest_monitor_path.txt"

IFS=',' read -ra lengths <<<"$LENGTHS"
for length in "${lengths[@]}"; do
  states=dual
  if (( length == 8192 )); then
    states=keep,far,near,both
  fi
  run_out="$OUT/L$length"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" \
    "$PYTHON" -u "$REPO/exp/bench/check_qwen_frozen_eager.py" \
    --model "$MODEL" --data-root "$DATA_ROOT" --context-length "$length" \
    --task niah_single_1 --states "$states" --max-new "$MAX_NEW" \
    --mode all --reps "$REPS" --profile --out "$run_out" \
    2>&1 | tee "$run_out.stdout.log"
done
