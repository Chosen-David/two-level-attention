#!/usr/bin/env bash
# 串行消费 EPS:K2:TASK 单元；每格只在空卡点火箭，并在20秒内做三重验证。
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU quality|operator EPS:K2:TASK [EPS:K2:TASK ...]" >&2
  exit 2
fi
GPU=$1
MODE=$2
shift 2
case "$MODE" in
  quality|operator) ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac
DISPATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER=$DISPATCH/run_${MODE}_grid_cell.sh
OUTROOT=${OUTROOT:-$HOME/ak_verify_results/nfa_k2_sensitivity_ebcae298_20261117}
mkdir -p "$OUTROOT/logs"
echo "$$" > "$OUTROOT/logs/grid_queue_${MODE}_gpu${GPU}_$$.pid"
echo "GRID_QUEUE_START mode=$MODE gpu=$GPU pid=$$ cells=$*"
for spec in "$@"; do
  IFS=: read -r EPS K2 TASK <<< "$spec"
  [[ "$EPS" =~ ^(0.01|0.05|0.1|0.2|0.5)$ ]]
  [[ "$K2" =~ ^(256|512|1024|2048)$ ]]
  [[ "$TASK" =~ ^(hotpotqa|gov_report)$ ]]
  LOG=$OUTROOT/logs/${MODE}_${TASK}_eps${EPS}_k2${K2}.log
  while true; do
    apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ -z "$apps" && "$used" -lt 500 ]]; then
      break
    fi
    echo "GRID_QUEUE_WAIT_GPU_FREE mode=$MODE gpu=$GPU next=$spec used_mib=$used apps=${apps:-none}"
    sleep 30
  done
  bash "$RUNNER" "$EPS" "$K2" "$TASK" "$GPU" > "$LOG" 2>&1 &
  child=$!
  echo "$child" > "$OUTROOT/logs/${MODE}_${TASK}_eps${EPS}_k2${K2}.launcher_pid"
  echo "GRID_QUEUE_LAUNCHED mode=$MODE gpu=$GPU cell=$spec launcher_pid=$child"
  sleep 20
  if ps -p "$child" >/dev/null 2>&1; then
    ps -p "$child" -o pid,etime,args
    nvidia-smi -i "$GPU" --query-gpu=index,memory.used --format=csv,noheader
    apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader)
    if [[ "$apps" == *$'\n'* ]]; then
      echo "GRID_QUEUE_IGNITION_FAILED_SHARED_GPU mode=$MODE gpu=$GPU cell=$spec apps=$apps" >&2
      exit 4
    fi
    tail -n 12 "$LOG"
    echo "GRID_TRIPLE_VERIFIED mode=$MODE gpu=$GPU cell=$spec process=alive compute_apps=${apps:-model-loading} log=present"
  fi
  if wait "$child"; then
    echo "GRID_QUEUE_CELL_DONE mode=$MODE gpu=$GPU cell=$spec"
  else
    status=$?
    echo "GRID_QUEUE_CELL_FAILED mode=$MODE gpu=$GPU cell=$spec status=$status" >&2
    exit "$status"
  fi
done
echo "GRID_QUEUE_ALL_DONE mode=$MODE gpu=$GPU cells=$*"
