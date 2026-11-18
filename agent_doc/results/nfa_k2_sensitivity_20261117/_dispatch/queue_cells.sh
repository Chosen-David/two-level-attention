#!/usr/bin/env bash
# 串行消费 K2:TASK 单元；每格只在空卡点火箭，并在20秒内做三重验证。
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU quality|operator K2:TASK [K2:TASK ...]" >&2
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
RUNNER=$DISPATCH/run_${MODE}_cell.sh
OUTROOT=${OUTROOT:-$HOME/ak_verify_results/nfa_k2_sensitivity_ebcae298_20261117}
mkdir -p "$OUTROOT/logs"
echo "$$" > "$OUTROOT/logs/queue_${MODE}_gpu${GPU}_$$.pid"
echo "QUEUE_START mode=$MODE gpu=$GPU pid=$$ cells=$*"

for spec in "$@"; do
  K2=${spec%%:*}
  TASK=${spec#*:}
  [[ "$K2" =~ ^(256|512|1024|2048)$ ]]
  [[ "$TASK" =~ ^(hotpotqa|gov_report)$ ]]
  LOG=$OUTROOT/logs/${MODE}_${TASK}_k2${K2}.log
  while true; do
    apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ -z "$apps" && "$used" -lt 500 ]]; then
      break
    fi
    echo "QUEUE_WAIT_GPU_FREE mode=$MODE gpu=$GPU next=$spec used_mib=$used apps=${apps:-none}"
    sleep 30
  done

  bash "$RUNNER" "$K2" "$TASK" "$GPU" > "$LOG" 2>&1 &
  child=$!
  echo "$child" > "$OUTROOT/logs/${MODE}_${TASK}_k2${K2}.launcher_pid"
  echo "QUEUE_LAUNCHED mode=$MODE gpu=$GPU cell=$spec launcher_pid=$child"
  sleep 20
  if ps -p "$child" >/dev/null 2>&1; then
    ps -p "$child" -o pid,etime,args
    nvidia-smi -i "$GPU" --query-gpu=index,memory.used --format=csv,noheader
    apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader)
    if [[ "$apps" == *$'\n'* ]]; then
      echo "QUEUE_IGNITION_FAILED_SHARED_GPU mode=$MODE gpu=$GPU cell=$spec apps=$apps" >&2
      exit 4
    fi
    tail -n 12 "$LOG"
    echo "TRIPLE_VERIFIED mode=$MODE gpu=$GPU cell=$spec process=alive compute_apps=${apps:-model-loading} log=present"
  fi
  if wait "$child"; then
    echo "QUEUE_CELL_DONE mode=$MODE gpu=$GPU cell=$spec"
  else
    status=$?
    echo "QUEUE_CELL_FAILED mode=$MODE gpu=$GPU cell=$spec status=$status" >&2
    exit "$status"
  fi
done
echo "QUEUE_ALL_DONE mode=$MODE gpu=$GPU cells=$*"
