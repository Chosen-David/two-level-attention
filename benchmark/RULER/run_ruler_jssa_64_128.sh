#!/bin/bash
# JSSA Qwen3-8B RULER 64K/128K 补齐派单（单机 4 卡并行）
# 用户指令 2026-10-20：「8B 补齐」；JSSA 8B RULER 32K 已有（ts=10181901），64K/128K 缺口。
# 用法：bash benchmark/RULER/run_ruler_jssa_64_128.sh <start_card>  (start_card=0 通常)
# 每卡串行跑 run_ruler_jssa.sh 的 2 长度 × 若干任务；N=100 对齐基线。
set -u
REPO=/home/wangyuanshuo02/two-level-attention
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
N=${1:-100}
NCARDS=${2:-4}
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/tmp/jssa_ruler_64_128_${TS}
mkdir -p "$LOG_DIR"
echo "TS=$TS" | tee "$LOG_DIR/ts.txt"
FAILED=0

i=0
for T in $TASKS; do
  card=$(( i % NCARDS ))
  (
    echo "[$(date +%H:%M:%S)] card=$card task=$T start" >> "$LOG_DIR/card$card.log"
    bash "$REPO/benchmark/RULER/run_ruler_jssa.sh" "$card" "$TS" "$N" "65536 131072" "$T" \
      >> "$LOG_DIR/card$card.log" 2>&1
    local_rc=$?
    echo "[$(date +%H:%M:%S)] card=$card task=$T done rc=$local_rc" >> "$LOG_DIR/card$card.log"
    if [ "$local_rc" -ne 0 ]; then echo "FAILED task=$T" >> "$LOG_DIR/failed.txt"; fi
  ) &
  i=$(( i + 1 ))
done
wait
if [ -s "$LOG_DIR/failed.txt" ]; then
  echo "ALL DONE jssa_64_128 (FAILED see $LOG_DIR/failed.txt) TS=$TS"
  exit 1
fi
echo "ALL DONE jssa_64_128 TS=$TS"
