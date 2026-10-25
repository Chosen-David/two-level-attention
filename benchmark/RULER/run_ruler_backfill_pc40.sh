#!/bin/bash
# pc40 补跑：FASA 128K(fwe,vt) + AsyncTLS 128K(vt)（8B RULER 128K 缺口）
# 用户指令 2026-10-20：「8B 补齐」；128K 档 asynctls 缺 vt、fasa 缺 fwe/vt。
# 卡0：fasa fwe；卡1：fasa vt → asynctls vt（串行）。
set -u
REPO=/home/wangyuanshuo02/two-level-attention
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/tmp/backfill_pc40_${TS}
mkdir -p "$LOG_DIR"
echo "TS=$TS" | tee "$LOG_DIR/ts.txt"

(
  bash "$REPO/benchmark/RULER/run_ruler_fasa.sh" 0 "$TS" 100 "131072" "fwe" \
    >> "$LOG_DIR/card0.log" 2>&1
  echo "[$(date +%H:%M:%S)] card0 fasa fwe done rc=$?" >> "$LOG_DIR/card0.log"
) &
(
  bash "$REPO/benchmark/RULER/run_ruler_fasa.sh" 1 "$TS" 100 "131072" "vt" \
    >> "$LOG_DIR/card1.log" 2>&1
  echo "[$(date +%H:%M:%S)] card1 fasa vt done rc=$?" >> "$LOG_DIR/card1.log"
  bash "$REPO/benchmark/RULER/run_ruler_asynctls.sh" 1 "$TS" 100 "131072" "vt" \
    >> "$LOG_DIR/card1.log" 2>&1
  echo "[$(date +%H:%M:%S)] card1 asynctls vt done rc=$?" >> "$LOG_DIR/card1.log"
) &
wait
echo "ALL DONE backfill_pc40 TS=$TS"
