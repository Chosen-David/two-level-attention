#!/bin/bash
# 单池 BSA（TLI 单池变体）RULER 派单：单一 method（minmax/avg）两级筛选
# 用户指令 2026-10-21：「增加单池子 BSA（单一 method,minmax/avg 的两级筛选），出 Qwen8B 数据」。
# 语义：α=0 β=0 单池（无 far/near 双池分区）+ L1 块粗筛（far_method 单一源）top-K1
#       + L2 细筛 top-K2；预算与 TLI 主方法同款（bs=64、K1=128、K2=1024、sink=4、swa=128）。
# 用法：bash benchmark/RULER/run_ruler_bsa.sh [ts] [method...]
# 无 method 时默认 minmax avg；一卡一进程，每轮最多 8 并发。
set -u
REPO=/home/wangyuanshuo02/two-level-attention
MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
METHODS="${*:2}"; [ -z "$METHODS" ] && METHODS="minmax avg"
TS=${1:-$(date +%Y%m%d-%H%M%S)}
OUT=$REPO/exp/results_ruler/Qwen3-8B
LOG_DIR=/tmp/bsa_ruler_${TS}
mkdir -p "$LOG_DIR"
echo "TS=$TS METHODS=[$METHODS]" | tee "$LOG_DIR/ts.txt"

run_len() {
  local L=$1
  local YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  local -a JOB=()
  for M in $METHODS; do
    for T in $TASKS; do
      JOB+=("$M|$T")
    done
  done
  local n=${#JOB[@]}
  local start=0
  while [ "$start" -lt "$n" ]; do
    local end=$(( start + 8 ))
    [ "$end" -gt "$n" ] && end=$n
    local n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$n_gpu" -eq 0 ] && n_gpu=1
    local i=0
    for (( j=start; j<end; j++ )); do
      local M=${JOB[$j]%|*}
      local T=${JOB[$j]#*|}
      local card=$(( i % n_gpu ))
      (
        echo "[$(date +%H:%M:%S)] L=$L M=$M task=$T card=$card start" >> "$LOG_DIR/card$card.log"
        PYTHONPATH=~/local_pylibs:$REPO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          CUDA_VISIBLE_DEVICES=$card python -u -m benchmark.RULER.pred_ruler \
          --model Qwen3-8B --model_path "$MODEL_PATH" \
          --task "$T" --context_length "$L" --method tli --t "$TS" \
          --data-root "$DATA" --output-dir "$OUT" \
          --tia_level1_topk 128 --tia_level2_topk 1024 \
          --tli_alpha 0 --tli_beta 0 --tli_gamma 0.625 \
          --tli_far_method "$M" \
          --tli_enable_kmeans false --tli_enable_layer_skip false \
          --tli_subspace tail --tia_level2_cmp_ratio 4 \
          --tli_sink_tokens 4 --tli_swa_size 128 \
          --pred_postfix _1024 --max-num 100 $YARN \
          >> "$LOG_DIR/card$card.log" 2>&1
        local rc=$?
        echo "[$(date +%H:%M:%S)] L=$L M=$M task=$T card=$card done rc=$rc" >> "$LOG_DIR/card$card.log"
        if [ "$rc" -ne 0 ]; then echo "FAILED L=$L M=$M task=$T" >> "$LOG_DIR/failed.txt"; fi
      ) &
      i=$(( i + 1 ))
    done
    wait
    echo "[$(date +%H:%M:%S)] L=$L round[$start/$n] done" | tee -a "$LOG_DIR/../bsa_ruler_summary.txt"
    start=$end
  done
  echo "[$(date +%H:%M:%S)] L=$L ALL DONE" | tee -a "$LOG_DIR/../bsa_ruler_summary.txt"
}

for L in 32768 65536 131072; do
  run_len "$L"
done
if [ -s "$LOG_DIR/failed.txt" ]; then
  echo "ALL DONE bsa_ruler (FAILED see $LOG_DIR/failed.txt)"
  exit 1
fi
echo "ALL DONE bsa_ruler TS=$TS"
