#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_quest.sh GPU}"
ROOT="/home/wangyuanshuo02/ak_verify_results/qwen3_official_ruler_perf_19edfebd_20261115"
REPO="/tmp/qwen3-official-smoke-19edfebd-20261115"
LOG="$ROOT/logs/quest_gpu${GPU}.log"
for L in 131072 65536 32768; do
  OUT="$ROOT/quest/quest_b1_$((L / 1024))k.json"
  if python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete" and r["source"]["head"]=="19edfebd404cfffadeadc30de7378f9fbe3e0069" and r["context_length"]==int(sys.argv[2]) and r["batch_size"]==1 and r["warmup_requests"]==1 and r["repeats"]==3 and r["max_new_tokens"]==33 and r["runtime"]["dense_or_torch_fallback"] is False' "$OUT" "$L" 2>/dev/null; then
    printf '[SKIP COMPLETE] L=%s B=1\n' "$L" >> "$LOG"
    continue
  fi
  printf '[START] L=%s B=1 output=%s\n' "$L" "$OUT" >> "$LOG"
  env -C "$REPO" CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" \
    python benchmark/official_baselines/long_context_official.py \
    --method quest --task niah_multikey_2 --sample-index 2 \
    --context-length "$L" --batch-size 1 --warmup-requests 1 --repeats 3 \
    --max-new-tokens 33 --budget 1024 --quest-page-size 16 \
    --device cuda:0 --output "$OUT" >> "$LOG" 2>&1
  printf '[DONE] L=%s B=1\n' "$L" >> "$LOG"
done
printf '[CHAIN DONE]\n' >> "$LOG"
