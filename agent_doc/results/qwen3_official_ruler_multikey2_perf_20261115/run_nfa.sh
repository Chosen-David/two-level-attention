#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_nfa.sh GPU}"
ROOT="/home/wangyuanshuo02/ak_verify_results/qwen3_official_ruler_perf_19edfebd_20261115"
REPO="/tmp/qwen3-official-smoke-19edfebd-20261115"
LOG="$ROOT/logs/nfa_graph_gpu${GPU}.log"
for L in 262144 131072 65536 32768; do
  for B in 2 4 1; do
    if [[ "$L" == 262144 && "$B" == 1 ]]; then
      continue
    fi
    if [[ "$L" == 262144 && "$B" == 4 ]]; then
      printf '[SKIP RESOURCE BLOCKED OOM] L=%s B=%s\n' "$L" "$B" >> "$LOG"
      continue
    fi
    OUT="$ROOT/nfa/nfa_graph_b${B}_$((L / 1024))k.json"
    if python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete" and r["source"]["head"]=="19edfebd404cfffadeadc30de7378f9fbe3e0069" and r["context_length"]==int(sys.argv[2]) and r["batch_size"]==int(sys.argv[3]) and r["warmup_requests"]==1 and r["repeats"]==3 and r["max_new_tokens"]==33' "$OUT" "$L" "$B" 2>/dev/null; then
      printf '[SKIP COMPLETE] L=%s B=%s\n' "$L" "$B" >> "$LOG"
      continue
    fi
    printf '[START] L=%s B=%s output=%s\n' "$L" "$B" "$OUT" >> "$LOG"
    env -C "$REPO" CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" \
      python benchmark/efficiency/benchmark_long_context_two_point.py \
      --method nfa --execution graph --task niah_multikey_2 --sample-index 2 \
      --context-length "$L" --batch-size "$B" --warmup-requests 1 --repeats 3 \
      --max-new-tokens 33 --device cuda:0 --output "$OUT" >> "$LOG" 2>&1
    printf '[DONE] L=%s B=%s\n' "$L" "$B" >> "$LOG"
  done
done
printf '[CHAIN DONE]\n' >> "$LOG"
