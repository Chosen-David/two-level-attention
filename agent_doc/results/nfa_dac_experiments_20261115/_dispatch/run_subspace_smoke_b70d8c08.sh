#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU>}"
WT=/tmp/nfa-dac-quality-smoke-b70d8c08-20261115
EXPECTED_HEAD=b70d8c08dcac64b5bfb4b86c78715a6956640e65
ROOT="$HOME/ak_verify_results/nfa_dac_subspace_smoke_b70d8c08_20261115"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER"
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
mkdir -p "$ROOT/_logs"
for DIM in 16 64; do
  LABEL="tail${DIM}_gate_off"
  OUTPUT="$ROOT/$LABEL/L32768"
  MANIFEST="$OUTPUT/quality_manifest_L32768_s0of1.json"
  LOG="$ROOT/_logs/${LABEL}_L32768.log"
  if [[ -f "$MANIFEST" ]]; then
    printf 'REFUSE existing smoke manifest %s\n' "$MANIFEST" >&2
    exit 5
  fi
  mkdir -p "$OUTPUT"
  printf 'START host=%s gpu=%s arm=%s length=32768 sample=0 head=%s shell_pid=%s time=%s\n' \
    "$(hostname)" "$GPU" "$LABEL" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
  env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    /home/fluentllmenv/bin/python -m benchmark.efficiency.benchmark_ruler_quality \
      --method nfa --execution eager --nfa-subspace tail --nfa-subspace-dim "$DIM" \
      --nfa-gate off --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
      --task niah_multikey_2 --context-length 32768 --data-root "$DATA" \
      --output-dir "$OUTPUT" --device cuda:0 --visible-len 8192 \
      --sample-indices 0 >>"$LOG" 2>&1
  if [[ ! -f "$MANIFEST" ]]; then
    printf 'smoke arm exited without manifest %s\n' "$MANIFEST" >&2
    exit 6
  fi
  printf 'DONE host=%s gpu=%s arm=%s sample=0 time=%s\n' \
    "$(hostname)" "$GPU" "$LABEL" "$(date -Is)" >>"$LOG"
done
