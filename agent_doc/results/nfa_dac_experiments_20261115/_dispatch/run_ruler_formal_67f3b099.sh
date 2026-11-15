#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <LENGTH:METHOD:SUBSPACE:GATE> [...]}"
shift
WT=/tmp/nfa-dac-quality-20261115
EXPECTED_HEAD=67f3b099073dd2d27fc0176b631eb9b4bd6a8214
ROOT="$HOME/ak_verify_results/nfa_dac_ruler_formal_67f3b099_20261115"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER"
if [[ "$#" -lt 1 ]]; then
  printf 'at least one formal arm is required\n' >&2
  exit 2
fi
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s RULER formal chain already running\n' "$GPU" >&2
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
rm -f "$ROOT/_status/gpu${GPU}.done" "$ROOT/_status/gpu${GPU}.failed"
finish() {
  rc=$?
  if [[ $rc -ne 0 ]]; then
    printf 'host=%s gpu=%s rc=%s time=%s\n' "$(hostname)" "$GPU" "$rc" "$(date -Is)" \
      >"$ROOT/_status/gpu${GPU}.failed"
  fi
}
trap finish EXIT
for SPEC in "$@"; do
  IFS=: read -r LENGTH METHOD SUBSPACE GATE <<<"$SPEC"
  if [[ "$LENGTH" != 32768 && "$LENGTH" != 65536 && "$LENGTH" != 131072 ]]; then
    printf 'unsupported context length %s\n' "$LENGTH" >&2
    exit 2
  fi
  if [[ "$METHOD" == nfa ]]; then
    LABEL="nfa_${SUBSPACE}_gate_${GATE}"
    EXTRA=(--nfa-subspace "$SUBSPACE" --nfa-gate "$GATE")
  elif [[ "$METHOD" == flashinfer ]]; then
    LABEL=flashinfer_fullkv
    EXTRA=()
  else
    printf 'unsupported method %s\n' "$METHOD" >&2
    exit 2
  fi
  LOG="$ROOT/_logs/${LABEL}_L${LENGTH}.log"
  OUTPUT="$ROOT/$LABEL/L${LENGTH}"
  MANIFEST="$OUTPUT/L${LENGTH}/quality_manifest_L${LENGTH}_s0of1.json"
  if [[ -f "$MANIFEST" ]]; then
    printf 'REFUSE existing formal manifest %s; audit before any rerun\n' "$MANIFEST" >&2
    exit 5
  fi
  mkdir -p "$OUTPUT"
  apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
  if [[ -n "$apps" ]]; then
    printf 'GPU%s occupied by compute PID=%s before %s L%s\n' "$GPU" "$apps" "$LABEL" "$LENGTH" >&2
    exit 4
  fi
  printf 'START host=%s gpu=%s arm=%s length=%s rows=100 head=%s shell_pid=%s time=%s\n' \
    "$(hostname)" "$GPU" "$LABEL" "$LENGTH" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
  env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    /home/fluentllmenv/bin/python -m benchmark.efficiency.benchmark_ruler_quality \
      --method "$METHOD" --execution eager "${EXTRA[@]}" \
      --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
      --task niah_multikey_2 --context-length "$LENGTH" \
      --data-root "$DATA" --output-dir "$OUTPUT" \
      --device cuda:0 --visible-len 8192 >>"$LOG" 2>&1
  if [[ ! -f "$MANIFEST" ]]; then
    printf 'formal arm exited without complete manifest %s\n' "$MANIFEST" >&2
    exit 6
  fi
  printf 'DONE host=%s gpu=%s arm=%s length=%s rows=100 time=%s\n' \
    "$(hostname)" "$GPU" "$LABEL" "$LENGTH" "$(date -Is)" >>"$LOG"
done
printf 'host=%s gpu=%s status=complete time=%s\n' "$(hostname)" "$GPU" "$(date -Is)" \
  >"$ROOT/_status/gpu${GPU}.done"
