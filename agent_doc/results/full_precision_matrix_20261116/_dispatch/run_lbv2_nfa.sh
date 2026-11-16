#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <SHARD> <NUM_SHARDS> <smoke|formal>}"
SHARD="${2:?usage: $0 <GPU> <SHARD> <NUM_SHARDS> <smoke|formal>}"
WORLD="${3:?usage: $0 <GPU> <SHARD> <NUM_SHARDS> <smoke|formal>}"
MODE="${4:?usage: $0 <GPU> <SHARD> <NUM_SHARDS> <smoke|formal>}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD to the clean producer commit}"
WT="${WT:-/tmp/full-precision-matrix-${EXPECTED_HEAD:0:8}}"
PYTHON_BIN="${PYTHON_BIN:-/home/fluentllmenv/bin/python}"
DATA="${DATA:-$HOME/datasets/LongBench/data}"
MODEL="${MODEL:-$HOME/Qwen3-8B}"
BASE_ROOT="${ROOT:-$HOME/ak_verify_results/full_precision_matrix_${EXPECTED_HEAD:0:8}_20261116}"
if (( SHARD < 0 || SHARD >= WORLD || WORLD < 1 )); then
  printf 'require 0<=shard<num-shards\n' >&2
  exit 3
fi
case "$MODE" in
  smoke)
    [[ "$SHARD" == 0 && "$WORLD" == 1 ]] || { printf 'smoke requires shard=0 world=1\n' >&2; exit 3; }
    OUTPUT="$BASE_ROOT/qwen3_8b_nfa_lbv2_smoke"
    SAMPLE_FLAGS=(--sample-indices 0)
    EXPECTED_ROWS=1
    ;;
  formal)
    OUTPUT="$BASE_ROOT/qwen3_8b_nfa_lbv2"
    SAMPLE_FLAGS=()
    EXPECTED_ROWS=$(( (503 - SHARD + WORLD - 1) / WORLD ))
    ;;
  *) printf 'mode must be smoke or formal\n' >&2; exit 3 ;;
esac
if [[ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  printf 'unexpected source HEAD\n' >&2
  exit 3
fi
if [[ -n "$(git -C "$WT" status --short)" ]]; then
  printf 'source worktree is not clean\n' >&2
  exit 3
fi
[[ -f "$MODEL/config.json" ]] || { printf 'missing model config\n' >&2; exit 3; }
[[ -f "$DATA/lbv2.json" || -f "$DATA/data.json" ]] || { printf 'missing LongBench v2 input\n' >&2; exit 3; }
MANIFEST="$OUTPUT/quality_manifest_s${SHARD}of${WORLD}.json"
PREDICTION="$OUTPUT/pred/lbv2-nfa_tail_gate_off_eager_s${SHARD}of${WORLD}.jsonl"
LOG="$BASE_ROOT/_logs/qwen3_8b_nfa_lbv2_${MODE}_s${SHARD}of${WORLD}.log"
if [[ -e "$MANIFEST" ]]; then
  printf 'REFUSE existing formal manifest %s\n' "$MANIFEST" >&2
  exit 5
fi
apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [[ -n "$apps" ]]; then
  printf 'GPU%s occupied by compute PID=%s\n' "$GPU" "$apps" >&2
  exit 4
fi
mkdir -p "$BASE_ROOT/_logs" "$BASE_ROOT/_status"
exec 9>"$BASE_ROOT/_status/qwen3_8b_nfa_lbv2_${MODE}_s${SHARD}of${WORLD}_gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s LBv2 shard already running\n' "$GPU" >&2
  exit 3
fi
printf 'START host=%s gpu=%s mode=%s shard=%s world=%s expected_rows=%s head=%s shell_pid=%s time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$SHARD" "$WORLD" "$EXPECTED_ROWS" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
(
  cd "$WT"
  env PYTHONPATH="$WT:$HOME/local_pylibs" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" -u -m benchmark.efficiency.benchmark_longbench_quality \
    --method nfa --execution eager --nfa-subspace tail --nfa-gate off \
    --model-path "$MODEL" --model-name Qwen3-8B \
    --data-root "$DATA" --output-dir "$OUTPUT" --tasks lbv2 \
    --shard-index "$SHARD" --num-shards "$WORLD" \
    --device cuda:0 --visible-len 8192 "${SAMPLE_FLAGS[@]}"
) >>"$LOG" 2>&1
[[ -f "$MANIFEST" && -f "$PREDICTION" ]] || {
  printf 'generation exited without manifest/prediction\n' >&2
  exit 6
}
SCORE_DIR="$OUTPUT/_score_s${SHARD}of${WORLD}"
mkdir -p "$SCORE_DIR"
PRED_NAME="$(basename "$PREDICTION")"
ln -sfn "$PREDICTION" "$SCORE_DIR/$PRED_NAME"
ln -sfn "$PREDICTION.runcap.json" "$SCORE_DIR/$PRED_NAME.runcap.json"
ln -sfn "$PREDICTION.source_manifest.json" "$SCORE_DIR/$PRED_NAME.source_manifest.json"
(
  cd "$WT"
  env PYTHONPATH="$WT:$HOME/local_pylibs" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m benchmark.LongBench.eval \
    --output-path "$SCORE_DIR" --expect-count "$EXPECTED_ROWS" \
    --manifest "$SCORE_DIR/$PRED_NAME.source_manifest.json"
) >>"$LOG" 2>&1
printf 'DONE host=%s gpu=%s mode=%s shard=%s world=%s rows=%s time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$SHARD" "$WORLD" "$EXPECTED_ROWS" "$(date -Is)" >>"$LOG"
printf 'host=%s gpu=%s mode=%s shard=%s world=%s rows=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$SHARD" "$WORLD" "$EXPECTED_ROWS" "$(date -Is)" \
  >"$BASE_ROOT/_status/qwen3_8b_nfa_lbv2_${MODE}_s${SHARD}of${WORLD}.done"
