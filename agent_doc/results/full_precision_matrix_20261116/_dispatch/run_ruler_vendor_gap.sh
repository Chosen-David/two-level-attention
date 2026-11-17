#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <sparq|hisa|clusterkv> <65536|131072>}"
METHOD="${2:?usage: $0 <GPU> <sparq|hisa|clusterkv> <65536|131072>}"
LENGTH="${3:?usage: $0 <GPU> <sparq|hisa|clusterkv> <65536|131072> [smoke|formal]}"
MODE="${4:-formal}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD to the clean producer commit}"
WT="${WT:-/tmp/full-precision-matrix-${EXPECTED_HEAD:0:8}}"
ROOT="${ROOT:-$HOME/ak_verify_results/full_precision_matrix_${EXPECTED_HEAD:0:8}_20261116}"
DATA="${DATA:-$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER}"
MODEL="${MODEL:-$HOME/Qwen3-8B}"
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt)
case "$METHOD" in
  sparq|hisa|clusterkv) ;;
  *) printf 'unknown method %s\n' "$METHOD" >&2; exit 3 ;;
esac
case "$LENGTH" in
  65536) YARN_FACTOR=2.0 ;;
  131072) YARN_FACTOR=4.0 ;;
  *) printf 'length must be 65536 or 131072\n' >&2; exit 3 ;;
esac
case "$MODE" in
  smoke)
    TASKS=(niah_single_1)
    MAX_NUM=1
    CELL_SUFFIX=_smoke
    ;;
  formal)
    MAX_NUM=100
    CELL_SUFFIX=
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
for TASK in "${TASKS[@]}"; do
  [[ -f "$DATA/$LENGTH/$TASK.jsonl" ]] || { printf 'missing input %s\n' "$TASK" >&2; exit 3; }
done
apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [[ -n "$apps" ]]; then
  printf 'GPU%s occupied by compute PID=%s\n' "$GPU" "$apps" >&2
  exit 4
fi
OUTPUT="$ROOT/${METHOD}_L${LENGTH}${CELL_SUFFIX}"
PRED_ROOT="$OUTPUT/L${LENGTH}/pred_full_matrix"
LOG="$ROOT/_logs/${METHOD}_L${LENGTH}${CELL_SUFFIX}.log"
if [[ -e "$PRED_ROOT" ]]; then
  printf 'REFUSE existing prediction generation directory %s\n' "$PRED_ROOT" >&2
  exit 5
fi
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/${METHOD}_L${LENGTH}${CELL_SUFFIX}_gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s method-length cell already running\n' "$GPU" >&2
  exit 3
fi
printf 'START host=%s gpu=%s mode=%s method=%s length=%s rows_per_task=%s head=%s shell_pid=%s time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$METHOD" "$LENGTH" "$MAX_NUM" "$EXPECTED_HEAD" "$$" "$(date -Is)" >"$LOG"
EXTRA_ENV=()
if [[ "$METHOD" == clusterkv ]]; then
  # ClusterKV 使用其锁定依赖环境；不改变其他方法的 Python 身份。
  cd "$WT"
  source "$WT/exp/vendor_baselines/clusterkv_vendor/ckv_env.sh"
  PYTHON_BIN="${PYTHON_BIN:-${CKV_PYTHON:-/home/fluentllmenv/bin/python}}"
else
  PYTHON_BIN="${PYTHON_BIN:-/home/fluentllmenv/bin/python}"
  EXTRA_ENV=(PYTHONPATH="$HOME/local_pylibs")
fi
for TASK in "${TASKS[@]}"; do
  printf 'TASK_START method=%s length=%s task=%s time=%s\n' \
    "$METHOD" "$LENGTH" "$TASK" "$(date -Is)" >>"$LOG"
  case "$METHOD" in
    sparq) METHOD_FLAGS=(--sparq_k 1024) ;;
    hisa) METHOD_FLAGS=(--hisa_budget 1024 --hisa_k_block_size 128 --hisa_block_topk 64) ;;
    clusterkv) METHOD_FLAGS=(--clusterkv_token_budget 1024 --clusterkv_sink 16 --clusterkv_window 320 --clusterkv_nlist 400) ;;
  esac
  (
    cd "$WT"
    env "${EXTRA_ENV[@]}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path "$MODEL" --task "$TASK" \
      --context_length "$LENGTH" --method "${METHOD}_vendor" \
      --t "full-matrix-${EXPECTED_HEAD:0:8}" --pred_postfix _full_matrix \
      --data-root "$DATA" --output-dir "$OUTPUT" --max-num "$MAX_NUM" --yarn \
      "${METHOD_FLAGS[@]}"
  ) >>"$LOG" 2>&1
  printf 'TASK_DONE method=%s length=%s task=%s time=%s\n' \
    "$METHOD" "$LENGTH" "$TASK" "$(date -Is)" >>"$LOG"
done
(
  cd "$WT"
  env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" benchmark/RULER/score_ruler_formal.py \
    --root "$OUTPUT" --pred-postfix _full_matrix --data-root "$DATA" \
    --out "$OUTPUT/formal_score.json" --expect-tasks "${#TASKS[@]}" --min-samples "$MAX_NUM" \
    --model-path "$MODEL" --yarn --yarn-factor "$YARN_FACTOR" --rope-policy legacy
) >>"$LOG" 2>&1
printf 'DONE host=%s gpu=%s mode=%s method=%s length=%s rows=%s time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$METHOD" "$LENGTH" "$((MAX_NUM * ${#TASKS[@]}))" "$(date -Is)" >>"$LOG"
printf 'host=%s gpu=%s mode=%s method=%s length=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$METHOD" "$LENGTH" "$(date -Is)" \
  >"$ROOT/_status/${METHOD}_L${LENGTH}${CELL_SUFFIX}.done"
