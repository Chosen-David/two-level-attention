#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <32768|65536|131072> [smoke|formal]}"
LENGTH="${2:?usage: $0 <GPU> <32768|65536|131072> [smoke|formal]}"
MODE="${3:-formal}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD to the clean producer commit}"
EXPECTED_QUEST_HEAD="${EXPECTED_QUEST_HEAD:-01c1623bf9395009520874e989e29f683203b357}"
WT="${WT:-/tmp/full-precision-active-${EXPECTED_HEAD:0:8}}"
ROOT="${ROOT:-$HOME/ak_verify_results/full_precision_active_${EXPECTED_HEAD:0:8}_20261117}"
DATA="${DATA:-$HOME/sparse-bench/third_party/KVCache-Factory/data/RULER}"
MODEL="${MODEL:-$HOME/Qwen3-8B}"
QUEST_RUNTIME_ARCHIVE="${QUEST_RUNTIME_ARCHIVE:-/tmp/quest_runtime_52abaae8.tar.gz}"
QUEST_RUNTIME_SHA256="${QUEST_RUNTIME_SHA256:-7ee987d9fa5e168ea81feed6c298c7937f4c9f427e03a32a0e044cf8d5abf67a}"
QUEST_KERNEL_SHA256="${QUEST_KERNEL_SHA256:-52abaae86e945d0aff1620f6890eba85fe17164a16b6939a414b1a69ec241de4}"
PYTHON_BIN="${PYTHON_BIN:-/home/fluentllmenv/bin/python}"
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt)
case "$LENGTH" in
  32768) YARN_ARGS=() ;;
  65536) YARN_ARGS=(--yarn --yarn-factor 2.0) ;;
  131072) YARN_ARGS=(--yarn --yarn-factor 4.0) ;;
  *) printf 'length must be 32768, 65536 or 131072\n' >&2; exit 3 ;;
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
if [[ "$(git -C "$WT/third-party/Quest" rev-parse HEAD)" != "$EXPECTED_QUEST_HEAD" ]]; then
  printf 'unexpected Quest submodule HEAD\n' >&2
  exit 3
fi
KERNEL="$WT/third-party/Quest/quest/_kernels.cpython-311-x86_64-linux-gnu.so"
[[ -f "$KERNEL" ]] || { printf 'missing Quest kernel\n' >&2; exit 3; }
[[ "$(sha256sum "$KERNEL" | cut -d' ' -f1)" == "$QUEST_KERNEL_SHA256" ]] || {
  printf 'unexpected Quest kernel SHA256\n' >&2
  exit 3
}
[[ "$(sha256sum "$QUEST_RUNTIME_ARCHIVE" | cut -d' ' -f1)" == "$QUEST_RUNTIME_SHA256" ]] || {
  printf 'unexpected Quest runtime archive SHA256\n' >&2
  exit 3
}
[[ -f "$MODEL/config.json" ]] || { printf 'missing model config\n' >&2; exit 3; }
for TASK in "${TASKS[@]}"; do
  [[ -f "$DATA/$LENGTH/$TASK.jsonl" ]] || { printf 'missing input %s\n' "$TASK" >&2; exit 3; }
done
apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [[ -n "$apps" ]]; then
  printf 'GPU%s occupied by compute PID=%s\n' "$GPU" "$apps" >&2
  exit 4
fi
OUTPUT="$ROOT/quest_L${LENGTH}${CELL_SUFFIX}"
PRED_ROOT="$OUTPUT/L${LENGTH}/pred_active"
LOG="$ROOT/_logs/quest_L${LENGTH}${CELL_SUFFIX}.log"
if [[ -e "$PRED_ROOT" ]]; then
  printf 'REFUSE existing prediction generation directory %s\n' "$PRED_ROOT" >&2
  exit 5
fi
mkdir -p "$ROOT/_logs" "$ROOT/_status"
exec 9>"$ROOT/_status/quest_L${LENGTH}${CELL_SUFFIX}_gpu${GPU}.lock"
if ! flock -n 9; then
  printf 'GPU%s Quest method-length cell already running\n' "$GPU" >&2
  exit 3
fi
printf 'START host=%s gpu=%s mode=%s method=quest_official length=%s rows_per_task=%s head=%s quest_head=%s quest_kernel_sha256=%s quest_runtime_sha256=%s shell_pid=%s time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$LENGTH" "$MAX_NUM" "$EXPECTED_HEAD" "$EXPECTED_QUEST_HEAD" \
  "$QUEST_KERNEL_SHA256" "$QUEST_RUNTIME_SHA256" "$$" "$(date -Is)" >"$LOG"
for TASK in "${TASKS[@]}"; do
  printf 'TASK_START method=quest_official length=%s task=%s time=%s\n' \
    "$LENGTH" "$TASK" "$(date -Is)" >>"$LOG"
  (
    cd "$WT"
    env PYTHONPATH="$WT/third-party/Quest:$WT${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m benchmark.official_baselines.ruler_official \
      --method quest --model Qwen3-8B --model-path "$MODEL" --task "$TASK" \
      --context-length "$LENGTH" --t "full-active-${EXPECTED_HEAD:0:8}" \
      --pred-postfix _active --data-root "$DATA" --output-dir "$OUTPUT" \
      --budget 1024 --quest-page-size 16 --max-num "$MAX_NUM" "${YARN_ARGS[@]}"
  ) >>"$LOG" 2>&1
  printf 'TASK_DONE method=quest_official length=%s task=%s time=%s\n' \
    "$LENGTH" "$TASK" "$(date -Is)" >>"$LOG"
done
(
  cd "$WT"
  env PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" benchmark/RULER/score_ruler_formal.py \
    --root "$OUTPUT" --pred-postfix _active --data-root "$DATA" \
    --out "$OUTPUT/formal_score.json" --expect-tasks "${#TASKS[@]}" --min-samples "$MAX_NUM" \
    --model-path "$MODEL" --rope-policy legacy "${YARN_ARGS[@]}"
) >>"$LOG" 2>&1
printf 'DONE host=%s gpu=%s mode=%s method=quest_official length=%s rows=%s time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$LENGTH" "$((MAX_NUM * ${#TASKS[@]}))" "$(date -Is)" >>"$LOG"
printf 'host=%s gpu=%s mode=%s method=quest_official length=%s status=complete time=%s\n' \
  "$(hostname)" "$GPU" "$MODE" "$LENGTH" "$(date -Is)" \
  >"$ROOT/_status/quest_L${LENGTH}${CELL_SUFFIX}.done"
