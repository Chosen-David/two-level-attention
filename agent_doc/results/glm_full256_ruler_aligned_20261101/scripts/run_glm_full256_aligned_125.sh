#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/hadoop-aipnlp/tli-glm-eval
OUT=/tmp/tli-glm-full256-aligned-full-run
CODE=/tmp/tli-glm-origin-main
MODEL="$ROOT/models/GLM-4.7-Flash"
DATA="$ROOT/data/LongBench"
RULER="$ROOT/data/RULER/65536/niah_multikey_2.jsonl"
MANIFEST="$ROOT/model_files.json"
SCRIPT="$CODE/glm_full256_full_aligned.py"
GPUS=(0 1 2 3 4 5 7)
WORKERS=${#GPUS[@]}
SOURCE_COMMIT=42821f81a4739d05d8bc81e1d4cc58307e34bb7e
mkdir -p "$OUT/logs" /tmp/tli-triton-cache

if [ ! -f "$SCRIPT" ]; then
  echo "runner script missing; abort" | tee -a "$OUT/launcher.log"; exit 2
fi
if [ ! -f "$MANIFEST" ]; then
  echo "model manifest missing; abort" | tee -a "$OUT/launcher.log"; exit 2
fi
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${used:-999999}" -ge 1000 ]; then
    echo "GPU$g busy $used MiB; abort" | tee -a "$OUT/launcher.log"; exit 2
  fi
done

python3 - "$OUT/status.json" "$SCRIPT" "$SOURCE_COMMIT" <<'PY'
import json,sys,time,hashlib
status_path,script,commit=sys.argv[1],sys.argv[2],sys.argv[3]
d={
 'phase':'launching','time':time.time(),'host':'33.253.189.125',
 'source_commit':commit,
 'script_sha256':hashlib.sha256(open(script,'rb').read()).hexdigest(),
 'scope':'Aligned latest-origin-main full RULER niah_multikey_2 (100) + LongBench AVG5 (1300), dense vs full256, 7 GPUs excluding reserved GPU6'
}
json.dump(d,open(status_path,'w'),indent=2)
PY

echo "[$(date -u +%H:%M:%SZ)] launching $WORKERS workers" | tee -a "$OUT/launcher.log"
pids=()
for i in "${!GPUS[@]}"; do
  rank="$i"
  gpu="${GPUS[$i]}"
  log="$OUT/logs/rank${rank}.log"
  echo "[$(date -u +%H:%M:%SZ)] launch rank=$rank gpu=$gpu" | tee -a "$OUT/launcher.log"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$CODE" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/tli-triton-cache \
    nohup "$ROOT/venv/bin/python" "$SCRIPT" \
      --model "$MODEL" --data "$DATA" --ruler "$RULER" --out "$OUT" \
      --model-manifest "$MANIFEST" --rank "$rank" --workers "$WORKERS" \
      >"$log" 2>&1 &
  echo $! > "$OUT/logs/rank${rank}.pid"
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    echo "[$(date -u +%H:%M:%SZ)] rank=$i pid=$pid exited nonzero" | tee -a "$OUT/launcher.log"
    failed=1
  else
    echo "[$(date -u +%H:%M:%SZ)] rank=$i pid=$pid exited zero" | tee -a "$OUT/launcher.log"
  fi
done

python3 - "$OUT/status.json" <<'PY'
import json,sys,time
p=sys.argv[1]
d=json.load(open(p))
d.update(phase='workers_exited_await_independent_review',time=time.time())
json.dump(d,open(p,'w'),indent=2)
PY
echo "[$(date -u +%H:%M:%SZ)] all workers exited" | tee -a "$OUT/launcher.log"
exit "$failed"
