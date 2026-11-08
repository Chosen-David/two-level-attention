#!/usr/bin/env bash
set -uo pipefail
# Generic GLM subspace scan launcher.
# Required env: PYTHON, PYTHONPATH, CODE, MODEL, DATA, RULER, MANIFEST,
#               OUT, DIMS, SCHEMA, LABEL, GPUS (comma list),
#               RULER_LIMIT, PER_TASK, GATE_MODE, EPSILON, STATISTIC
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS:-}"
WORKERS=${#GPU_ARRAY[@]}
[ "$WORKERS" -gt 0 ] || { echo "GPUS empty"; exit 2; }
mkdir -p "$OUT/logs" /tmp/tli-triton-cache
SCRIPT="$CODE/glm_subspace_runner.py"

if [ ! -f "$SCRIPT" ]; then echo "runner missing: $SCRIPT"; exit 2; fi
if [ ! -f "$MANIFEST" ]; then echo "manifest missing: $MANIFEST"; exit 2; fi
for g in "${GPU_ARRAY[@]}"; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ "${used:-999999}" -ge 1000 ]; then echo "GPU$g busy $used MiB"; exit 2; fi
done

python3 - "$OUT/status.json" "$SCRIPT" <<'PY'
import json,sys,time,hashlib
out,script=sys.argv[1],sys.argv[2]
json.dump({'phase':'launching','time':time.time(),'host':__import__('socket').gethostname(),
           'script_sha256':hashlib.sha256(open(script,'rb').read()).hexdigest(),
           'scope':'GLM subspace screen','dims':__import__('os').environ.get('DIMS',''),
           'schema':__import__('os').environ.get('SCHEMA','')}, open(out+'/status.json','w'), indent=2)
PY

pids=()
for i in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$i]}"
  log="$OUT/logs/rank${i}.log"
  echo "[$(date -u +%H:%M:%SZ)] launch rank=$i gpu=$gpu" | tee -a "$OUT/launcher.log"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/tli-triton-cache \
    nohup "$PYTHON" "$SCRIPT" \
      --model "$MODEL" --data "$DATA" --ruler "$RULER" --out "$OUT" \
      --model-manifest "$MANIFEST" --rank "$i" --workers "$WORKERS" \
      --dims "$DIMS" --schema "$SCHEMA" --selector-label "$LABEL" \
      --ruler-limit "$RULER_LIMIT" --per-task "$PER_TASK" \
      --gate-mode "$GATE_MODE" --epsilon "$EPSILON" --statistic "$STATISTIC" \
      >"$log" 2>&1 &
  echo $! > "$OUT/logs/rank${i}.pid"
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    echo "rank=$i pid=$pid nonzero" | tee -a "$OUT/launcher.log"; failed=1
  else
    echo "rank=$i pid=$pid zero" | tee -a "$OUT/launcher.log"
  fi
done
python3 - "$OUT/status.json" <<'PY'
import json,sys,time
p=sys.argv[1]
d=json.load(open(p))
d.update(phase='workers_exited_await_independent_review',time=time.time())
json.dump(d,open(p,'w'),indent=2)
PY
exit "$failed"
