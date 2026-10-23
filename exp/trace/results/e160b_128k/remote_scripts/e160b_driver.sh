#!/bin/bash
# E160b driver：E160 128K 档重试 —— PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 消碎片
# 协议照抄 /tmp/e160_driver.sh 的 128K 段；全臂统一 expandable_segments 保证两臂公平。
# 成功路径：A1 -> eager gate -> gen_gate -> B1 -> B2 -> A2
# 失败路径（A1 3 attempts OOM）：只跑 dense B1 作对照证据。
PY=/home/fluentllmenv/bin/python3
REPO=$HOME/two-level-attention
BENCH=$REPO/exp/bench
OUT=$HOME/e160b
MODEL=$HOME/models/huggingface.co/Qwen/Qwen3-32B
export TLI_SHADOW_PKG=$HOME/tli_ako/solution
export TLI_SPARSE_PREFILL=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHAMP="--method tli --tli_far_method minmax --tli_near_method avg --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 --tli_enable_kmeans false --tli_enable_layer_skip false --tli_subspace tail --tia_level2_cmp_ratio 4 --tli_sink_tokens 4 --tli_swa_size 128 --tia_level1_topk 128 --tia_level2_topk 1024"
mkdir -p "$OUT/logs"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

wait_gpu_free() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$used" -lt 2000 ]; then return 0; fi
    sleep 15
  done
}

run_task() {
  local name=$1 out=$2; shift 2
  if [ -f "$OUT/$name.done" ]; then log "SKIP $name (done)"; return 0; fi
  for att in 1 2 3; do
    wait_gpu_free
    if [ -f "$out" ]; then mv "$out" "$out.fail-$(date +%s)"; fi
    log "RUN $name att=$att start"
    "$PY" "$@" > "$OUT/logs/$name.att$att.log" 2>&1 &
    local pid=$!
    ( while kill -0 "$pid" 2>/dev/null; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0
        sleep 10
      done ) > "$OUT/logs/$name.att$att.gpu" 2>&1 &
    local mon=$!
    wait "$pid"; local rc=$?
    kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
    if [ "$rc" -eq 0 ]; then
      touch "$OUT/$name.done"
      log "DONE $name att=$att gpu_peak=$(sort -n "$OUT/logs/$name.att$att.gpu" 2>/dev/null | tail -1)"
      return 0
    fi
    log "FAIL $name att=$att rc=$rc tail5:"
    tail -5 "$OUT/logs/$name.att$att.log" | sed 's/^/  /'
    if [ -f "$out" ]; then mv "$out" "$out.fail-$(date +%s)"; fi
    sleep 30
  done
  log "GIVEUP $name"
  return 1
}

gen_gate() {
  local L=$1
  "$PY" - "$OUT" "$L" <<'PYEOF'
import json, sys, os
out, L = sys.argv[1], sys.argv[2]
def load_gens(path):
    recs = json.load(open(path))
    gens = []
    for rec in recs:
        for r in rec["runs"]:
            gens.append(r["gen_tokens"])
    return gens
ref = load_gens(os.path.join(out, f"tli_eager_L{L}.json"))
bad = []
for arm in ("a1", "a2"):
    p = os.path.join(out, f"tli_graph_L{L}_{arm}.json")
    if not os.path.exists(p):
        continue
    for gi, g in enumerate(load_gens(p)):
        if any(g != e for e in ref):
            bad.append((arm, gi))
        if len(g) != 128:
            bad.append((arm, gi, f"len={len(g)}"))
if bad:
    print(f"GATE-FAIL L={L} mismatches={bad[:5]}")
    sys.exit(42)
print(f"GATE-PASS L={L} all gen_tokens bit-exact vs eager (128 tok)")
PYEOF
}

L=131072
run_task "tli_graph_L${L}_a1" "$OUT/tli_graph_L${L}_a1.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 2 --warmup 1 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a1.json" --label 32b-A1-exp \
  $CHAMP --model-path "$MODEL"
if [ ! -f "$OUT/tli_graph_L${L}_a1.done" ]; then
  log "128K A1 expandable 仍不可容纳（3 attempts OOM）——失败路径：跑 dense B1 对照"
  echo "L128K_A1_STILL_OOM" > "$OUT/L128K_STATUS"
  run_task "dense_L${L}_b1" "$OUT/dense_L${L}_b1.json" \
    "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
    --method none --out "$OUT/dense_L${L}_b1.json" --label 32b-B1-exp --model-path "$MODEL"
  log "===== E160b 失败路径结束 ====="
else
  echo "L128K_A1_OK" > "$OUT/L128K_STATUS"
  run_task "tli_eager_L${L}" "$OUT/tli_eager_L${L}.json" \
    "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 1 --max-gen 128 \
    --graph-mode eager --out "$OUT/tli_eager_L${L}.json" --label 32b-eager-gate-exp \
    $CHAMP --model-path "$MODEL"
  if [ ! -f "$OUT/tli_eager_L${L}.done" ]; then
    log "128K eager 对拍臂 OOM——bit-exact 门禁不可执行，如实记录（graph 臂 a1/a2 跨臂对拍作为替代一致性检查）"
    echo "EAGER_GATE_OOM" > "$OUT/EAGER_GATE_STATUS"
  else
    echo "EAGER_GATE_OK" > "$OUT/EAGER_GATE_STATUS"
    gen_gate $L
    if [ $? -eq 42 ]; then echo "GATE_FAIL_L${L}" > "$OUT/ABORT"; log "GATE-FAIL——中止协议"; exit 42; fi
  fi
  run_task "dense_L${L}_b1" "$OUT/dense_L${L}_b1.json" \
    "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
    --method none --out "$OUT/dense_L${L}_b1.json" --label 32b-B1-exp --model-path "$MODEL"
  run_task "dense_L${L}_b2" "$OUT/dense_L${L}_b2.json" \
    "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
    --method none --out "$OUT/dense_L${L}_b2.json" --label 32b-B2-exp --model-path "$MODEL"
  run_task "tli_graph_L${L}_a2" "$OUT/tli_graph_L${L}_a2.json" \
    "$BENCH/e152v4_graph_latency.py" -L $L --repeats 2 --warmup 1 --max-gen 128 \
    --graph-mode graph --out "$OUT/tli_graph_L${L}_a2.json" --label 32b-A2-exp \
    $CHAMP --model-path "$MODEL"
  log "===== 128K 档完成 ====="
fi

log "ALL-PROTOCOL-END"
