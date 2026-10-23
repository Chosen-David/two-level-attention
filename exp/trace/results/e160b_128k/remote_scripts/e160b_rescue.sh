#!/bin/bash
# E160b rescue driver：128K 档救援协议（expandable_segments:True 全臂统一）
# 背景：E160b 原协议（A 臂 warmup1+repeats2）在 run=1 第二次 prefill OOM——
#   单 run 峰值 142216/143771 MiB 可容纳，但同进程多 prefill 不可容纳
#   （E160 3 attempts + E160b att1 共 4 次确定性同形态 OOM，att2/3 短路跳过）。
# 救援协议（与 E160 主表的唯一差异，判决中如实披露）：
#   A 臂 = warmup 0 + repeats 1 × 4 个独立进程（a1..a4），进程级 repeat
#   替代进程内 repeat，TPOT 口径不变（measured run 内 capture 发生在 fill
#   阶段、不进 TPOT；decode 走 replay；E160 A 臂 measured runs 同为纯
#   replay）；样本数 2×2 → 4×1 相同。
#   eager 对拍门禁臂 = warmup 0 + repeats 1（同上原因）。
#   dense B1/B2 = E160 原协议不动（warmup 1 + repeats 3；dense 无 static
#   cache，内存余量大）。
# 排程（ABBA 精神）：a1 → eager-gate → gen_gate → b1 → b2 → a2 → a3 → a4 → 终门禁
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
  # bit-exact 门禁：graph 臂（a1..a4 现有者）vs eager 对拍臂 gen_tokens 逐位比较
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
checked = 0
for arm in ("a1", "a2", "a3", "a4"):
    p = os.path.join(out, f"tli_graph_L{L}_{arm}.json")
    if not os.path.exists(p):
        continue
    for gi, g in enumerate(load_gens(p)):
        checked += 1
        if any(g != e for e in ref):
            bad.append((arm, gi))
        if len(g) != 128:
            bad.append((arm, gi, f"len={len(g)}"))
if bad:
    print(f"GATE-FAIL L={L} checked={checked} mismatches={bad[:5]}")
    sys.exit(42)
print(f"GATE-PASS L={L} checked={checked} gen_tokens bit-exact vs eager (128 tok)")
PYEOF
}

L=131072

# ---- A1（救援臂：单 run） ----
run_task "tli_graph_L${L}_a1" "$OUT/tli_graph_L${L}_a1.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 0 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a1.json" --label 32b-A1r-exp \
  $CHAMP --model-path "$MODEL" || { echo A1_RESCUE_FAILED > "$OUT/ABORT"; log "A1 救援臂不可容纳——128K TLI 真不可容纳"; exit 1; }

# ---- eager 对拍门禁臂（救援：单 run） ----
run_task "tli_eager_L${L}" "$OUT/tli_eager_L${L}.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 0 --max-gen 128 \
  --graph-mode eager --out "$OUT/tli_eager_L${L}.json" --label 32b-eager-gate-exp \
  $CHAMP --model-path "$MODEL"
if [ ! -f "$OUT/tli_eager_L${L}.done" ]; then
  log "128K eager 对拍臂 OOM——bit-exact 门禁不可执行，如实记录"
  echo "EAGER_GATE_OOM" > "$OUT/EAGER_GATE_STATUS"
else
  echo "EAGER_GATE_OK" > "$OUT/EAGER_GATE_STATUS"
  gen_gate $L
  if [ $? -eq 42 ]; then echo "GATE_FAIL_L${L}" > "$OUT/ABORT"; log "GATE-FAIL——中止协议"; exit 42; fi
fi

# ---- B1 / B2（E160 原协议） ----
run_task "dense_L${L}_b1" "$OUT/dense_L${L}_b1.json" \
  "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
  --method none --out "$OUT/dense_L${L}_b1.json" --label 32b-B1-exp --model-path "$MODEL"
run_task "dense_L${L}_b2" "$OUT/dense_L${L}_b2.json" \
  "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
  --method none --out "$OUT/dense_L${L}_b2.json" --label 32b-B2-exp --model-path "$MODEL"

# ---- A2 / A3 / A4（救援臂：单 run） ----
run_task "tli_graph_L${L}_a2" "$OUT/tli_graph_L${L}_a2.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 0 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a2.json" --label 32b-A2r-exp \
  $CHAMP --model-path "$MODEL"
run_task "tli_graph_L${L}_a3" "$OUT/tli_graph_L${L}_a3.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 0 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a3.json" --label 32b-A3r-exp \
  $CHAMP --model-path "$MODEL"
run_task "tli_graph_L${L}_a4" "$OUT/tli_graph_L${L}_a4.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 0 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a4.json" --label 32b-A4r-exp \
  $CHAMP --model-path "$MODEL"

# ---- 终门禁（全 a1..a4） ----
if [ -f "$OUT/EAGER_GATE_OK" ] || [ "$(cat "$OUT/EAGER_GATE_STATUS" 2>/dev/null)" = "EAGER_GATE_OK" ]; then
  gen_gate $L
  if [ $? -eq 42 ]; then echo "GATE_FAIL_L${L}_FINAL" > "$OUT/ABORT"; exit 42; fi
fi

log "===== E160b rescue 协议结束 ====="
log "ALL-PROTOCOL-END"
