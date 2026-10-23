#!/bin/bash
# E160R driver：Qwen3-32B decode e2e 延迟 bench 复刻臂（pc31 GPU0）
# 原判：pc32 ~/e160（E160 判决 JSON e160_32b_e2e_latency_verdict.json）
# 协议与 pc32 /tmp/e160_driver.sh 完全一致（32K/64K 两档 ABBA + eager 门禁；
# 128K 段按协议剔除——原判单卡不可容纳跳过，复刻臂不跑）
# 纪律：只碰 GPU0；每次发射前等卡空（<2000MiB，poll 15s 抢窗）；×3 attempts；
#       FAIL 归档日志与残骸 JSON；历史产物不覆写（.done 幂等跳过）。
PY=/home/fluentllmenv/bin/python3
REPO=$HOME/two-level-attention
BENCH=$REPO/exp/bench
OUT=$HOME/e160r
MODEL=$HOME/models/huggingface.co/Qwen/Qwen3-32B
export TLI_SHADOW_PKG=$HOME/tli_ako/solution
export TLI_SPARSE_PREFILL=0
export CUDA_VISIBLE_DEVICES=0
CHAMP="--method tli --tli_far_method minmax --tli_near_method avg --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 --tli_enable_kmeans false --tli_enable_layer_skip false --tli_subspace tail --tia_level2_cmp_ratio 4 --tli_sink_tokens 4 --tli_swa_size 128 --tia_level1_topk 128 --tia_level2_topk 1024"
mkdir -p "$OUT/logs"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

wait_gpu_free() {
  # 抢窗门：poll 15s 直到 GPU0 used < 2000 MiB
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$used" -lt 2000 ]; then return 0; fi
    sleep 15
  done
}

run_task() {
  # $1=name $2=out_json；$3.. = python 命令
  local name=$1 out=$2; shift 2
  if [ -f "$OUT/$name.done" ]; then log "SKIP $name (done)"; return 0; fi
  for att in 1 2 3; do
    wait_gpu_free
    # 归档上次失败残骸（保历史）
    if [ -f "$out" ]; then mv "$out" "$out.fail-$(date +%s)"; fi
    log "RUN $name att=$att start"
    # 上面已 shift 2，此时 "$@" 即 python 命令（勿再用 ${@:3}）
    "$PY" "$@" > "$OUT/logs/$name.att$att.log" 2>&1 &
    local pid=$!
    # GPU0 显存峰值轮询（10s 间隔，随进程退出停）
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
  # bit-exact 门禁：graph 臂（a1+a2）vs eager 对拍臂 gen_tokens 逐位比较
  # $1=L
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

# ============ 32K 档 ============
L=32768
run_task "tli_graph_L${L}_a1" "$OUT/tli_graph_L${L}_a1.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 2 --warmup 1 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a1.json" --label 32b-A1 \
  $CHAMP --model-path "$MODEL" || { echo A1_FAILED > "$OUT/ABORT"; exit 1; }
run_task "tli_eager_L${L}" "$OUT/tli_eager_L${L}.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 1 --max-gen 128 \
  --graph-mode eager --out "$OUT/tli_eager_L${L}.json" --label 32b-eager-gate \
  $CHAMP --model-path "$MODEL" || { echo EAGER_GATE_FAILED > "$OUT/ABORT"; exit 1; }
# A1 vs eager 门禁先验（不等 B 采完才发现问题）
gen_gate $L
if [ $? -eq 42 ]; then echo "GATE_FAIL_L${L}" > "$OUT/ABORT"; exit 42; fi
run_task "dense_L${L}_b1" "$OUT/dense_L${L}_b1.json" \
  "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
  --method none --out "$OUT/dense_L${L}_b1.json" --label 32b-B1 --model-path "$MODEL"
run_task "dense_L${L}_b2" "$OUT/dense_L${L}_b2.json" \
  "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
  --method none --out "$OUT/dense_L${L}_b2.json" --label 32b-B2 --model-path "$MODEL"
run_task "tli_graph_L${L}_a2" "$OUT/tli_graph_L${L}_a2.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 2 --warmup 1 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a2.json" --label 32b-A2 \
  $CHAMP --model-path "$MODEL"
log "===== 32K 档完成 ====="

# ============ 64K 档 ============
L=65536
run_task "tli_graph_L${L}_a1" "$OUT/tli_graph_L${L}_a1.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 2 --warmup 1 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a1.json" --label 32b-A1 \
  $CHAMP --model-path "$MODEL" || { echo A1_64K_FAILED > "$OUT/ABORT"; exit 1; }
run_task "tli_eager_L${L}" "$OUT/tli_eager_L${L}.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 1 --warmup 1 --max-gen 128 \
  --graph-mode eager --out "$OUT/tli_eager_L${L}.json" --label 32b-eager-gate \
  $CHAMP --model-path "$MODEL" || { echo EAGER_GATE_64K_FAILED > "$OUT/ABORT"; exit 1; }
gen_gate $L
if [ $? -eq 42 ]; then echo "GATE_FAIL_L${L}" > "$OUT/ABORT"; exit 42; fi
run_task "dense_L${L}_b1" "$OUT/dense_L${L}_b1.json" \
  "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
  --method none --out "$OUT/dense_L${L}_b1.json" --label 32b-B1 --model-path "$MODEL"
run_task "dense_L${L}_b2" "$OUT/dense_L${L}_b2.json" \
  "$BENCH/e152_e2e_latency.py" -L $L --repeats 3 --warmup 1 --max-gen 128 \
  --method none --out "$OUT/dense_L${L}_b2.json" --label 32b-B2 --model-path "$MODEL"
run_task "tli_graph_L${L}_a2" "$OUT/tli_graph_L${L}_a2.json" \
  "$BENCH/e152v4_graph_latency.py" -L $L --repeats 2 --warmup 1 --max-gen 128 \
  --graph-mode graph --out "$OUT/tli_graph_L${L}_a2.json" --label 32b-A2 \
  $CHAMP --model-path "$MODEL"
log "===== 64K 档完成 ====="

log "ALL-PROTOCOL-END"
