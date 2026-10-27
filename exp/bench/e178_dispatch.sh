#!/bin/bash
# E178 parallel dispatch — up to 8× H20, 24 arms
# 2 methods × 3 bs × 4 ctx = 24 configs
# Layout:
#   pc33 GPU0: fa2 × 32K/64K/128K/256K bs=1  (4 arms)
#   pc33 GPU1: tli  × 32K/64K/128K/256K bs=1  (4 arms)
#   pc40 GPU0: fa2 × 32K/64K/128K/256K bs=4  (4 arms)
#   pc40 GPU1: tli  × 32K/64K/128K/256K bs=4  (4 arms)
#   pc02 GPU1: fa2 × 32K/64K/128K/256K bs=6  (4 arms)
#   pc02 GPU2: tli  × 32K/64K/128K/256K bs=6  (4 arms)
#   pc02 GPU3: fa2 × 32K/64K/128K/256K bs=1  (rep)(4 arms)
#   pc31 GPU1: tli  × 32K/64K/128K/256K bs=1  (rep)(4 arms)

export SSHPASS='zHWXp=eK5JvX~I|v'
REPO=/home/wangyuanshuo02/two-level-attention
mkdir -p $REPO/exp/trace/results/e178_attn
LOGDIR=$REPO/exp/trace/results/e178_attn/logs
mkdir -p $LOGDIR

# Cross-machine sync: rsync bench script first
for HOST in 10.238.122.187 10.238.139.251 10.245.58.57; do
  rsync -az -e "sshpass -e ssh -o StrictHostKeyChecking=no -p 8416" \
    $REPO/exp/bench/e178_fa2_attn_microbench.py wangyuanshuo02@$HOST:$REPO/exp/bench/ &
done
wait
echo "=== Bench script synced ==="

run_local() {
  local GPU=$1 METHOD=$2 CTX=$3 BS=$4 TAG=$5
  local OUT=$REPO/exp/trace/results/e178_attn/${TAG}.json
  CUDA_VISIBLE_DEVICES=$GPU python -u $REPO/exp/bench/e178_fa2_attn_microbench.py \
    --method $METHOD --ctx $CTX --bs $BS --reps 100 --warmup 20 \
    --out $OUT > $LOGDIR/${TAG}.log 2>&1 &
  echo "  [local] GPU=$GPU $METHOD L$((CTX/1024))K bs=$BS → $TAG (pid $!)"
}

run_remote() {
  local HOST=$1 GPU=$2 METHOD=$3 CTX=$4 BS=$5 TAG=$6
  local OUT=$REPO/exp/trace/results/e178_attn/${TAG}.json
  sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 8416 wangyuanshuo02@$HOST \
    "cd $REPO && CUDA_VISIBLE_DEVICES=$GPU python -u exp/bench/e178_fa2_attn_microbench.py \
     --method $METHOD --ctx $CTX --bs $BS --reps 100 --warmup 20 \
     --out $OUT" > $LOGDIR/${TAG}.log 2>&1 &
  echo "  [ssh] $HOST GPU=$GPU $METHOD L$((CTX/1024))K bs=$BS → $TAG (pid $!)"
}

echo "=== Dispatching E178 FA2 attn microbench (24 arms / 8 GPUs) ==="
for CTX in 32768 65536 131072 262144; do
  L=$((CTX/1024))
  run_local  0 fa2 $CTX 1 "fa2_b1_L${L}k"          # pc33 GPU0
  run_local  1 tli  $CTX 1 "tli_b1_L${L}k"         # pc33 GPU1
  run_remote 10.238.139.251 0 fa2 $CTX 4 "fa2_b4_L${L}k"    # pc40 GPU0
  run_remote 10.238.139.251 1 tli  $CTX 4 "tli_b4_L${L}k"   # pc40 GPU1
  run_remote 10.238.122.187 1 fa2 $CTX 6 "fa2_b6_L${L}k"   # pc02 GPU1
  run_remote 10.238.122.187 2 tli  $CTX 6 "tli_b6_L${L}k"   # pc02 GPU2
  run_remote 10.238.122.187 3 fa2 $CTX 1 "fa2_b1_rep_L${L}k"  # pc02 GPU3 replication
  run_remote 10.245.58.57   1 tli  $CTX 1 "tli_b1_rep_L${L}k"  # pc31 GPU1 replication
done

echo ""
echo "Dispatched 24 arms across 8 GPUs."
echo "Monitor: ls $REPO/exp/trace/results/e178_attn/*.json | wc -l"
