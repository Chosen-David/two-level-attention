#!/bin/bash
# Qwen3-32B AsyncTLS 论文语义复现：LongBench v1 英文 16 任务确定性分片。
# 用法：bash benchmark/LongBench/run_longbench_asynctls_qwen32b_per_group.sh <GPU> <RANK> <WORLD_SIZE> <TIMESTAMP>
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <GPU> <RANK> <WORLD_SIZE> <TIMESTAMP>" >&2
  exit 2
fi
GPU=$1
RANK=$2
WORLD_SIZE=$3
TS=$4
EXPECTED_HEAD=${EXPECTED_HEAD:-1cfed8481e87a62b3a705cd981049e8e699a1f81}
REPO=/home/wangyuanshuo02/two-level-attention
MODEL=/home/wangyuanshuo02/models/huggingface.co/Qwen/Qwen3-32B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
CHANNEL=$REPO/agent_doc/results/qwen32b_asynctls_per_group_20261115/calibration/qwen3_32b_qasper_per_group.json
CHANNEL_SHA256=0fc1c57319cebf6c1f6674f9c7239b33bd3d81d82211e5d73be1d5eaf6cb3cbb
OUT_ROOT=${OUT_ROOT:-/home/wangyuanshuo02/ak_verify_results/qwen32b_asynctls_lb_v1_per_group_20261115}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
TASKS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p)

cd "$REPO"
[[ $WORLD_SIZE =~ ^[1-9][0-9]*$ ]] || { echo "invalid world size" >&2; exit 2; }
[[ $RANK =~ ^[0-9]+$ ]] && (( RANK < WORLD_SIZE )) || { echo "invalid rank" >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "producer HEAD mismatch" >&2; exit 2; }
git diff --quiet HEAD -- \
  benchmark/LongBench/pred.py benchmark/LongBench/run_cap_manifest.py \
  exp/vendor_baselines/asynctls_vendor sparse_attn/info.py sparse_attn/arguments.py || {
  echo "AsyncTLS runtime has uncommitted changes" >&2
  exit 2
}
[[ $(sha256sum "$CHANNEL" | cut -d' ' -f1) == "$CHANNEL_SHA256" ]] || {
  echo "channel identity mismatch" >&2
  exit 2
}
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pred_r${RANK}"

complete_file() {
  local task=$1
  local count=$2
  local expected=$(( (count - RANK + WORLD_SIZE - 1) / WORLD_SIZE ))
  local prefix=${task%%-*}
  python - "$OUT_ROOT/pred_r${RANK}" "$prefix" "$TS" "$expected" "$RANK" "$WORLD_SIZE" "$EXPECTED_HEAD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

folder, prefix, timestamp, expected, rank, world, head = sys.argv[1:]
expected = int(expected)
path = Path(folder)
pattern = f"{prefix}-asynctls_vendor_qwen3_32b_qasper_per_group_k1024_b64_kb128*-{timestamp}.jsonl"
candidates = sorted(path.glob(pattern))
if not candidates:
    raise SystemExit(1)
if len(candidates) != 1:
    print(f"multiple outputs match {pattern}: {candidates}", file=sys.stderr)
    raise SystemExit(2)
result = candidates[0]
sidecar = Path(str(result) + ".runcap.json")
if not sidecar.is_file():
    raise SystemExit(1)
receipt = json.loads(sidecar.read_text())
sharding = receipt.get("sample_sharding", {})
producer = receipt.get("producer", {})
if receipt.get("status") != "complete" or receipt.get("n_samples") != expected:
    raise SystemExit(1)
if receipt.get("result_lines") != expected:
    raise SystemExit(1)
if sharding.get("rank") != int(rank) or sharding.get("world_size") != int(world):
    raise SystemExit(2)
if sharding.get("selected_samples") != expected:
    raise SystemExit(2)
if producer.get("git_commit") != head:
    print(f"producer mismatch: {producer.get('git_commit')} != {head}", file=sys.stderr)
    raise SystemExit(2)
lines = result.read_bytes().splitlines()
if len(lines) != expected:
    raise SystemExit(1)
sha256 = hashlib.sha256(result.read_bytes()).hexdigest()
if receipt.get("result_sha256") != sha256:
    raise SystemExit(1)
print(result)
PY
}

declare -A COUNTS=(
  [narrativeqa]=200 [qasper]=200 [multifieldqa_en]=150 [hotpotqa]=200
  [2wikimqa]=200 [musique]=200 [gov_report]=200 [qmsum]=200 [multi_news]=200
  [trec]=200 [triviaqa]=200 [samsum]=200 [passage_count]=200
  [passage_retrieval_en]=200 [lcc]=500 [repobench-p]=500
)

for TASK in "${TASKS[@]}"; do
  COUNT=${COUNTS[$TASK]}
  EXPECTED=$(( (COUNT - RANK + WORLD_SIZE - 1) / WORLD_SIZE ))
  if OUTPUT=$(complete_file "$TASK" "$COUNT"); then
    echo "[SKIP COMPLETE] task=$TASK rank=$RANK/$WORLD_SIZE samples=$EXPECTED output=$OUTPUT"
    continue
  else
    RC=$?
    if [[ $RC -eq 2 ]]; then
      echo "[FAIL IDENTITY] task=$TASK rank=$RANK/$WORLD_SIZE" >&2
      exit "$RC"
    fi
  fi
  echo "[START] task=$TASK rank=$RANK/$WORLD_SIZE expected=$EXPECTED gpu=$GPU"
  set +e
  PYTHONPATH=/home/wangyuanshuo02/local_pylibs CUDA_VISIBLE_DEVICES=$GPU \
    "$PYBIN" -u -m benchmark.LongBench.pred \
    --model Qwen3-32B --model_path "$MODEL" \
    --task "$TASK" --method asynctls_vendor --t "$TS" \
    --config-path "$REPO/benchmark/LongBench/config" --dataset-path "$DATA" \
    --output-dir "$OUT_ROOT" --pred_postfix "_r${RANK}" \
    --sample-rank "$RANK" --sample-world-size "$WORLD_SIZE" \
    --asynctls_channel_file "$CHANNEL" \
    --asynctls_token_budget 1024 --asynctls_block_size 64 --asynctls_top_kb 128 \
    --nfa_enable_layer_skip false \
    2>&1 | tee -a "$OUT_ROOT/logs/${TASK}_r${RANK}.log"
  RC=${PIPESTATUS[0]}
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "[FAIL] task=$TASK rank=$RANK/$WORLD_SIZE rc=$RC" >&2
    exit "$RC"
  fi
  if ! OUTPUT=$(complete_file "$TASK" "$COUNT"); then
    echo "[FAIL INCOMPLETE] task=$TASK rank=$RANK/$WORLD_SIZE expected=$EXPECTED" >&2
    exit 3
  fi
  echo "[DONE] task=$TASK rank=$RANK/$WORLD_SIZE samples=$EXPECTED output=$OUTPUT"
done

echo "[CHAIN DONE] rank=$RANK/$WORLD_SIZE timestamp=$TS"
