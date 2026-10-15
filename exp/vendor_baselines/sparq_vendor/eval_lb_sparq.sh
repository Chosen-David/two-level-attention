#!/bin/bash
# SparQ vendor LongBench v1 打分（消费 pred_sparq_1024/pred/ 下全部 jsonl）
# 用法：bash eval_lb_sparq.sh <output-dir> <ts>
set -eu

REPO=/home/wangyuanshuo02/two-level-attention
QVENV=$REPO/third_party/quest/.venv/bin/python
OUT=${1:-$REPO/exp/results_longbench_sparq/Qwen3-8B/pred_sparq_1024}
TS=${2:-}

cd "$REPO"
PYTHONPATH="$REPO" "$QVENV" -m benchmark.LongBench.eval \
  --model Qwen3-8B \
  --t "$TS" \
  --output-dir "$OUT" \
  --pred_postfix ""
echo "=== eval done rc=$? ==="
cat "$OUT/pred/result.json"
