#!/usr/bin/env bash
# Qwen3-8B × LongBench v2：Quest/JSSA 官方实现与 AsyncTLS 论文语义复现。
# 用法：bash benchmark/LongBench/run_longbench_v2_baselines.sh \
#   <quest|jssa|asynctls> <gpu_id> <rank> <world_size> <timestamp>
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <quest|jssa|asynctls> <gpu_id> <rank> <world_size> <timestamp>" >&2
  exit 2
fi
ARM=$1
GPU=$2
RANK=$3
WORLD=$4
TS=$5
if (( WORLD < 1 || RANK < 0 || RANK >= WORLD )); then
  echo "require 0 <= rank < world_size" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
MODEL_PATH=${MODEL_PATH:-/home/wangyuanshuo02/Qwen3-8B}
LBV2_DATA=${LBV2_DATA:-/home/wangyuanshuo02/datasets/LongBench/data/lbv2.json}
JSSA_BASIS_PATH=${JSSA_BASIS_PATH:-$REPO_ROOT/bases/qwen3_8b.pt}
ASYNCTLS_CHANNEL_FILE=${ASYNCTLS_CHANNEL_FILE:-$REPO_ROOT/dimension_channels/qwen3-8b/qasper_channels_per_group.json}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
OUT_ROOT=${OUT_ROOT:-$REPO_ROOT/exp/results_longbench_v2/Qwen3-8B/$TS/$ARM}
POSTFIX=_r${RANK}of${WORLD}
OUT_DIR=$OUT_ROOT/pred$POSTFIX
EXPECTED=$(( (503 - RANK + WORLD - 1) / WORLD ))

case "$ARM" in
  quest)
    METHOD=quest_vendor_kernel
    METHOD_ARGS=(
      --quest_token_budget 1024 --quest_chunk_size 16
      --quest_page_size 16 --quest_max_seq_len 40960
    )
    ;;
  jssa)
    METHOD=jssa_official
    METHOD_ARGS=(
      --jssa_budget 1024 --jssa_rank 32 --jssa_basis_path "$JSSA_BASIS_PATH"
    )
    ;;
  asynctls)
    METHOD=asynctls_vendor
    METHOD_ARGS=(
      --asynctls_token_budget 1024 --asynctls_block_size 64
      --asynctls_top_kb 128 --asynctls_channel_file "$ASYNCTLS_CHANNEL_FILE"
    )
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 2
    ;;
esac

# 完整且身份匹配的片才幂等跳过；partial/多文件/错分片一律 fail-closed。
if compgen -G "$OUT_DIR/*.jsonl" > /dev/null; then
  "$PYBIN" - "$OUT_DIR" "$EXPECTED" "$RANK" "$WORLD" <<'PY'
import glob
import json
import os
import sys

out_dir, expected, rank, world = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
paths = glob.glob(os.path.join(out_dir, "*.jsonl"))
if len(paths) != 1:
    raise SystemExit(f"expected exactly one shard output, found {len(paths)}: {paths}")
path = paths[0]
with open(path + ".runcap.json", encoding="utf-8") as f:
    meta = json.load(f)
shard = meta.get("sample_sharding", {})
lines = sum(1 for _ in open(path, encoding="utf-8"))
valid = (
    meta.get("status") == "complete"
    and meta.get("result_lines") == expected == lines
    and meta.get("n_samples") == expected
    and shard.get("rank") == rank
    and shard.get("world_size") == world
    and shard.get("selected_samples") == expected
)
if not valid:
    raise SystemExit(f"existing shard is incomplete or has wrong identity: {path}")
print(f"SKIP complete shard: {path}")
PY
  exit 0
fi

mkdir -p "$OUT_ROOT"
export PYTHONPATH="$REPO_ROOT:$HOME/local_pylibs${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u -m benchmark.LongBench.pred \
  --model Qwen3-8B --model_path "$MODEL_PATH" \
  --task lbv2 --method "$METHOD" --t "$TS" \
  --config-path "$REPO_ROOT/benchmark/LongBench/config" \
  --dataset-path "$LBV2_DATA" --output-dir "$OUT_ROOT" \
  --sample-rank "$RANK" --sample-world-size "$WORLD" \
  --pred_postfix "$POSTFIX" --nfa_enable_layer_skip false \
  "${METHOD_ARGS[@]}"
