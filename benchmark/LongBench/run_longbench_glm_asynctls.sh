#!/usr/bin/env bash
# GLM-4.7-Flash × AsyncTLS × LongBench v1 单任务确定性样本分片。
# 用法：bash benchmark/LongBench/run_longbench_glm_asynctls.sh \
#   <task> <gpu_id> <rank> <world_size> <timestamp>
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <task> <gpu_id> <rank> <world_size> <timestamp>" >&2
  exit 2
fi
TASK=$1
GPU=$2
RANK=$3
WORLD=$4
TS=$5
if (( WORLD < 1 || RANK < 0 || RANK >= WORLD )); then
  echo "require 0 <= rank < world_size" >&2
  exit 2
fi

case "$TASK" in
  narrativeqa|qasper|hotpotqa|2wikimqa|musique|gov_report|qmsum|multi_news|trec|triviaqa|samsum|passage_count|passage_retrieval_en)
    SOURCE_SAMPLES=200
    ;;
  multifieldqa_en)
    SOURCE_SAMPLES=150
    ;;
  lcc|repobench-p)
    SOURCE_SAMPLES=500
    ;;
  *)
    echo "unknown LongBench v1 English task: $TASK" >&2
    exit 2
    ;;
esac
EXPECTED=$(( (SOURCE_SAMPLES - RANK + WORLD - 1) / WORLD ))

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
MODEL_PATH=${MODEL_PATH:-/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash}
DATA_DIR=${DATA_DIR:-/home/wangyuanshuo02/datasets/LongBench/data}
CHANNEL_FILE=${ASYNCTLS_CHANNEL_FILE:-$REPO_ROOT/exp/vendor_baselines/asynctls_vendor/calib/glm-4.7-flash_qasper_channels.json}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
OUT_ROOT=${OUT_ROOT:-$REPO_ROOT/exp/results_longbench_vendor/GLM-4.7-Flash/$TS/asynctls}
POSTFIX=_r${RANK}of${WORLD}
OUT_DIR=$OUT_ROOT/$TASK/pred$POSTFIX
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MIN_AVAILABLE_RAM_MIB=${MIN_AVAILABLE_RAM_MIB:-80000}

# 完整且分片身份匹配的产物才幂等跳过；partial 或错分片 fail-closed。
if compgen -G "$OUT_DIR/*.jsonl" > /dev/null; then
  "$PYBIN" - "$OUT_DIR" "$EXPECTED" "$SOURCE_SAMPLES" "$RANK" "$WORLD" <<'PY'
import glob
import hashlib
import json
import os
import sys

out_dir = sys.argv[1]
expected, source, rank, world = map(int, sys.argv[2:6])
paths = glob.glob(os.path.join(out_dir, "*.jsonl"))
if len(paths) != 1:
    raise SystemExit(f"expected exactly one shard output, found {len(paths)}: {paths}")
path = paths[0]
with open(path + ".runcap.json", encoding="utf-8") as f:
    meta = json.load(f)
shard = meta.get("sample_sharding", {})
with open(path, "rb") as f:
    raw = f.read()
lines = len(raw.splitlines())
valid = (
    meta.get("status") == "complete"
    and meta.get("method") == "asynctls_vendor"
    and meta.get("result_lines") == expected == lines
    and meta.get("n_samples") == expected
    and meta.get("result_sha256") == hashlib.sha256(raw).hexdigest()
    and shard.get("algorithm") == "source_order[rank::world_size]"
    and shard.get("rank") == rank
    and shard.get("world_size") == world
    and shard.get("source_samples") == source
    and shard.get("selected_samples") == expected
)
if not valid:
    raise SystemExit(f"existing shard is incomplete or has wrong identity: {path}")
print(f"SKIP complete shard: {path}")
PY
  exit 0
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
AVAILABLE_RAM_MIB=$(python3 -c 'print(next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable:")) // 1024)')
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "GPU $GPU free ${FREE_MIB}MiB < ${MIN_FREE_MIB}MiB; refuse to start" >&2
  exit 3
fi
if (( AVAILABLE_RAM_MIB < MIN_AVAILABLE_RAM_MIB )); then
  echo "available RAM ${AVAILABLE_RAM_MIB}MiB < ${MIN_AVAILABLE_RAM_MIB}MiB; refuse to start" >&2
  exit 3
fi

mkdir -p "$OUT_ROOT/$TASK"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/exp/vendor_baselines/asynctls_vendor:$HOME/.local/pylibs_tf512:$HOME/local_pylibs${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u -m benchmark.LongBench.pred \
  --model GLM-4.7-Flash --model_path "$MODEL_PATH" \
  --task "$TASK" --method asynctls_vendor --t "$TS" \
  --config-path "$REPO_ROOT/benchmark/LongBench/config" \
  --dataset-path "$DATA_DIR" --output-dir "$OUT_ROOT/$TASK" \
  --sample-rank "$RANK" --sample-world-size "$WORLD" \
  --pred_postfix "$POSTFIX" --nfa_enable_layer_skip false \
  --asynctls_token_budget 1024 --asynctls_block_size 64 \
  --asynctls_top_kb 128 --asynctls_channel_file "$CHANNEL_FILE"
