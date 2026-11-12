#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
RANK="$2"
WORLD_SIZE=5
EXPECTED_COMMIT=cfedb3d598ccafb11db7e3d410216c88495dc4ef
REPO=/tmp/tli-glm-fullkv-fi-run-20261112
PY=/home/fluentllmenv/bin/python
MODEL=/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash
LB_DATA=/home/wangyuanshuo02/datasets/LongBench/data
OUT=/home/wangyuanshuo02/ak_verify_results/glm_fullkv_fi_longbench16_20261112
DATASETS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
MIN_FREE_MIB=135000
MIN_AVAILABLE_MIB=70000

actual_commit="$(git -C "$REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "LB16_FI_BAD_COMMIT expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
exec 9>"/tmp/glm_fullkv_fi_lb16_rank${RANK}.lock"
if ! flock -n 9; then
    echo "LB16_FI_ALREADY_ACTIVE rank=$RANK gpu=$GPU time=$(date -Is)"
    exit 0
fi
if [[ -f "$OUT/worker_done_rank${RANK}.json" ]]; then
    echo "LB16_FI_ALREADY_DONE rank=$RANK time=$(date -Is)"
    exit 0
fi
while true; do
    free_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
    available_mib=0
    while read -r key value unit; do
        if [[ "$key" == "MemAvailable:" ]]; then
            available_mib=$((value / 1024))
            break
        fi
    done < /proc/meminfo
    if (( free_mib >= MIN_FREE_MIB && available_mib >= MIN_AVAILABLE_MIB )); then
        echo "LB16_FI_ADMIT rank=$RANK world_size=$WORLD_SIZE gpu=$GPU free_mib=$free_mib available_mib=$available_mib commit=$EXPECTED_COMMIT time=$(date -Is)"
        break
    fi
    echo "LB16_FI_WAIT rank=$RANK world_size=$WORLD_SIZE gpu=$GPU free_mib=$free_mib available_mib=$available_mib time=$(date -Is)"
    sleep 30
done

export PYTHONPATH="/home/wangyuanshuo02/.local/pylibs_tf512:$REPO:/tmp/glm_score_deps:/home/wangyuanshuo02/pylibs:/home/wangyuanshuo02/ak_verify_pylibs:/tmp/tli-pytest-deps"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIT_HEAD="$EXPECTED_COMMIT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/glm-fullkv-fi-triton-rank$RANK"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/benchmark/LongBench/pred_glm_fullkv.py" \
    --model "$MODEL" --data "$LB_DATA" --out "$OUT" \
    --backend flashinfer --samples 0 --seed 42 --datasets "$DATASETS" \
    --context-cap 31500 --device cuda:0 --rank "$RANK" --world-size "$WORLD_SIZE"
echo "LB16_FI_DONE rank=$RANK gpu=$GPU time=$(date -Is)"
