#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
RANK="$2"
WORLD_SIZE="$3"
EXPECTED_IDENTITY="$4"
REPO=/tmp/tli-glm-jssa-prod-20261106
PY=/home/fluentllmenv/bin/python
MODEL=/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash
LB_DATA=/home/wangyuanshuo02/datasets/LongBench/data
OUT=/home/wangyuanshuo02/ak_verify_results/glm_longbench_16en_nope_last32_20261109
DATASETS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
MIN_FREE_MIB=135000
MIN_AVAILABLE_MIB=100000

exec 9>"/tmp/glm_longbench_16en_last32_rank${RANK}.lock"
if ! flock -n 9; then
    echo "LB16_LAST32_ALREADY_ACTIVE rank=$RANK gpu=$GPU time=$(date -Is)"
    exit 0
fi
if [[ -f "$OUT/worker_done_rank${RANK}.json" ]]; then
    echo "LB16_LAST32_ALREADY_DONE rank=$RANK time=$(date -Is)"
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
        echo "LB16_LAST32_ADMIT rank=$RANK world_size=$WORLD_SIZE gpu=$GPU free_mib=$free_mib available_mib=$available_mib time=$(date -Is)"
        break
    fi
    echo "LB16_LAST32_WAIT rank=$RANK world_size=$WORLD_SIZE gpu=$GPU free_mib=$free_mib available_mib=$available_mib time=$(date -Is)"
    sleep 30
done

export PYTHONPATH="/home/wangyuanshuo02/.local/pylibs_tf512:$REPO:/tmp/glm_score_deps:/home/wangyuanshuo02/pylibs:/home/wangyuanshuo02/ak_verify_pylibs:/tmp/tli-pytest-deps"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIT_HEAD=6760cc495f554ffa85e8eb6d6e552ce02911f457
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/tli-triton-cache-lb16-last32-rank$RANK"
export GLM_REPO="$REPO"
export GLM_EXPECT_IDENTITY="$EXPECTED_IDENTITY"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" /tmp/glm_longbench_16en_frozen_20261108.py \
    --model "$MODEL" --data "$LB_DATA" --out "$OUT" \
    --samples 0 --seed 42 --datasets "$DATASETS" \
    --context-cap 31500 --thresholds '' --attention nfa \
    --nfa-subspace glm_nope_last32 --nfa-subspace-seed 0 \
    --native-full-arm --device cuda:0 --rank "$RANK" --world-size "$WORLD_SIZE"
echo "LB16_LAST32_DONE rank=$RANK gpu=$GPU time=$(date -Is)"
