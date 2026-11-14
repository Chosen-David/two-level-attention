#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
RANK="${2:?rank required}"
WORLD_SIZE="${3:?world size required}"
EXPECTED_IDENTITY="${4:?expected identity required}"
REPO=/tmp/tli-glm-nfa-eps010-graph-20261112-r2
PY=/home/fluentllmenv/bin/python
MODEL=/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash
LB_DATA=/home/wangyuanshuo02/datasets/LongBench/data
OUT="${OUT:-/home/wangyuanshuo02/ak_verify_results/glm_longbench_16en_nfa_eps010_graph_20261112_r2}"
DATASETS="${DATASETS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
SAMPLES="${SAMPLES:-0}"
ROW_INDEX="${ROW_INDEX:-}"
RUN_TAG="${RUN_TAG:-lb16r2}"
MIN_FREE_MIB=135000
MIN_AVAILABLE_MIB=100000

exec 9>"/tmp/glm_nfa_eps010_graph_${RUN_TAG}_rank${RANK}.lock"
if ! flock -n 9; then
    echo "GLM_NFA_ALREADY_ACTIVE tag=$RUN_TAG rank=$RANK gpu=$GPU time=$(date -Is)"
    exit 0
fi
if [[ -f "$OUT/worker_done_rank${RANK}.json" ]]; then
    echo "GLM_NFA_ALREADY_DONE tag=$RUN_TAG rank=$RANK time=$(date -Is)"
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
        echo "GLM_NFA_ADMIT tag=$RUN_TAG rank=$RANK world_size=$WORLD_SIZE gpu=$GPU free_mib=$free_mib available_mib=$available_mib time=$(date -Is)"
        break
    fi
    echo "GLM_NFA_WAIT tag=$RUN_TAG rank=$RANK world_size=$WORLD_SIZE gpu=$GPU free_mib=$free_mib available_mib=$available_mib time=$(date -Is)"
    sleep 30
done

export PYTHONPATH="/home/wangyuanshuo02/.local/pylibs_tf512:$REPO:/tmp/glm_score_deps:/home/wangyuanshuo02/pylibs:/home/wangyuanshuo02/ak_verify_pylibs:/tmp/tli-pytest-deps"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIT_HEAD=7a0f6de8fa4eea1d0e216aaef7b5a8c9275a44fb
export CODE_DIFF_SHA256=cd3d83b1e9c02ebd9b8176e478ae57a2dac640137c8380297b4846c0da3c5b85
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/glm-nfa-eps010-graph-${RUN_TAG}-rank$RANK"
export GLM_REPO="$REPO"
export GLM_EXPECT_IDENTITY="$EXPECTED_IDENTITY"

CMD=(
    --model "$MODEL" --data "$LB_DATA" --out "$OUT"
    --samples "$SAMPLES" --seed 42 --datasets "$DATASETS"
    --context-cap 31500 --thresholds 0.1 --gate-modes dual
    --skip-full-arm --attention nfa
    --nfa-subspace glm_nope_last32 --nfa-subspace-seed 0
    --device cuda:0 --rank "$RANK" --world-size "$WORLD_SIZE"
)
if [[ -n "$ROW_INDEX" ]]; then
    CMD+=(--row-index "$ROW_INDEX")
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PY" /tmp/glm_nfa_eps010_graph_frozen_20261112.py "${CMD[@]}"
echo "GLM_NFA_DONE tag=$RUN_TAG rank=$RANK gpu=$GPU time=$(date -Is)"
