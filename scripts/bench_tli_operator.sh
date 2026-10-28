#!/usr/bin/env bash
# Paired whole-model graph runs. Requires an existing local model and GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the existing local Qwen3 model directory}"
[[ -d "$MODEL_PATH" ]] || { echo 'MODEL_PATH is not a directory' >&2; exit 2; }
[[ -z "${TLI_SHADOW_PKG:-}" ]] || { echo 'Unset TLI_SHADOW_PKG for an auditable comparison' >&2; exit 2; }
python_bin="${PYTHON:-python}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TLI_OPERATOR_BACKEND="${TLI_OPERATOR_BACKEND:-simt}"
export TLI_OPERATOR_SPLITS="${TLI_OPERATOR_SPLITS:-4}"
export TLI_PROTECTED_SPLITS="${TLI_PROTECTED_SPLITS:-1}"
export TLI_BATCH_INDEXER=legacy
export TLI_L2_BYPASS=0
export TLI_SPARSE_PREFILL=0
out_dir="${OUT_DIR:-results/operator-overlap-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out_dir"
bash scripts/test_tli_operator.sh --cuda 2>&1 | tee "$out_dir/correctness.log"
"$python_bin" - "$out_dir/environment.json" <<'PY'
import json, os, subprocess, sys, torch, triton
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True),
    python=sys.version,torch=torch.__version__,triton=triton.__version__,
    gpu=torch.cuda.get_device_name(),
    environment={k:v for k,v in os.environ.items() if k.startswith('TLI_') or k in ('MODEL_PATH','CONTEXT_LENGTH','MAX_GEN')}
),indent=2)+'\n')
PY
for mode in off serial overlap; do
    TLI_OPERATOR_MODE="$mode" "$python_bin" exp/bench/e152v4_graph_latency.py \
        --model-path "$MODEL_PATH" --context-length "${CONTEXT_LENGTH:-32768}" \
        --max-gen "${MAX_GEN:-128}" --repeats 3 --warmup 1 --graph-mode graph \
        --method tli --tli_far_method minmax --tli_near_method avg \
        --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 \
        --tli_subspace tail --tia_level2_cmp_ratio 4 \
        --tli_sink_tokens 4 --tli_swa_size 128 --tia_level1_topk 128 --tia_level2_topk 1024 \
        --tli_far_skip_backend compact --tli_far_skip_cache evict \
        --label "operator-$mode-$TLI_OPERATOR_BACKEND" --out "$out_dir/$mode.json" \
        2>&1 | tee "$out_dir/$mode.log"
done
echo "Saved raw paired results in $out_dir. Inspect selected sets, generated tokens and TPOT; no automatic speedup verdict."
